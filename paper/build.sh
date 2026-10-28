#!/bin/bash
# Build all submission PDFs and Word versions. Run from anywhere: bash paper/build.sh
set -e
cd "$(dirname "$0")"
(cd figures && pdflatex -interaction=nonstopmode framework.tex >/dev/null && pdftoppm -png -r 300 -singlefile framework.pdf framework)
(cd figures && pdflatex -interaction=nonstopmode graphical_abstract.tex >/dev/null && pdftoppm -png -scale-to-x 2000 -scale-to-y -1 -singlefile graphical_abstract.pdf graphical_abstract)
# manuscript, supplementary and response cross-reference each other (xr), so build them twice
for pass in 1 2; do
  for f in manuscript supplementary manuscript_highlighted response_to_reviewers; do
    latexmk -pdf -interaction=nonstopmode -quiet "$f.tex" >/dev/null 2>&1 || echo "LaTeX failed: $f (pass $pass)"
    [ $pass = 1 ] && latexmk -pdf -g -interaction=nonstopmode -quiet "$f.tex" >/dev/null 2>&1
  done
done
for f in cover_letter highlights title_page review/internal_review; do
  (cd "$(dirname $f)" && latexmk -pdf -interaction=nonstopmode -quiet "$(basename $f).tex" >/dev/null 2>&1) || echo "LaTeX failed: $f"
done

# Word versions: flatten \input, use PNG figures, numeric Elsevier citations.
mkdir -p word
for f in manuscript supplementary; do
  python3 - "$f" <<'EOF'
import re, sys, pathlib
name = sys.argv[1]
src = pathlib.Path(f"{name}.tex").read_text()
def flatten(t):
    return re.sub(r"\\input\{([^}]*)\}", lambda m: flatten(pathlib.Path(m.group(1) + ("" if m.group(1).endswith(".tex") else ".tex")).read_text()), t)
t = flatten(src)
t = re.sub(r"(figures/[A-Za-z0-9_]+)\.pdf", r"\1.png", t)
# resolve cross-document (xr) references, which pandoc cannot follow
for pre, aux in (("S-", "supplementary.aux"), ("M-", "manuscript.aux")):
    labels = dict(re.findall(r"\\newlabel\{([^}]*)\}\{\{([^}]*)\}", pathlib.Path(aux).read_text())) if pathlib.Path(aux).exists() else {}
    t = re.sub(r"\\eqref\{" + pre + r"([^}]*)\}", lambda m: "(" + labels.get(m.group(1), "?") + ")", t)
    t = re.sub(r"\\ref\{" + pre + r"([^}]*)\}", lambda m: labels.get(m.group(1), "?"), t)
t = t.replace(r"\resizebox{\textwidth}{!}{", r"{")
t = re.sub(r"\\linenumbers", "", t)
t = re.sub(r"% Revision marking.*?\n\\fi\n", "", t, flags=re.S)
t = re.sub(r"\\revnote\{[^}]*\}", "", t)
t = t.replace("\\rev{", "{")
t = t.replace(r"\ifdefined\HIGHLIGHT\color{blue}\fi", "")
if name == "manuscript":
    title = re.search(r"\\title\{(.*?)\}\n", t, re.S).group(1)
    absr = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", t, re.S).group(1)
    kw = re.search(r"\\begin\{keyword\}(.*?)\\end\{keyword\}", t, re.S).group(1).replace("\\sep", ";")
    front = ("\\title{" + title + "}\n\\author{P. N. Bhargav Teja$^{a}$, Lanka Sree Chathurya$^{a}$, Rajay Vedaraj I.S.$^{a,*}$\\\\ "
             "$^{a}$Vellore Institute of Technology, Vellore 632014, Tamil Nadu, India\\\\ "
             "$^{*}$Corresponding author: rajay@vit.ac.in}\n\\date{}\n")
    t = re.sub(r"\\begin\{frontmatter\}.*?\\end\{frontmatter\}",
               lambda m: "\\maketitle\n\\begin{abstract}" + absr + "\\end{abstract}\n\\textbf{Keywords:} " + kw.strip() + "\n",
               t, flags=re.S)
    t = t.replace("\\begin{document}", front + "\\begin{document}", 1)
pathlib.Path(f"word/{name}_flat.tex").write_text(t)
EOF
  pandoc "word/${f}_flat.tex" -o "word/${f}.docx" --citeproc --bibliography=references.bib \
    --csl=elsevier-with-titles.csl --resource-path=.:figures -M link-citations=true 2>/dev/null || echo "pandoc failed: $f"
done
for f in cover_letter highlights; do
  python3 - "$f" <<'EOF'
import re, sys, pathlib
name = sys.argv[1]
t = pathlib.Path(f"{name}.tex").read_text()
t = t.replace(r"\input{numbers}", pathlib.Path("numbers.tex").read_text())
pathlib.Path(f"word/{name}_flat.tex").write_text(t)
EOF
  pandoc "word/${f}_flat.tex" -o "word/${f}.docx" 2>/dev/null || echo "pandoc failed: $f"
done
rm -f word/*_flat.tex
echo "built"
