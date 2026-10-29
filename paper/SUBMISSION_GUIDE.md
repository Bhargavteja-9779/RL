# Revision guide: Array, ARRAY-D-26-01253 (due 17 Nov 2026)

## Files to upload in Editorial Manager

| Editorial Manager item | File |
|---|---|
| Response to reviewers (rebuttal letter) | `response_to_reviewers.pdf` |
| Revised manuscript, changes marked (highlighted version) | `manuscript_highlighted.pdf` |
| Revised manuscript, clean version | `manuscript.pdf`, plus LaTeX sources (`manuscript.tex`, `sections/`, `figures/`, `tables/`, `numbers.tex`, `references.bib`) or `word/manuscript.docx` |
| Cover letter | `cover_letter.pdf` (or `word/cover_letter.docx`) |
| Graphical abstract | `figures/graphical_abstract.png` (2000 × 707 px), or `figures/graphical_abstract.pdf` |
| Highlights | `highlights.pdf` (or `word/highlights.docx`): 5 bullets, each ≤ 85 characters |
| Supplementary material | `supplementary.pdf` (or `word/supplementary.docx`) |
| Title page, if requested | `title_page.pdf` |
| Do not upload | `review/internal_review.pdf` (internal simulated review from the first submission) |

## How the highlighted version works

Almost the whole manuscript was rewritten after the first submission, so a word-level diff would mark nearly every sentence. In `manuscript_highlighted.pdf`, each passage that answers a reviewer or editor comment is printed in **blue**, and a margin tag names the comment it answers (e.g. `R4.3`, `Ed.`). The response letter explains this choice. To rebuild it: `\def\HIGHLIGHT{1}\input{manuscript}` (this is the whole of `manuscript_highlighted.tex`). Wrap any further new text in `\rev{...}` and add `\revnote{R…}`.

## Before you upload: points only the authors can confirm

1. **Author list.** The editor does not allow adding or removing authors. The manuscript lists P. N. Bhargav Teja, Lanka Sree Chathurya and Rajay Vedaraj I.S. (corresponding author). Check that this matches the original Array submission exactly, including the order.
2. **Data availability.** Make `https://github.com/bhargavteja-9779/rl` public, or create a Zenodo DOI, before resubmitting.
3. **AI declaration.** The manuscript discloses the use of an AI assistant for code development, language editing and organisation. Keep the declaration and make sure all authors have read and approved the revised text.
4. **Affiliations, e-mails and ORCID iDs.** Add these in `title_page.tex` and in Editorial Manager.

## Rebuilding everything

```
PYTHONPATH=. python3 -m analysis.analyze   # statistics, tables, figures, numbers.tex
bash paper/build.sh                        # all PDFs and Word files
```
Every number quoted in the text comes from `numbers.tex`, so the prose, tables and figures cannot drift apart.

## Summary of the revised findings

- Rules that never rest perform poorly (EDF: 46.8% of tasks on time). Tuned rest rules are strong; the best is SPT-Threshold (return 12.30, 71.4% on time).
- AA-DQN (hybrid actions) ties with SPT-Threshold on return and completes more tasks on time (72.8%).
- AA-PPO, the policy-gradient learner added at Reviewer 4's request with untuned default settings, beats every rule. Against SPT-Threshold: return 13.04 (+0.73, 95% CI [0.60, 0.87]), 75.1% on time (+3.7 pp) and difficulty-weighted on-time rate +7.0 pp. All ten seeds beat the rule on each of these.
- AA-PPO stays the best method under sensor noise, for both worker profiles, with 16 tasks per day, and in all 14 models where one cognitive parameter is changed by ±30%. With 8 tasks per day the EDF rules are slightly better. AA-DQN generalises poorly to unseen workloads.
- The original claim that "RL helps only when cognition is modelled" was contradicted by the controlled ablations and is retracted in the paper and in the response.
