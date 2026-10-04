# Submission guide

## Files to upload

| File | Purpose |
|---|---|
| `manuscript.pdf` / `word/manuscript.docx` (+ LaTeX sources `manuscript.tex`, `sections/`, `figures/`, `tables/`, `numbers.tex`, `references.bib`) | Main manuscript (Elsevier `elsarticle`, review format with line numbers) |
| `supplementary.pdf` / `word/supplementary.docx` | Supplementary material |
| `cover_letter.pdf` / `word/cover_letter.docx` | Cover letter to the editor |
| `highlights.pdf` / `word/highlights.docx` | Highlights (5 bullets, each ≤ 85 characters) |
| `title_page.pdf` | Title page with author details (needed for double-anonymous journals) |
| `figures/*.pdf` (vector) and `figures/*.png` (300 dpi) | Separate figure files if the journal asks for them |
| `review/internal_review.pdf` | **Internal only. Do not upload.** Simulated reviewer reports and the responses to them |

## Recommended journals (Q1, not MDPI)

Review speeds below are the journals' own published medians or third-party estimates. They change over time and are not guarantees. **No Q1 journal guarantees acceptance within 60 days.** The first option is the closest match to that target that I found.

| Journal | Publisher | Quartile | Reported speed | Fit | Notes |
|---|---|---|---|---|---|
| **Results in Engineering** (primary target) | Elsevier, open access | JCR Q1 (IF ≈ 7.9) | ~6 days to first decision, ~32 days to decision after review, **~76 days median to acceptance** | Broad engineering + AI; welcomes rigorous simulation studies | APC ≈ USD 2,160. The manuscript is already formatted for it. |
| Machine Learning with Applications | Elsevier, open access | Q1/Q2 (SJR) | ~89–102 days to acceptance | Applied ML | Faster than most AI journals |
| Intelligent Systems with Applications | Elsevier, open access | JCR Q1–Q2 by category | ~131 days to acceptance | Intelligent decision systems | APC ≈ USD 1,500 |
| Engineering Science and Technology, an International Journal | Elsevier, open access | SJR Q1 | ~112–183 days | Engineering applications | |
| Computers & Industrial Engineering | Elsevier | Q1 | ~208 days to acceptance | Best topical fit (human-centric scheduling) | Slower, but highly relevant readership |
| IEEE Access (fallback for speed) | IEEE, open access | **JCR Q2** (IF ≈ 3.6) | ~4–6 weeks to decision | Broad | Fast, but not Q1 in JCR |

Sources: Results in Engineering insights page (sciencedirect.com/journal/results-in-engineering/about/insights); journalmetrics.org; journal pages for ISWA, MLWA, JESTECH, C&IE; IEEE Access "Rapid Peer Review" page. Check the current figures on each journal's "Journal insights" page before you submit.

## Checklist before submission: items only the authors can confirm

1. **Affiliations and e-mails.** The author block lists "Vellore Institute of Technology, Vellore 632014". Add each author's school or department and the two student e-mail addresses (`title_page.tex`), and confirm that the corresponding author is Dr. Rajay Vedaraj I.S. (rajay@vit.ac.in).
2. **Author order and CRediT roles.** Check the contribution statement at the end of `manuscript.tex`.
3. **Data availability.** The manuscript points to `https://github.com/bhargavteja-9779/rl`. Make the repository public (or create a Zenodo DOI) before submission.
4. **Generative-AI declaration.** Elsevier requires one, and the manuscript includes it. Keep it, and make sure every author has read and approved the full text.
5. **Funding and acknowledgements.** Edit if any grant or institutional support applies.
6. **ORCID iDs.** Add them in the submission system.
7. **Suggested reviewers.** Many Elsevier journals ask for 3–5. Choose researchers who work on human factors in scheduling or DRL for scheduling and who are not collaborators.

## Honest summary of the findings (for the authors)

- The pre-specified agent **matches** the best of nine tuned rules on return (not significantly different) and completes **more tasks on time**.
- The compact intensity-only agent, a secondary analysis with 10 seeds, **beats** the best rule on return, on-time rate and difficulty-weighted on-time rate (p < 0.001). Because it was identified in a component study first, the paper presents it as secondary.
- The learned agents are robust to sensor noise, and their advantage is large when cognitive-state estimates are accurate. They generalise poorly to unseen workloads (8 or 16 tasks per day). The paper states this as a limitation.
- The earlier draft's claims (e.g. the 48% variance reduction, the "Figure X" bar chart, and "RL helps only when cognition is modelled") were not reproducible and have been replaced by results computed from the released code.
