# Experiments

Analyses that were run during the study and left out of the paper. They are not
part of the pipeline that produces the PDF: nothing in `scripts/` imports them,
and no value in the manuscript depends on them. They are here because each one
was used to decide something, and a reader who wants to check that decision
should not have to reconstruct the code.

Each script reads the same inputs as the main analysis and takes an optional
export path.

| Script | What it was used for | Why it is not in the paper |
|---|---|---|
| `dropout_analysis.py` | Response funnel, and completers vs mid-survey dropouts on the page-1 context variables. | Backs the external-validity paragraph in `chapters/discussion.tex`, which states the direction of the bias in words. At n = 88 the domain-breadth contrast is too thin to carry a printed robustness claim. |
| `factor_structure.py` | Exploratory factor analysis of the five Want objective items, testing the environmental / operational split. | The split is carried by prior work and by the reliability figures in Section IV-B. The EFA agreed but added a KMO caveat, so the paper reports the reliability evidence instead. |
| `outer_model.py` | Outer weights and loadings for the formative blocks, with bootstrap intervals. | The paper reports VIF for those blocks and has no measurement-model table. |
| `extra_macros.py` | Every value the robustness report prints that the paper does not: the PLS-SEM paths and their bootstrap intervals, the alternative specifications, the per-variant sensitivity correlations, the ordinal-logit intervals, the early-versus-late responder comparison, and the power figure behind the discriminant comparison. | The manuscript prints none of them. Keeping them out of `gen_latex.py` is what keeps the generator that feeds the PDF free of the PLS stack and the `.npz` caches. |
| `gen_robustness_report.py` | The seven robustness checks, readable without the paper. Renders macros — computes nothing of its own. | The paper has no robustness section. The checks constrain how far its conclusions carry, and this is where they are reported in full. |

`extra_macros.py` extends the registry `gen_latex.build()` returns, so the
report's shared values are the ones the PDF prints, and it derives everything
else from the same functions in `hypothesis_test.py` and the same specification
lists in `config.py`. It reads three committed caches in `scripts/` and refuses
any whose fingerprint does not match the current export.

`outer_model.py` writes `outer_model_cache.npz` beside itself; `dropout_analysis.py`
and `factor_structure.py` compute in seconds and cache nothing.

```bash
cd replication
python experiments/dropout_analysis.py
python experiments/factor_structure.py
python experiments/outer_model.py --status
python experiments/extra_macros.py             # the report-only values, as a table
python experiments/gen_robustness_report.py    # -> reports/robustness_report.txt
```
