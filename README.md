# Replication Package for "Green Software Metrics: An Enterprise Survey"

This repository contains the full replication package accompanying the paper:

**"Green Software Metrics: An Enterprise Survey"**
Submitted to Information and Software Technology (Elsevier), Greenvolve Special Issue, 2026.

Everything needed to reproduce the analysis. Self-contained: nothing here is generated from the
manuscript, and nothing here is consumed by it except the LaTeX macro file.

```
replication/
  scripts/       the analysis — start at config.py
  data/          survey export and questionnaire instrument (inputs)
  reports/       generated artifacts, committed so the paper can cite them
  experiments/   analyses that were run and left out of the paper
```

Every analysis reads one input, `data/results-survey668719.csv`, and one
definitions file, `scripts/config.py`. No number in the paper is typed by hand.

## Setup

```bash
pip install -r requirements.txt
```

Python 3.10+. The pandas upper bound in `requirements.txt` is load-bearing —
read the comment there before relaxing it.

## Inputs (`data/`)

| File | What it is |
|---|---|
| `results-survey668719.csv` | LimeSurvey export, completed responses. The input to every analysis. |
| `results-survey668719-all.csv` | The same export including incomplete responses. Read only by `experiments/dropout_analysis.py`. |
| `Questionnaire.xlsx` | The questionnaire as administered. |
| `limesurvey_survey_668719.lss` | LimeSurvey definition of the instrument. |
| `limesurvey_survey_668719.mmd`, `SurveyFlowchart.md` | Questionnaire flowchart. Both generated from the `.lss`. |

## Outputs (`reports/`)

Committed rather than gitignored: the paper cites them, and a diff after a data
cut shows which findings moved. Regenerate rather than editing by hand.

| File | Written by |
|---|---|
| `hypothesis_report.txt` | `hypothesis_test.py` |
| `descriptive_report.txt`, `descriptive_plots.png` | `descriptive_analysis.py` |
| `robustness_report.txt` | `experiments/gen_robustness_report.py` |
| `HypothesisTestOverview.md` | `gen_overview.py` |

## Code (`scripts/`)

**Definitions.** Edit here, and every output follows.

| File | Role |
|---|---|
| `config.py` | Column indices, Likert maps, scoring rules, group thresholds, significance levels, robustness specs. The single source of truth. |

**Analysis.**

| File | Role | Writes |
|---|---|---|
| `hypothesis_test.py` | Builds all dimension scores and context groups, runs H1–H5 and H8 (Mann-Whitney, Spearman, Holm). Exports the scoring functions every script below reuses. | `reports/hypothesis_report.txt` |
| `descriptive_analysis.py` | Per-dimension descriptives and scale reliability (Cronbach α, Spearman-Brown). | `reports/descriptive_report.txt` |
| `pls_bootstrap.py` | Defines the PLS-SEM — measurement model, structural model — and its seeded, resumable bootstrap. Every other PLS script imports or mirrors this one. | `boot_cache_b5000.npz` |
| `pls_sem.py` | The same model, fitted once and printed in full: outer model, reliability, VIF, R², GoF. | stdout |
| `pls_alternatives.py` | Alternative specifications behind the Context/*Do* construct-overlap check. | `pls_alt_cache.npz` |
| `validate_rho_diff.py` | Validation and power analysis for the dependent-correlation comparison. | `rho_diff_power_cache.npz` |

**Generators.** Never edit their output by hand; rerun the generator.

| File | Role | Writes |
|---|---|---|
| `gen_latex.py` | Every data-derived number in the paper, as LaTeX macros — and nothing else. A value the manuscript does not print is not computed here. | `../chapters/generated/macros.tex` |
| `gen_overview.py` | Hypothesis reference table. | `reports/HypothesisTestOverview.md` |
| `gen_figures.py` | The ranked item profiles of Section V-A, as TikZ bar panels. | `../figures/generated/fig_*.tex` |

**Support.**

| File | Role | Writes |
|---|---|---|
| `limesurvey2mermaid.py` | Questionnaire flowchart from the `.lss` export. Standard library only. | `data/*.mmd`, `data/SurveyFlowchart.md` |

## How they interconnect

Solid arrows are imports; dotted arrows are `.npz` caches on disk.

```
config.py
   └─► hypothesis_test.py ──► gen_overview.py
          │  scoring functions reused by everything below
          ├─► descriptive_analysis.py
          ├─► pls_bootstrap.py ──► pls_sem.py
          ├─► pls_alternatives.py
          └─► validate_rho_diff.py

       gen_latex.py       gen_figures.py

   experiments/extra_macros.py ──► experiments/gen_robustness_report.py
            ▲   └─► extends the registry gen_latex.build() returns
            ┆ boot_cache_b5000.npz        (pls_bootstrap.py)
            ┆ pls_alt_cache.npz           (pls_alternatives.py)
            ┆ rho_diff_power_cache.npz    (validate_rho_diff.py)
```

Change a definition once in `config.py` and every output stays consistent. The
same applies to the robustness outputs: the eight scorings of *Do*
(`DO_VARIANT_SPECS`) and the ordinal-logit specifications (`ORD_SPECS`) live in
`config.py`, and the report renders macros rather than computing its own — the
ones it shares with the paper come from `gen_latex.build()` itself, so a value
in the standalone report cannot disagree with the same value in the PDF.

**Bootstraps.** Three quantities the paper prints rest on a resampling loop:
the Spearman intervals (`spearman_ci`, 10,000 draws each), the rank effect-size
intervals (`mannwhitney_r_ci`, 5,000), and the dependent-correlation comparison
(`dependent_rho_diff`, 10,000). All three draw their resamples from a named
seed and evaluate the statistic a batch at a time rather than calling
`scipy.stats` once per draw. The block draw is the same random stream as the
per-draw one, and the batched statistic makes the same `rankdata` and
`np.corrcoef` calls scipy makes, so every interval is bit-for-bit what the
per-draw loop produced. Regenerating the macros takes about twelve seconds.

**Where a number lives.** `gen_latex.py` computes exactly what the manuscript
prints. Values that exist only for the robustness report are added by
`experiments/extra_macros.py`, which extends the same registry; values nothing
reads are commented out in `gen_latex.py` beside the code that produced them.
That is what keeps the generator on the critical path fast: the PLS-SEM fit,
the bootstrap caches and the power simulation are all report-only, so none of
them runs when the paper is regenerated.

**Hypothesis numbering.** The `run_*` functions in `hypothesis_test.py` carry a
historical offset (`run_h3` is the paper's H4, and so on), documented in that
file's header. Macro names are relationship-based (`gsmWantDo…`, `gsmCtxDo…`)
rather than numbered, and `gen_overview.py` maps to the paper's numbers in a
single place, so the offset never reaches an output.

## The caches

The three `.npz` files hold resamples and simulations that take roughly a
quarter of an hour to rebuild, so they ship with the package. Each carries a
SHA-1 fingerprint of the data frame it was computed from.
`experiments/extra_macros.py` refuses any cache whose fingerprint does not
match the current export, and refuses one carrying no fingerprint at all, since
that cannot be checked. A stale cache therefore stops the report instead of
quietly reaching it. No cache feeds the paper: `gen_latex.py` reads none.

## Running them

Each script takes an optional export path and otherwise uses the default.

```bash
cd replication/scripts
python hypothesis_test.py
python descriptive_analysis.py
python gen_latex.py
python gen_overview.py
python gen_figures.py

cd ..
python experiments/gen_robustness_report.py
```

## After a new data cut

The paper reads no cache, so the manuscript regenerates on its own:

```bash
cd replication/scripts
python gen_latex.py                    # recompute all macros
python gen_figures.py                  # redraw the Section V-A figures
# rebuild the PDF and review the diff of macros.tex and figures/generated/
```

A new export invalidates all three caches, so every producer must run before
the robustness report:

```bash
cd replication/scripts
python pls_bootstrap.py --seconds 35   # repeat until it reports 5000/5000
python pls_alternatives.py
python validate_rho_diff.py
cd ..
python experiments/gen_robustness_report.py
```

This step is deliberately **manual**, not wired into the LaTeX build, so number
changes between data cuts are reviewed rather than moving silently. Effective
sample sizes vary by statistic (full sample; Want-subscale excluding item
non-response; listwise-complete excluding *"I do not know"* maturity);
`gen_latex.py` computes each on its correct denominator and exposes the
matching `\gsmSample…` macros.

`gen_latex.py` defines every macro it computes, whether or not the paper cites
it, and prints the uncited ones on each run. That keeps `macros.tex` a pure
function of the data and the generator, so its diff shows a data cut and
nothing else — editing a sentence never rewrites a generated file.

## Experiments

`experiments/` holds three analyses that were run during the study and left out
of the paper — the response funnel and dropout comparison, an exploratory
factor analysis of the Want items, and the outer measurement model — plus the
macro and report layer that turns the PLS-SEM and robustness checks into
`reports/robustness_report.txt` (`extra_macros.py`, `gen_robustness_report.py`).
Nothing in `scripts/` imports them and no macro in the paper depends on them;
they are kept because each analysis settled a decision and the report bounds
how far the paper's conclusions carry. See `experiments/README.md`.

## Tooling disclaimer

AI assistance (Claude, Anthropic) was used to help author the Python analysis
scripts, the macro layer that carries every reported value into the manuscript
(`gen_latex.py`), and the TikZ figure generator (`gen_figures.py`). All
outputs were reviewed and verified by the authors.
