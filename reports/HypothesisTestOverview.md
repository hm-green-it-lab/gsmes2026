# Hypothesis Test Overview — Green Software Metrics Enterprise Survey

Reference table for the pairwise tests behind H1, H2, H4, H5 and H8 of the Want-Can-Do-Outlook-Context model. Hypothesis numbers are the paper's (chapters/hypotheses.tex). H3 (Context → Do), H6 and H7 are assessed jointly in the PLS-SEM and are not pairwise rows; see `reports/robustness_report.txt`. n=88, significance threshold p < 0.05.

**Auto-generated** from `scripts/gen_overview.py` (same scoring/test functions as `hypothesis_test.py`). Do not edit by hand — rerun the generator.

---

## Part 1 — Dimensions & Operationalization

For Context rows the **Scale / Groups** column shows the two compared groups (A = reference, B = comparison); for all other rows it shows the measurement scale range.

| Dimension | Variable | Scale / Groups | Definition |
|---|---|---|---|
| **Context** | `size_250` | SME / Enterprise | ≤249 employees = SME; ≥250 = Enterprise (EU SME definition) — primary H1/H2 split |
| | `size_1000` | SME / Enterprise | ≤999 = SME; ≥1000 = Enterprise (alternative threshold; sensitivity only) |
| | `domain_web` | Non-Web / Web | Operates in the Web/Internet domain — primary H1 moderator |
| | `domain_breadth` | Narrow / Broad | ≤1 domain = Narrow; ≥2 = Broad (tech-stack heterogeneity proxy — primary H2 moderator) |
| | `domain_count` | 0–5 | Number of domains per respondent (ordinal robustness complement to `domain_breadth`) |
| | `domain_Business_Enterprise` | Non / Yes | Operates in Business/Enterprise software domain |
| | `domain_Platform_Infra` | Non / Yes | Operates in Platform/Infrastructure domain |
| | `domain_Mobile` | Non / Yes | Operates in Mobile domain |
| | `domain_IoT_Embedded` | Non / Yes | Operates in IoT/Embedded domain |
| | `maturity_strict` | Low / High | Low = none or ad hoc; High = KPIs or SDLC-integrated; "I do not know" excluded |
| | `maturity_ord` | 0–3 | Ordinal maturity (None→Ad hoc→KPIs→SDLC); complement using full gradient |
| **Want** | `want_env` | 1–5 | mean(CO₂ reduction, Energy reduction) |
| | `want_ops` | 1–5 | mean(Cost reduction, Performance, Compliance) |
| | `want_all` | 1–5 | mean of all five objectives |
| **Can** | `can_constraint` | 0–2 | 0 structurally blocked / 1 addressable-only / 2 clear; primary for H2 |
| | `can_effort` | 1–5 | "Reducing energy is worth the required effort" Likert; primary for H4 |
| | `can_score` | 0–10 | 10 − number of barriers reported; legacy, H2 sensitivity only |
| **Do** | `do_breadth` | 0–3 | Tier ladder — 1: proxy/resource, 2: +physical energy, 3: +environmental (CO₂); strict |
| | `do_institutionalized` | 0–3 | Decision frequency (0/1/2) + green requirements in place (0/1) |
| | `do_composite` | 0–6 | `do_breadth` + `do_institutionalized` |
| **Outlook** | `outlook_intent` | 1–5 | Likelihood of adopting structured measurement guidelines |
| | `outlook_awareness` | 0–9 | Count of decision areas recognised as impacted |

---

## Legend

**Tests** — Mann-Whitney U: two-group score comparison (H1, H2). Spearman ρ: monotonic association (H1–H5). **Sign (MW rows):** ΔMdn = Mdn_B − Mdn_A (A = reference, B = comparison).

**Role** — Primary: pre-specified, Holm-Bonferroni corrected (H1/H2) or confirmatory pair (H3–H5). Complementary: secondary operationalization. Sensitivity: alternative threshold/operationalization. Exploratory: uncorrected breadth row.

**Status** — **✓** primary surviving Holm · ✓ significant uncorrected (p<0.05) · ~ correct direction, n.s. · ✗ wrong direction · ⚠ caveat (not interpretable as intended).

---

## Part 2 — Statistical Tests

Primary pairs for H1 and H2 are Holm-Bonferroni corrected (k=3 per hypothesis); secondary, sensitivity, and complementary rows are uncorrected. Primary tests are confirmatory. Effective n varies: Want pairs exclude item non-response; `maturity` rows exclude "I do not know".

### Mann-Whitney U — Context group comparisons (H1, H2)

| H | Context split | Outcome | Role | n_A | n_B | ΔMdn | p_raw | p_adj | Status |
|---|---|---|---|---:|---:|---:|---:|---:|:---:|
| H1 | `size_250` | want_ops | Primary | 30 | 55 | +0.33 | 0.223 | 0.223 | ~ |
| **H1** | **`domain_web`** | **want_env** | **Primary** | **41** | **41** | **+1.00** | **0.005** | **0.009** | **✓** |
| **H1** | **`maturity_strict`** | **want_env** | **Primary** | **49** | **19** | **+1.00** | **0.002** | **0.005** | **✓** |
| H1 | `size_250` | want_env | Exploratory | 30 | 52 | +0.25 | 0.926 | — | ~ |
| H1 | `size_250` | want_all | Exploratory | 30 | 55 | +0.50 | 0.124 | — | ~ |
| H1 | `domain_web` | want_ops | Exploratory | 42 | 43 | +0.33 | 0.911 | — | ~ |
| H1 | `domain_web` | want_all | Exploratory | 42 | 43 | +0.40 | 0.073 | — | ~ |
| H1 | `maturity_strict` | want_ops | Exploratory | 51 | 19 | +0.33 | 0.512 | — | ~ |
| H1 | `maturity_strict` | want_all | Exploratory | 51 | 19 | +0.60 | 0.007 | — | ✓ |
| H1 | `size_1000` | want_env | Sensitivity | 47 | 35 | +0.50 | 0.913 | — | ~ |
| H1 | `size_1000` | want_ops | Sensitivity | 47 | 38 | +0.33 | 0.209 | — | ~ |
| H1 | `size_1000` | want_all | Sensitivity | 47 | 38 | +0.40 | 0.120 | — | ~ |
| H1 | `domain_Business_Enterprise` | want_env | Exploratory | 33 | 49 | -0.50 | 0.553 | — | ~ |
| H1 | `domain_Business_Enterprise` | want_ops | Exploratory | 34 | 51 | +0.00 | 0.761 | — | ~ |
| H1 | `domain_Business_Enterprise` | want_all | Exploratory | 34 | 51 | +0.20 | 0.864 | — | ~ |
| H1 | `domain_Platform_Infra` | want_env | Exploratory | 36 | 46 | +0.25 | 0.340 | — | ~ |
| H1 | `domain_Platform_Infra` | want_ops | Exploratory | 37 | 48 | +0.17 | 0.257 | — | ~ |
| H1 | `domain_Platform_Infra` | want_all | Exploratory | 37 | 48 | +0.50 | 0.147 | — | ~ |
| H1 | `domain_Mobile` | want_env | Exploratory | 65 | 17 | +0.50 | 0.178 | — | ~ |
| H1 | `domain_Mobile` | want_ops | Exploratory | 67 | 18 | +0.33 | 0.203 | — | ~ |
| H1 | `domain_Mobile` | want_all | Exploratory | 67 | 18 | +0.70 | 0.027 | — | ✓ |
| H1 | `domain_IoT_Embedded` | want_env | Exploratory | 64 | 18 | +0.50 | 0.300 | — | ~ |
| H1 | `domain_IoT_Embedded` | want_ops | Exploratory | 67 | 18 | +0.33 | 0.790 | — | ~ |
| H1 | `domain_IoT_Embedded` | want_all | Exploratory | 67 | 18 | +0.20 | 0.682 | — | ~ |
| H2 | `domain_breadth` | can_constraint | Primary | 29 | 57 | +0.00 | 0.599 | 1.000 | ~ |
| H2 | `size_250` | can_constraint | Primary | 31 | 55 | +0.00 | 0.826 | 1.000 | ~ |
| H2 | `maturity_strict` | can_constraint | Primary | 51 | 19 | +0.00 | 0.347 | 1.000 | ~ |
| H2 | `domain_breadth` | can_effort | Complementary | 31 | 57 | +1.00 | 0.458 | — | ~ |
| H2 | `size_250` | can_effort | Complementary | 32 | 56 | +1.00 | 0.078 | — | ~ |
| H2 | `maturity_strict` | can_effort | Complementary | 51 | 19 | +0.00 | 0.848 | — | ~ |
| H2 | `domain_breadth` | can_score | Sensitivity | 31 | 57 | -1.00 | 0.115 | — | ~ |
| H2 | `size_250` | can_score | Sensitivity | 32 | 56 | -1.00 | 0.157 | — | ~ |
| H2 | `maturity_strict` | can_score | Sensitivity | 51 | 19 | +0.00 | 0.076 | — | ~ |
| H2 | `domain_Business_Enterprise` | can_constraint | Exploratory | 35 | 51 | +0.00 | 0.931 | — | ~ |
| H2 | `domain_Platform_Infra` | can_constraint | Exploratory | 37 | 49 | +0.00 | 0.818 | — | ~ |
| H2 | `domain_Web` | can_constraint | Exploratory | 43 | 43 | +0.00 | 0.680 | — | ~ |
| H2 | `domain_Mobile` | can_constraint | Exploratory | 68 | 18 | +0.00 | 0.635 | — | ~ |
| H2 | `domain_IoT_Embedded` | can_constraint | Exploratory | 68 | 18 | +0.00 | 0.516 | — | ~ |

### Spearman ρ — Monotonic associations (H1, H2, H4, H5, H8)

| H | Predictor | Outcome | Role | n | ρ | p | Status |
|---|---|---|---|---:|---:|---:|:---:|
| H1 | `maturity_ordinal` | `want_env` | Complementary | 68 | +0.440 | <0.001 | ✓ |
| H1 | `maturity_ordinal` | `want_ops` | Complementary | 70 | +0.135 | 0.265 | ~ |
| H1 | `maturity_ordinal` | `want_all` | Complementary | 70 | +0.367 | 0.002 | ✓ |
| H2 | `maturity_ordinal` | `can_constraint` | Complementary | 70 | +0.050 | 0.682 | ~ |
| H2 | `maturity_ordinal` | `can_effort` | Complementary | 70 | +0.089 | 0.461 | ~ |
| H2 | `domain_count` | `can_constraint` | Complementary | 86 | -0.063 | 0.567 | ~ |
| H2 | `domain_count` | `can_effort` | Complementary | 88 | +0.122 | 0.257 | ~ |
| H4 | `want_env` | `do_breadth` | Exploratory | 82 | +0.073 | 0.512 | ~ |
| **H4** | **`want_env`** | **`do_institutionalized`** | **Primary** | **82** | **+0.293** | **0.008** | **✓** |
| H4 | `want_env` | `do_composite` | Exploratory | 82 | +0.224 | 0.043 | ✓ |
| H4 | `want_ops` | `do_breadth` | Exploratory | 85 | +0.164 | 0.133 | ~ |
| H4 | `want_ops` | `do_institutionalized` | Exploratory | 85 | +0.022 | 0.845 | ~ |
| H4 | `want_ops` | `do_composite` | Exploratory | 85 | +0.125 | 0.256 | ~ |
| H4 | `want_all` | `do_breadth` | Exploratory | 85 | +0.143 | 0.193 | ~ |
| H4 | `want_all` | `do_institutionalized` | Exploratory | 85 | +0.195 | 0.074 | ~ |
| H4 | `want_all` | `do_composite` | Exploratory | 85 | +0.210 | 0.054 | ~ |
| **H5** | **`can_effort`** | **`do_composite`** | **Primary** | **88** | **+0.417** | **<0.001** | **✓** |
| **H5** | **`can_effort`** | **`do_breadth`** | **Primary** | **88** | **+0.295** | **0.005** | **✓** |
| **H5** | **`can_effort`** | **`do_institutionalized`** | **Primary** | **88** | **+0.367** | **<0.001** | **✓** |
| H5 | `can_score` | `do_composite` | Sensitivity | 88 | -0.040 | 0.708 | ⚠ |
| **H8** | **`do_composite`** | **`outlook_intent`** | **Primary** | **88** | **+0.401** | **<0.001** | **✓** |
| **H8** | **`do_composite`** | **`outlook_awareness`** | **Primary** | **88** | **+0.329** | **0.002** | **✓** |

