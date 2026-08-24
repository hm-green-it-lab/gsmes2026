"""
=============================================================================
Green Software Metrics Survey – Shared Configuration
=============================================================================
Single source of truth for column indices, dimension definitions, Likert
mappings, role columns, and statistical thresholds.

Edit ONLY this file when the LimeSurvey export changes column positions,
when new questions are added, or when threshold decisions are revised.

Every script in scripts/ and experiments/ imports from here.
=============================================================================
"""

import os

# -----------------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------------
# Layout, all anchored to this file so nothing depends on the working directory:
#
#   replication/scripts/   this code
#   replication/data/      survey export and instrument (inputs)
#   replication/reports/   generated artifacts (tracked; shipped to reviewers)
#   chapters/generated/    macros.tex, consumed by the LaTeX build
#   figures/generated/     TikZ figure bodies, consumed by the LaTeX build
#
# Centralised rather than spelled out in each generator so there is one place to
# change, and so no generator can quietly write somewhere .gitignore swallows --
# which would leave a cited file missing from a fresh clone, a failure nobody
# notices until a reviewer hits it.
_HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.abspath(os.path.join(_HERE, ".."))
REPO_DIR = os.path.abspath(os.path.join(PACKAGE_DIR, ".."))

DATA_DIR = os.path.join(PACKAGE_DIR, "data")
REPORTS_DIR = os.path.join(PACKAGE_DIR, "reports")
GENERATED_TEX_DIR = os.path.join(REPO_DIR, "chapters", "generated")
FIGURES_TEX_DIR = os.path.join(REPO_DIR, "figures", "generated")


def data_path(filename):
    """Absolute path to a survey input in replication/data/."""
    return os.path.join(DATA_DIR, filename)


def figure_path(filename):
    """Absolute path to a generated figure body in figures/generated/."""
    return os.path.join(FIGURES_TEX_DIR, filename)


def report_path(filename):
    """Absolute path to a generated artifact in replication/reports/."""
    return os.path.join(REPORTS_DIR, filename)


def csv_arg(argv, default):
    """The optional export path from a command line, ignoring any --flags.

    Every entry point takes `[path/to/export.csv]` and its own flags. Parsing
    that in one place keeps `--quiet` from being mistaken for a filename.
    """
    positional = [a for a in argv[1:] if not a.startswith("--")]
    return positional[0] if positional else default


def survey_fingerprint(df):
    """SHA-1 of a loaded survey export -- the key every .npz cache is stored under.

    The caches hold resamples and simulations that take about a quarter of an
    hour to rebuild, so they ship with the package. That is only safe if a cache
    from a different data cut cannot be picked up silently, so each writer
    stamps this digest and each reader refuses anything that does not match.

    Taken over the canonical-ordered frame rather than the file, so it tracks
    the data itself and not the export's byte layout.
    """
    import hashlib
    return hashlib.sha1(
        df.to_csv(index=False).encode("utf-8")).hexdigest()


def display_path(path):
    """A path as it should appear inside a generated report: relative to the
    repository root, with forward slashes.

    Reports are committed artifacts, so an absolute path would record whichever
    machine last regenerated them and would show up as a spurious diff on the
    next run elsewhere. Falls back to the basename for anything outside the
    repository.
    """
    try:
        rel = os.path.relpath(os.path.abspath(path), REPO_DIR)
    except ValueError:               # different drive on Windows
        return os.path.basename(path)
    return os.path.basename(path) if rel.startswith("..") else rel.replace(os.sep, "/")


# -----------------------------------------------------------------------------
# SURVEY DIMENSIONS
# Each key is a named block. "type" controls which analysis handler runs.
# "items" maps short variable names → 0-based column index in the CSV export.
# Columns marked "free text / Other" are excluded from all quantitative tests.
# -----------------------------------------------------------------------------
SURVEY_DIMENSIONS = {

    # ── CONTEXT ──────────────────────────────────────────────────────────────
    "context_size": {
        "label": "CONTEXT – Organization size",
        "type":  "ordinal",
        "items": {"Org_size": 16},
        "order": ["1–49", "50–249", "250–999", "1000–4999", "5000+"],
    },
    "context_maturity": {
        "label": "CONTEXT – Sustainability maturity",
        "type":  "ordinal",
        "items": {"Sustainability_maturity": 34},
        "order": [
            "None",
            "Ad hoc pilots",
            "Defined sustainability KPIs",
            "Integrated into the Software Development Life Cycle (SDLC)",
            "I do not know",
        ],
    },

    # ── WANT ─────────────────────────────────────────────────────────────────
    "want_objectives": {
        "label": "WANT – Importance of objectives (Likert 1–5)",
        "type":  "likert",
        "items": {
            "CO2_reduction":    46,
            "Energy_reduction": 47,
            "Cost_reduction":   48,
            "Performance":      49,
            "Compliance":       50,
        },
        "scale": (1, 5),
        "clusters": {
            "Environmental": ["CO2_reduction", "Energy_reduction"],
            "Operational":   ["Cost_reduction", "Performance", "Compliance"],
        },
    },
    "want_influence": {
        "label": "WANT – Energy influences decisions",
        "type":  "binary",
        "items": {"Energy_influences": 51},
        "positive_value": "Yes",
    },
    "want_effort": {
        "label": "WANT – Effort worthwhile (Likert 1–5)",
        "type":  "likert_text",
        "items": {"Effort_worthwhile": 52},
        "mapping": {
            "Strongly disagree":       1,
            "Disagree":                2,
            "Neither agree nor disagree": 3,
            "Agree":                   4,
            "Strongly agree":          5,
        },
        "scale": (1, 5),
    },

    # ── CAN ──────────────────────────────────────────────────────────────────
    "can_barriers": {
        "label": "CAN – Barriers to measurement",
        "type":  "multiselect",
        # col 63 = Other (free text) – excluded from quantitative scoring
        "items": {
            "Lack_methods":        53,
            "Lack_ground_truth":   54,
            "High_tooling":        55,
            "Data_integration":    56,
            "Unclear_resp":        57,
            "Skills_gaps":         58,
            "Cost_instrument":     59,
            "Restricted_access":   60,
            "Inconsistent_results":61,
            "No_barriers":         62,
        },
    },
    "can_enablers": {
        "label": "CAN – Valuable improvements",
        "type":  "multiselect",
        # col 70 = Other (free text) – excluded from quantitative scoring
        "items": {
            "Std_metrics":       64,
            "Procedures":        65,
            "Granularity":       66,
            "Uncertainty_report":67,
            "Ref_dashboards":    68,
            "CI_CD_patterns":    69,
        },
    },
    "can_political": {
        "label": "CAN – Regulation needed",
        "type":  "binary",
        "items": {"Regulation_needed": 71},
        "positive_value": "Yes",
    },

    # ── DO ───────────────────────────────────────────────────────────────────
    "do_decisions": {
        "label": "DO – Frequency of energy-based decisions",
        "type":  "ordinal",
        "items": {"Decision_frequency": 72},
        "order": ["Never", "Random / ad hoc", "Regularly"],
    },
    "do_requirements": {
        "label": "DO – Green requirements in design",
        "type":  "binary",
        "items": {"Green_requirements": 73},
        "positive_value": "Yes",
    },
    "do_metrics": {
        "label": "DO – Metrics currently tracked",
        "type":  "multiselect",
        # col 87 = Other (free text) – excluded from quantitative scoring
        "items": {
            "Power_W":      74,
            "Energy_kWh":   75,
            "CO2e":         76,
            "Water":        77,
            "Cost_proxy":   78,
            "SCI":          79,
            "CPU_util":     80,
            "Memory_util":  81,
            "Disk_IO":      82,
            "Network_util": 83,
            "Latency":      84,
            "Throughput":   85,
            "Error_rate":   86,
        },
    },
    "do_co2_scope": {
        "label": "DO – CO₂e differentiated by scope (Scope 1/2/3)",
        "type":  "binary",
        "items": {"CO2_scope": 88},
        "positive_value": "Yes",
    },
    "do_lifecycle": {
        "label": "DO – Lifecycle phases where measurement occurs",
        "type":  "multiselect",
        # col 94 = Other (free text) – excluded
        "items": {
            "Development":  89,
            "Testing_QA":   90,
            "Production":   91,
            "Maintenance":  92,
            "Ad_hoc":       93,
        },
    },
    "do_observation_scope": {
        "label": "DO – Observation scopes used for measurement",
        "type":  "multiselect",
        # col 101 = Other (free text) – excluded
        "items": {
            "Infra_physical":    95,
            "Infra_VM":          96,
            "Infra_container":   97,
            "SW_service":        98,
            "SW_transaction":    99,
            "SW_code":          100,
        },
    },
    "do_automation": {
        "label": "DO – Measurement automated",
        "type":  "binary",
        "items": {"Automated": 102},
        "positive_value": "Yes",
    },
    "do_measurement_runtime": {
        "label": "DO – Runtime environments where measurements are taken",
        "type":  "multiselect",
        "items": {
            "Meas_bare_metal": 103,
            "Meas_VMs":        104,
            "Meas_containers": 105,
            "Meas_serverless": 106,
        },
    },

    # ── OUTLOOK ──────────────────────────────────────────────────────────────
    "outlook_likely": {
        "label": "OUTLOOK – Likely to adopt guidelines (Likert 1–5)",
        "type":  "likert_text",
        "items": {"Adopt_guidelines": 107},
        "mapping": {
            "Strongly unlikely":       1,
            "Unlikely":                2,
            "Neither likely nor unlikely": 3,
            "Likely":                  4,
            "Strongly likely":         5,
        },
        "scale": (1, 5),
    },
    "outlook_impact": {
        "label": "OUTLOOK – Decision areas impacted",
        "type":  "multiselect",
        # col 117 = Other (free text) – excluded
        "items": {
            "Architecture":            108,
            "Hardware":                109,
            "Libraries":               110,
            "Right_sizing":            111,
            "Source_code":             112,
            "Cloud_region":            113,
            "Procurement":             114,
            "Requirements":            115,
            "Sustainability_reporting":116,
        },
    },
}

# -----------------------------------------------------------------------------
# LIKERT TEXT MAPPINGS  (used wherever text labels must be converted to numbers)
# -----------------------------------------------------------------------------
EFFORT_MAP = {
    "Strongly disagree":          1,
    "Disagree":                   2,
    "Neither agree nor disagree": 3,
    "Agree":                      4,
    "Strongly agree":             5,
}

OUTLOOK_MAP = {
    "Strongly unlikely":          1,
    "Unlikely":                   2,
    "Neither likely nor unlikely": 3,
    "Likely":                     4,
    "Strongly likely":            5,
}

# -----------------------------------------------------------------------------
# RESPONDENT CONTEXT COLUMNS  (not used in H1; kept for descriptive breakdowns)
# All multi-select Yes/No unless noted.
# -----------------------------------------------------------------------------
ROLE_COLS = {
    "Management":    7,
    "Technical":     8,
    "Non-Technical": 9,
}

DOMAIN_COLS = {
    "Mobile":              10,
    "IoT_Embedded":        11,
    "Web":                 12,
    "Platform_Infra":      13,
    "Business_Enterprise": 14,
    # col 15 = Other (free text) – excluded
}

# Which organisational roles are involved in green metrics activities (multi-select Yes/No)
# col 33 = Other (free text) – excluded
GREEN_METRICS_ROLE_COLS = {
    "Developers":            26,
    "Architects":            27,
    "DevOps_SRE":            28,
    "Product_Management":    29,
    "Sustainability_ESG":    30,
    "Operations_IT":         31,
    "External_Stakeholders": 32,
}

WORKLOAD_COLS = {
    "Transactional":   18,
    "Analytics":       19,
    "Event_Realtime":  20,
    "Batch_Pipeline":  21,
    "ML_AI":           22,
    "Embedded_Edge":   23,
    "HPC":             24,
    # col 25 = Other (free text) – excluded
}

RUNTIME_COLS = {
    "Bare_metal":  35,
    "VMs":         36,
    "Containers":  37,
    "Serverless":  38,
}

INFRASTRUCTURE_COLS = {
    "Self_operated":      41,
    "Hosted_Colocated":   42,
    "VM_External":        43,
    "PaaS_External":      44,
    "Fully_Managed":      45,
}

# Ordinal / categorical – single-answer columns
AUTHORITY_COL    = 17   # Who has authority for sustainability decisions
ARCH_STYLE_COL   = 39   # Architectural style (Monolithic / Modular / Microservices)
                         # col 40 = Other (free text) – excluded

# -----------------------------------------------------------------------------
# HYPOTHESIS SCORING – context grouping rules
# Defined here so both scripts can reference the same group boundaries.
# -----------------------------------------------------------------------------
# Size: PRIMARY split at 250 (EU SME definition); the 250-cut coincides with the
# EU CSRD reporting threshold and avoids splitting the middle of the size range.
CONTEXT_SIZE_GROUPS = {
    "SME (1–249)":       ["1–49", "50–249"],
    "Enterprise (250+)": ["250–999", "1000–4999", "5000+"],
}

# SENSITIVITY split: alternative 1000 threshold, retained for transparency
# and reported uncorrected alongside the primary model.
CONTEXT_SIZE_GROUPS_LEGACY_1000 = {
    "SME (1–999)":        ["1–49", "50–249", "250–999"],
    "Enterprise (1000+)": ["1000–4999", "5000+"],
}

# Primary domain moderator for H1 (elevated want_env in Web-domain respondents;
# survives Holm correction). Single multi-select column flag.
CONTEXT_DOMAIN_PRIMARY = "Web"

# -----------------------------------------------------------------------------
# DO – institutionalization scoring
# Decision frequency mapped to 0/1/2; green requirements to 0/1.
# do_institutionalized = freq_score + req_score  (range 0–3)
# do_composite         = do_breadth + do_institutionalized  (range 0–6)
# -----------------------------------------------------------------------------
DO_DECISION_MAP = {
    "Never":            0,
    "Random / ad hoc":  1,
    "Regularly":        2,
}
DO_DECISION_COL     = 72   # Decision_frequency
DO_REQUIREMENTS_COL = 73   # Green_requirements (Yes=1, No=0)

# Maturity: PRIMARY split is STRICT (excludes "I do not know" – informative
# missingness). Legacy permissive split retained under _LEGACY name for
# sensitivity reporting.
CONTEXT_MATURITY_LOW  = ["None", "Ad hoc pilots"]
CONTEXT_MATURITY_HIGH = [
    "Defined sustainability KPIs",
    "Integrated into the Software Development Life Cycle (SDLC)",
]
CONTEXT_MATURITY_UNKNOWN = ["I do not know"]   # excluded from strict tests

# Ordinal scoring for Spearman analyses (0–3). "I do not know" is absent →
# maps to NaN (informative missingness; excluded from ordinal tests).
CONTEXT_MATURITY_ORDINAL_MAP = {
    "None":               0,
    "Ad hoc pilots":      1,
    "Defined sustainability KPIs": 2,
    "Integrated into the Software Development Life Cycle (SDLC)": 3,
}

# Legacy: includes "I do not know" in the Low group
CONTEXT_MATURITY_LOW_LEGACY = ["Ad hoc pilots", "I do not know", "None"]

# Do-score metric tier weights (used in hypothesis_test.py)
DO_METRIC_WEIGHTS = {
    # Environmental impact indicators – hardest to collect, highest signal
    "CO2e":       3,
    "Water":      3,
    "SCI":        3,
    # Physical energy measurements
    "Power_W":    2,
    "Energy_kWh": 2,
    # Proxy / resource utilisation – weight 1 (default; not listed = 1)
}

# -----------------------------------------------------------------------------
# CAN – barrier classification for sub-scale scoring
# Structural = externally imposed, org cannot fix alone
# Addressable = internal, fixable via training / tooling / process
# -----------------------------------------------------------------------------
CAN_STRUCTURAL_BARRIERS = [
    "Restricted_access",    # provider / org policy blocks instrumentation
    "Unclear_resp",         # governance: nobody owns this
    "Cost_instrument",      # budget/procurement constraint
    "Data_integration",     # data architecture lock-in
]
CAN_ADDRESSABLE_BARRIERS = [
    "Lack_methods",         # no agreed methodology – trainable
    "Lack_ground_truth",    # no direct power measurement – tooling gap
    "High_tooling",         # tooling overhead – solvable
    "Skills_gaps",          # skills – addressable via hiring / training
    "Inconsistent_results", # methodology maturity – improvable
]
# Note: "No_barriers" is excluded from both sub-scales (it is a sentinel flag)

# Free-text "Other" barrier column. A non-empty entry counts as ONE additional
# barrier, so respondents who reported only an "Other" barrier are no longer
# mis-scored as having no barriers. This raises the barrier universe to 10
# (9 listed + Other), so can_score = 10 - barrier_count.
CAN_OTHER_BARRIER_COL = 63

# -----------------------------------------------------------------------------
# DO – tier definitions for breadth score (0–3 ordinal maturity ladder)
# A respondent reaches tier N if they track ≥1 metric from ALL tiers 1..N
# -----------------------------------------------------------------------------
DO_TIER_PROXY        = {"CPU_util", "Memory_util", "Disk_IO", "Network_util",
                        "Latency", "Throughput", "Error_rate", "Cost_proxy"}
DO_TIER_PHYSICAL     = {"Power_W", "Energy_kWh"}
DO_TIER_ENVIRONMENTAL= {"CO2e", "Water", "SCI"}

# -----------------------------------------------------------------------------
# ROBUSTNESS – specifications shared by gen_latex.py and experiments/extra_macros.py
#
# Both the paper's macros and the standalone robustness report are built from
# these lists, so a scoring variant or a regression term cannot be present in
# one output and absent from the other.
# -----------------------------------------------------------------------------

# Eight scorings of Do. The tier ladder and the equal weighting of coverage
# against governance embedding are author decisions rather than the data's;
# every association is recomputed under each alternative so a reader can see
# which findings survive the choice.  `build` takes the score dict returned by
# hypothesis_test.build_dimension_scores().
#   tag    -> macro infix (\gsmSens<tag><Hypothesis>Rho)
#   label  -> row label in the robustness report
DO_VARIANT_SPECS = [
    ("Paper",    "cumulative tiers + governance  (reported)",
     lambda s: s["do_composite"]),
    ("Highest",  "highest tier reached + governance",
     lambda s: s["do_highest"] + s["do_institutionalized"]),
    ("RawCount", "raw metric count + governance",
     lambda s: s["do_raw"] + s["do_institutionalized"]),
    ("Weighted", "tier-weighted metrics + governance",
     lambda s: s["do_weighted"] + s["do_institutionalized"]),
    ("CovTwice", "coverage weighted 2:1 over governance",
     lambda s: 2 * s["do_breadth"] + s["do_institutionalized"]),
    ("GovTwice", "governance weighted 2:1 over coverage",
     lambda s: s["do_breadth"] + 2 * s["do_institutionalized"]),
    ("CovOnly",  "metric coverage only  (governance dropped)",
     lambda s: s["do_breadth"]),
    ("GovOnly",  "governance only  (coverage dropped)",
     lambda s: s["do_institutionalized"]),
]

# The three hypotheses recomputed under each scoring above.
#   key -> score name        tag -> macro infix        label -> report column
DO_VARIANT_HYPOTHESES = [
    ("can_effort",     "Can",     "H5 Can(Val)→Do"),
    ("want_env",       "Want",    "H4 Want(env)→Do"),
    ("outlook_intent", "Outlook", "H8 Do→Outlook"),
]

# Proportional-odds ordinal logistic specifications (estimator-independent
# check on the PLS-SEM results).  All three are fitted on the SAME
# listwise-complete cases as the PLS model they check.
#   tag -> macro infix (\gsmSample<tag>, \gsm<tag><Term>Or)
# -----------------------------------------------------------------------------
# FREE-TEXT BARRIER CODING
# -----------------------------------------------------------------------------
# The barrier item offers nine closed categories plus an open "Other" field.
# The open entries are coded here so the counts the paper reports regenerate
# from the data rather than being transcribed.
#
# Keyed by the verbatim entry (lower-cased, stripped) so the mapping survives
# reordering of the export; an entry present in the data but absent here raises
# in gen_latex rather than being silently dropped.
#
# `closed_equivalent` names the closed category the entry restates, or None if
# the instrument had no category for it.  That distinction is the point of the
# exercise: it measures how far the closed list under-covers what respondents
# volunteer, which the paper otherwise only concedes qualitatively.
FREETEXT_BARRIER_CODING = {
    "lack of management support & interest":   ("Leadership priority",  None),
    "priority":                                ("Leadership priority",  None),
    "other business priorities":               ("Leadership priority",  None),
    "out of scope":                            ("Leadership priority",  None),
    "lack of relevance in digitalization strategy": ("Leadership priority", None),
    "no priority from leadership":             ("Leadership priority",  None),
    "the motivation to actually do it":        ("Motivation",           None),
    "interest":                                ("Motivation",           None),
    "lack of motivation":                      ("Motivation",           None),
    "no one gives a fuck":                     ("Motivation",           None),
    "not a strong business case":              ("Business case",        "Cost_instrument"),
    "cost (efforts, know how) vs. benefit":    ("Business case",        "Cost_instrument"),
    "costs":                                   ("Business case",        "Cost_instrument"),
    "availability of data from vendors":       ("Provider data",        "Restricted_access"),
    "mainly saas":                             ("Provider data",        "Restricted_access"),
    "lack of details in hardware specifications": ("Provider data",     "Restricted_access"),
    "limited visibility into ai energy consumption. not provided by hyperscalers!":
                                               ("Provider data",        "Restricted_access"),
    "lack of reliable measurement tools":      ("Tooling",              "High_tooling"),
    "very complex software landscape":         ("Tooling",              "High_tooling"),
    # Describes a response to the ground-truth gap rather than a barrier.
    "in my case, we use the lack of ground truth and etc to try to develop some of those, for example":
                                               ("Not a barrier",        None),
}

ORD_SPECS = [
    # --- Primary joint estimates -------------------------------------------
    # Do outcomes carry H3 (maturity), H4 (want) and H5 (valuation); the two
    # components are fitted alongside the composite because maturity overlaps
    # in content with governance embedding but not with metric coverage.
    ("OrdComposite",   "composite",  ["want", "valuation", "maturity"],
     "Do (practice composite)"),
    ("OrdCoverage",    "coverage",   ["want", "valuation", "maturity"],
     "Do (metric coverage)"),
    ("OrdGovernance",  "governance", ["want", "valuation", "maturity"],
     "Do (governance embedding)"),
    # Outlook outcomes carry H6 (want), H7 (valuation) and H8 (practice).
    ("OrdIntent",      "intent",     ["want", "valuation", "composite"],
     "Outlook (adoption intent)"),
    ("OrdAwareness",   "awareness",  ["want", "valuation", "composite"],
     "Outlook (decision-area awareness)"),
    # --- Context-removed refits (construct-overlap check) -------------------
    # Maturity shares content with governance embedding, so dropping it shows
    # how much of the Want -> Do path it absorbs.
    ("OrdCompositeNoCtx", "composite", ["want", "valuation"],
     "Do (practice composite), Context removed"),
    ("OrdCoverageNoCtx",  "coverage",  ["want", "valuation"],
     "Do (metric coverage), Context removed"),
]

# Column name -> regression term name (macro infix) -> report label.
ORD_TERMS = {
    "want":      ("Want",      "Want(env) – motivation"),
    "valuation": ("Valuation", "Can(Val) – valuation"),
    "maturity":  ("Maturity",  "Context – maturity"),
    "composite": ("Practice",  "Do – enacted practice"),
}

# -----------------------------------------------------------------------------
# STATISTICAL THRESHOLDS  (single source of truth – all verdicts derive here)
# -----------------------------------------------------------------------------
T = {
    # Cronbach's alpha bands
    "alpha_good":        0.70,
    "alpha_pilot":       0.60,
    "alpha_weak":        0.50,
    # Spearman / correlation bands
    "r_strong":          0.70,
    "r_moderate":        0.40,
    "r_weak":            0.20,
    # Corrected item-total correlation bands
    "r_it_good":         0.40,
    "r_it_problematic":  0.20,
    # Alpha-if-deleted: flag if removing an item improves α by this much
    "alpha_if_del_flag": 0.05,
    # Item variance quality
    "cv_good":           35.0,   # CV% threshold – good spread
    "cv_moderate":       20.0,   # CV% threshold – moderate spread
    "skew_flag":          1.0,   # |skew| above this → flag ceiling/floor
    "kurt_flag":          2.0,   # |kurt| above this → flag extreme shape
    # Cohen's d bands
    "d_large":            0.80,
    "d_medium":           0.50,
    "d_small":            0.20,
    # Mann-Whitney effect size r
    "mw_r_moderate":      0.30,
    # Significance levels
    "p_sig":              0.05,
    "p_sig2":             0.01,
    "p_sig3":             0.001,
    # Bootstrap iterations for alpha CI
    "alpha_ci_boot":      1000,
    # Minimum median delta to flag as directional trend (hypothesis tests)
    "trend_delta":        0.50,
}

# =============================================================================
# DATA LOADING  (header-based column resolution + completeness filter + checks)
# =============================================================================
# Columns are resolved by HEADER TEXT, not position, so the analysis is robust to
# column reordering or insertion in future LimeSurvey exports. CANONICAL_COLUMNS is
# the reference schema; load_survey() reorders each export to these positions so
# that the existing positional access (df.iloc[:, N]) stays valid. Update this list
# ONLY when the questionnaire itself changes.

CANONICAL_COLUMNS = [
    'Response ID',
    'Date submitted',
    'Last page',
    'Start language',
    'Seed',
    'Date started',
    'Date last action',
    'Which of the following describes your role in relation to software systems in your organization? [Management]',
    'Which of the following describes your role in relation to software systems in your organization? [Technical]',
    'Which of the following describes your role in relation to software systems in your organization? [Non-Technical]',
    'Which domains best describe the software systems you primarily work with?  [Mobile application]',
    'Which domains best describe the software systems you primarily work with?  [IoT (Internet of Things) or embedded software]',
    'Which domains best describe the software systems you primarily work with?  [Web application]',
    'Which domains best describe the software systems you primarily work with?  [Platform or infrastructure software (e.g., Kubernetes, CI/CD platforms, runtime environments)]',
    'Which domains best describe the software systems you primarily work with?  [Business / Enterprise application]',
    'Which domains best describe the software systems you primarily work with?  [Other]',
    'How large is your organization?',
    'Who holds formal decision authority for sustainability-related decisions regarding software systems in your organization? ',
    'Which workload types characterize the majority of the software systems you primarily work with?  [Transactional application workloads (e.g., web applications, APIs)]',
    'Which workload types characterize the majority of the software systems you primarily work with?  [Data analytics / reporting workloads]',
    'Which workload types characterize the majority of the software systems you primarily work with?  [Event-driven / real-time workloads (e.g., message queues, streaming platforms)]',
    'Which workload types characterize the majority of the software systems you primarily work with?  [Batch processing / data pipelines (e.g., Extract–Transform–Load (ETL), scheduled jobs)]',
    'Which workload types characterize the majority of the software systems you primarily work with?  [Machine learning / artificial intelligence (e.g., training or inference)]',
    'Which workload types characterize the majority of the software systems you primarily work with?  [Embedded / edge workloads]',
    'Which workload types characterize the majority of the software systems you primarily work with?  [High-performance or compute-intensive workloads (e.g., simulations, scientific computing)]',
    'Which workload types characterize the majority of the software systems you primarily work with?  [Other]',
    'Which roles are involved in activities related to green software metrics (e.g., defining, assessing, interpreting, or decision-making) within your organization?  \xa0  [Developers]',
    'Which roles are involved in activities related to green software metrics (e.g., defining, assessing, interpreting, or decision-making) within your organization?  \xa0  [Architects]',
    'Which roles are involved in activities related to green software metrics (e.g., defining, assessing, interpreting, or decision-making) within your organization?  \xa0  [DevOps / SRE (Site Reliability Engineering)]',
    'Which roles are involved in activities related to green software metrics (e.g., defining, assessing, interpreting, or decision-making) within your organization?  \xa0  [Product / Management]',
    'Which roles are involved in activities related to green software metrics (e.g., defining, assessing, interpreting, or decision-making) within your organization?  \xa0  [Sustainability / ESG (Environmental, Social, and Governance)]',
    'Which roles are involved in activities related to green software metrics (e.g., defining, assessing, interpreting, or decision-making) within your organization?  \xa0  [Operations / IT]',
    'Which roles are involved in activities related to green software metrics (e.g., defining, assessing, interpreting, or decision-making) within your organization?  \xa0  [External stakeholders / customers]',
    'Which roles are involved in activities related to green software metrics (e.g., defining, assessing, interpreting, or decision-making) within your organization?  \xa0  [Other]',
    'How would you rate the sustainability maturity of the software systems you primarily work with? ',
    'Which runtime environments are used by the\xa0majority of the software systems you primarily work with?  [Bare metal]',
    'Which runtime environments are used by the\xa0majority of the software systems you primarily work with?  [Virtual machines]',
    'Which runtime environments are used by the\xa0majority of the software systems you primarily work with?  [Containers]',
    'Which runtime environments are used by the\xa0majority of the software systems you primarily work with?  [Serverless]',
    'Which architectural style best describes the majority of the software systems you primarily work with? ',
    'Which architectural style best describes the majority of the software systems you primarily work with?  [Other]',
    'Where do the majority of the software systems you primarily work with run?  \xa0  [Self-operated infrastructure with full administrative access (e.g., on-premises servers, edge devices, embedded systems)]',
    'Where do the majority of the software systems you primarily work with run?  \xa0  [Hosted or colocated infrastructure with full administrative access (e.g., own hardware in external data centers)]',
    'Where do the majority of the software systems you primarily work with run?  \xa0  [External infrastructure provider with virtual machine access (e.g., root/admin access to VMs)]',
    'Where do the majority of the software systems you primarily work with run?  \xa0  [External infrastructure provider with platform-level access (e.g., containers or PaaS without host access)]',
    'Where do the majority of the software systems you primarily work with run?  \xa0  [Fully managed or serverless services (e.g., functions, managed databases, SaaS components)]',
    'How important are the following objectives for your organization? [CO₂ reduction]',
    'How important are the following objectives for your organization? [Energy reduction]',
    'How important are the following objectives for your organization? [Cost reduction]',
    'How important are the following objectives for your organization? [Performance / latency]',
    'How important are the following objectives for your organization? [Compliance / reporting]',
    'Does energy consumption currently influence decision-making in the software systems you primarily work with? ',
    'How strongly do you agree or disagree with the following statement? [Reducing energy consumption through software optimization is worth the required effort for our organization.]',
    '\xa0\xa0\xa0\xa0\xa0\xa0\xa0Which factors hinder the measurement of energy consumption for the software systems you primarily work with?  [Lack of comparable methods]',
    '\xa0\xa0\xa0\xa0\xa0\xa0\xa0Which factors hinder the measurement of energy consumption for the software systems you primarily work with?  [Lack of ground truth (direct power measurements)]',
    '\xa0\xa0\xa0\xa0\xa0\xa0\xa0Which factors hinder the measurement of energy consumption for the software systems you primarily work with?  [High tooling overhead]',
    '\xa0\xa0\xa0\xa0\xa0\xa0\xa0Which factors hinder the measurement of energy consumption for the software systems you primarily work with?  [High data integration effort]',
    '\xa0\xa0\xa0\xa0\xa0\xa0\xa0Which factors hinder the measurement of energy consumption for the software systems you primarily work with?  [Unclear responsibilities]',
    '\xa0\xa0\xa0\xa0\xa0\xa0\xa0Which factors hinder the measurement of energy consumption for the software systems you primarily work with?  [Skills gaps]',
    '\xa0\xa0\xa0\xa0\xa0\xa0\xa0Which factors hinder the measurement of energy consumption for the software systems you primarily work with?  [Cost of instrumentation]',
    '\xa0\xa0\xa0\xa0\xa0\xa0\xa0Which factors hinder the measurement of energy consumption for the software systems you primarily work with?  [Restricted access due to provider or organizational policies]',
    '\xa0\xa0\xa0\xa0\xa0\xa0\xa0Which factors hinder the measurement of energy consumption for the software systems you primarily work with?  [Inconsistent results across environments]',
    '\xa0\xa0\xa0\xa0\xa0\xa0\xa0Which factors hinder the measurement of energy consumption for the software systems you primarily work with?  [None – no significant hindering factors identified]',
    '\xa0\xa0\xa0\xa0\xa0\xa0\xa0Which factors hinder the measurement of energy consumption for the software systems you primarily work with?  [Other]',
    'Which improvements would help enable or enhance energy measurement for the software systems you primarily work with?  [Standardized energy and sustainability metrics (definitions, units, formats)]',
    'Which improvements would help enable or enhance energy measurement for the software systems you primarily work with?  [Clear measurement procedures or checklists]',
    'Which improvements would help enable or enhance energy measurement for the software systems you primarily work with?  [Guidance on appropriate measurement granularity (e.g., system, service, transaction level)]',
    'Which improvements would help enable or enhance energy measurement for the software systems you primarily work with?  [Guidelines or templates for reporting uncertainty and accuracy]',
    'Which improvements would help enable or enhance energy measurement for the software systems you primarily work with?  [Reference dashboards or key performance indicators (KPIs)]',
    'Which improvements would help enable or enhance energy measurement for the software systems you primarily work with?  [Patterns for integrating measurements into CI/CD pipelines]',
    'Which improvements would help enable or enhance energy measurement for the software systems you primarily work with?  [Other]',
    'Do you think additional regulation or policy incentives are needed to increase the practical relevance of energy-efficient software? ',
    'How often are decisions regarding the software systems you primarily work with based on energy consumption? ',
    'Are green or energy-related requirements formally part of the design process for the software systems you primarily work with? ',
    'Which metrics are currently tracked for the software systems you primarily work with?  [Power consumption (e.g., watt)]',
    'Which metrics are currently tracked for the software systems you primarily work with?  [Energy consumption (e.g., kWh, joule)]',
    'Which metrics are currently tracked for the software systems you primarily work with?  [CO₂e emissions]',
    'Which metrics are currently tracked for the software systems you primarily work with?  [Water consumption / water footprint]',
    'Which metrics are currently tracked for the software systems you primarily work with?  [Costs used as a proxy for energy or CO₂e]',
    'Which metrics are currently tracked for the software systems you primarily work with?  [Software Carbon Intensity (SCI)]',
    'Which metrics are currently tracked for the software systems you primarily work with?  [CPU utilization]',
    'Which metrics are currently tracked for the software systems you primarily work with?  [Memory utilization]',
    'Which metrics are currently tracked for the software systems you primarily work with?  [Disk I/O]',
    'Which metrics are currently tracked for the software systems you primarily work with?  [Network utilization]',
    'Which metrics are currently tracked for the software systems you primarily work with?  [Response time / latency]',
    'Which metrics are currently tracked for the software systems you primarily work with?  [Throughput / traffic]',
    'Which metrics are currently tracked for the software systems you primarily work with?  [Error rate]',
    'Which metrics are currently tracked for the software systems you primarily work with?  [Other]',
    'Are CO₂e emissions differentiated by emission scope (Scope 1, 2, 3)?',
    'In which phases of the software life cycle are energy measurements performed?\xa0  [Development]',
    'In which phases of the software life cycle are energy measurements performed?\xa0  [Testing / Quality Assurance (QA)]',
    'In which phases of the software life cycle are energy measurements performed?\xa0  [Production / operations]',
    'In which phases of the software life cycle are energy measurements performed?\xa0  [Maintenance / refactoring]',
    'In which phases of the software life cycle are energy measurements performed?\xa0  [Ad hoc / exploratory measurements]',
    'In which phases of the software life cycle are energy measurements performed?\xa0  [Other]',
    'What are the scopes of observation when measuring energy or resource consumption?  [Infrastructure – physical hardware (server / node / rack)]',
    'What are the scopes of observation when measuring energy or resource consumption?  [Infrastructure – virtual machine (VM)]',
    'What are the scopes of observation when measuring energy or resource consumption?  [Infrastructure – container or pod]',
    'What are the scopes of observation when measuring energy or resource consumption?  [Software – service or component (module)]',
    'What are the scopes of observation when measuring energy or resource consumption?  [Software – transaction or request (API call, inference request)]',
    'What are the scopes of observation when measuring energy or resource consumption?  [Software – code level or static analysis]',
    'What are the scopes of observation when measuring energy or resource consumption?  [Other]',
    'Is the assessment of energy or resource consumption automated in your organization? ',
    'In which runtime environments are measurements taken?  [Bare metal]',
    'In which runtime environments are measurements taken?  [Virtual machines]',
    'In which runtime environments are measurements taken?  [Containers]',
    'In which runtime environments are measurements taken?  [Serverless         ]',
    'How likely is your organization to take the following action within the next 12 months?  [Adopt structured guidelines for measuring and interpreting software energy consumption.]',
    'Which decision areas in your organization would be impacted by improved energy or resource measurements?  [Architecture choices]',
    'Which decision areas in your organization would be impacted by improved energy or resource measurements?  [Hardware components (e.g., ARM64, RISC-V, GPUs, servers)]',
    'Which decision areas in your organization would be impacted by improved energy or resource measurements?  [Software libraries (e.g., databases, APIs, Bluetooth Low Energy)]',
    'Which decision areas in your organization would be impacted by improved energy or resource measurements?  [Service right-sizing]',
    'Which decision areas in your organization would be impacted by improved energy or resource measurements?  [Source code (data structures, coding best practices)]',
    'Which decision areas in your organization would be impacted by improved energy or resource measurements?  [Cloud region / provider]',
    'Which decision areas in your organization would be impacted by improved energy or resource measurements?  [Procurement / Service Level Agreements (SLAs)]',
    'Which decision areas in your organization would be impacted by improved energy or resource measurements?  [Product requirements]',
    'Which decision areas in your organization would be impacted by improved energy or resource measurements?  [Sustainability reporting]',
    'Which decision areas in your organization would be impacted by improved energy or resource measurements?  [Other]',
    'Total time',
    'Group time: Context',
    'Question time: contextrole',
    'Question time: contextdomain',
    'Question time: contextsize',
    'Question time: contextresponsabilit',
    'Question time: contextworkload',
    'Question time: contextdecission',
    'Question time: contextsustainabilit',
    'Question time: contextruntime',
    'Question time: contextarchitecture',
    'Question time: contextlocation',
    'Group time: Want',
    'Question time: wantobjectives',
    'Question time: wantinfluence',
    'Question time: wanteffort',
    'Group time: Can',
    'Question time: canbarrier',
    'Question time: canvalueable',
    'Question time: canpolitical',
    'Group time: Do',
    'Question time: dodecission',
    'Question time: dorequirements',
    'Question time: dometrics',
    'Question time: doesgscope',
    'Question time: dolifecycle',
    'Question time: doscope',
    'Question time: doautomation',
    'Question time: docontext',
    'Group time: Outlook',
    'Question time: outlooklikely',
    'Question time: outlookimpact',
]

SUBMITTED_COL = "Date submitted"   # LimeSurvey completion indicator


def load_survey(csv_path, completed_only=True, verbose=False):
    """Load a LimeSurvey export with header-based column alignment.

    - completed_only: keep only submitted responses (Date submitted not null).
    - aligns columns to CANONICAL_COLUMNS *by name* (robust to reordering /
      insertion), raising a clear error if any expected column is absent
      (i.e. the questionnaire changed).
    - runs lightweight integrity checks.

    Returns a DataFrame whose columns are in canonical order, so every script's
    positional access (df.iloc[:, N]) and the index maps above remain valid.
    """
    import pandas as pd
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if completed_only and SUBMITTED_COL in df.columns:
        before = len(df)
        df = df[df[SUBMITTED_COL].notna()].reset_index(drop=True)
        if verbose and before != len(df):
            print(f"[load_survey] dropped {before - len(df)} incomplete response(s); "
                  f"n={len(df)}")
    missing = [h for h in CANONICAL_COLUMNS if h not in df.columns]
    if missing:
        raise ValueError(
            f"[load_survey] {len(missing)} expected column(s) not found in {csv_path} "
            f"(questionnaire layout changed?). First missing: {missing[0]!r}")
    df = df.loc[:, CANONICAL_COLUMNS].reset_index(drop=True)
    _validate_survey(df)
    return df


def _validate_survey(df):
    """Lightweight structural assertions; raise loudly on malformed input."""
    import pandas as pd
    assert df.shape[1] == len(CANONICAL_COLUMNS), "column count mismatch after alignment"
    # Importance Likert items (CO2..Compliance) must be within 1-5 where present.
    for _idx in SURVEY_DIMENSIONS["want_objectives"]["items"].values():
        s = pd.to_numeric(df.iloc[:, _idx], errors="coerce").dropna()
        assert s.between(1, 5).all(), f"Likert item at col {_idx} outside range 1-5"
    # Sustainability maturity values must be in the known set (blank == 'None').
    _mat_idx = SURVEY_DIMENSIONS["context_maturity"]["items"]["Sustainability_maturity"]
    _known = set(SURVEY_DIMENSIONS["context_maturity"]["order"])
    _seen = set(df.iloc[:, _mat_idx].fillna("None").unique())
    _unknown = _seen - _known
    assert not _unknown, f"unexpected maturity value(s): {_unknown}"

