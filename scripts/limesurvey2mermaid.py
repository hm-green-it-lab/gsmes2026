#!/usr/bin/env python3
"""Questionnaire flowchart from a LimeSurvey `.lss` XML export.

Extracts groups and questions, lays them out in administered order, and adds
the edges implied by LimeSurvey conditions and relevance expressions. Labels
are cleaned and wrapped for readable Mermaid nodes.

Standard library only: the flowchart is documentation of the instrument and
should stay buildable without the analysis dependencies.

Usage:  python limesurvey2mermaid.py [survey.lss] [-o out.mmd] [--md out.md]
        Defaults to the export in ../data/ and writes beside it.

`--md` additionally writes the rendered Markdown view. Both outputs are
generated: the diagram exists once, in the .lss, and nothing downstream is
maintained by hand.
"""

from __future__ import annotations

import argparse
import html
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Anchored to this file, so the default works from any working directory.
DEFAULT_LSS = (Path(__file__).resolve().parent.parent
               / "data" / "limesurvey_survey_668719.lss")

# Precompiled patterns used across multiple functions
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass(frozen=True)
class Group:
    gid: str
    title: str
    order: int


@dataclass
class Question:
    qid: str
    gid: str
    code: str
    text: str
    order: int
    relevance: Optional[str]
    mandatory: bool = False


def _as_int(x: Optional[str], default: int = 0) -> int:
    try:
        return int(x) if x is not None else default
    except ValueError:
        return default


def _strip(s: Optional[str]) -> str:
    return (s or "").strip()


def _first_text(elem: Optional[ET.Element]) -> str:
    if elem is None:
        return ""
    return "".join(elem.itertext()).strip()


def _findall_any_ns(root: ET.Element, tag: str) -> List[ET.Element]:
    """Find elements by local-name, ignoring XML namespaces."""
    suffix = "}" + tag
    return [e for e in root.iter() if e.tag == tag or e.tag.endswith(suffix)]


def _find_child_text_any_ns(elem: ET.Element, tag: str) -> str:
    """Return text of the first child element matching `tag` (any ns)."""
    suffix = "}" + tag
    for c in elem:
        if c.tag == tag or c.tag.endswith(suffix):
            return _first_text(c)
    return ""


def clean_human_text(s: str) -> str:
    """Clean text for human-readable labels.

    Actions:
    - Unescape HTML entities
    - Remove HTML tags
    - Normalize non-breaking spaces and collapse whitespace
    """
    s = html.unescape(s or "")
    s = s.replace("\xa0", " ")
    s = _HTML_TAG_RE.sub(" ", s)
    return _WS_RE.sub(" ", s).strip()


def parse_lss(path: Path) -> Tuple[Dict[str, Group], Dict[str, Question], List[dict]]:
    """Parse groups, questions, and conditions from a single XML parse."""
    root = ET.parse(path).getroot()

    # ── Groups ────────────────────────────────────────────────────────────────
    groups: Dict[str, Group] = {}
    group_rows: List[ET.Element] = []
    for g_container in _findall_any_ns(root, "groups"):
        group_rows.extend(_findall_any_ns(g_container, "row"))
    if not group_rows:
        group_rows.extend(_findall_any_ns(root, "group"))

    for row in group_rows:
        gid = _strip(_find_child_text_any_ns(row, "gid")) or _strip(row.get("gid"))
        if not gid:
            continue
        title_raw = (
            _strip(_find_child_text_any_ns(row, "group_name"))
            or _strip(_find_child_text_any_ns(row, "group_title"))
            or _strip(_find_child_text_any_ns(row, "title"))
            or _strip(_find_child_text_any_ns(row, "name"))
            or f"Group {gid}"
        )
        order = _as_int(
            _find_child_text_any_ns(row, "group_order")
            or _find_child_text_any_ns(row, "order"),
            0,
        )
        groups[gid] = Group(gid=gid, title=clean_human_text(title_raw), order=order)

    # ── Questions ─────────────────────────────────────────────────────────────
    questions: Dict[str, Question] = {}
    q_rows: List[ET.Element] = []
    for q_container in _findall_any_ns(root, "questions"):
        q_rows.extend(_findall_any_ns(q_container, "row"))
    if not q_rows:
        q_rows.extend(_findall_any_ns(root, "question"))

    for row in q_rows:
        qid = _strip(_find_child_text_any_ns(row, "qid")) or _strip(row.get("qid"))
        if not qid:
            continue
        gid = _strip(_find_child_text_any_ns(row, "gid")) or _strip(row.get("gid")) or "0"
        code = (
            _strip(_find_child_text_any_ns(row, "title"))
            or _strip(_find_child_text_any_ns(row, "code"))
            or f"Q{qid}"
        )
        text_raw = (
            _strip(_find_child_text_any_ns(row, "question"))
            or _strip(_find_child_text_any_ns(row, "question_text"))
            or _strip(_find_child_text_any_ns(row, "text"))
            or code
        )
        order = _as_int(
            _find_child_text_any_ns(row, "question_order")
            or _find_child_text_any_ns(row, "order"),
            0,
        )
        relevance_raw = (
            _strip(_find_child_text_any_ns(row, "relevance"))
            or _strip(_find_child_text_any_ns(row, "relevance_equation"))
        )
        relevance = clean_human_text(relevance_raw) if relevance_raw else None
        mandatory = _find_child_text_any_ns(row, "mandatory").strip().upper() == "Y"
        questions[qid] = Question(
            qid=qid, gid=gid, code=code,
            text=clean_human_text(text_raw),
            order=order, relevance=relevance, mandatory=mandatory,
        )

    # ── Conditions ────────────────────────────────────────────────────────────
    conditions: List[dict] = []
    cond_rows: List[ET.Element] = []
    for c_container in _findall_any_ns(root, "conditions"):
        cond_rows.extend(_findall_any_ns(c_container, "row"))
    if not cond_rows:
        cond_rows.extend(_findall_any_ns(root, "condition"))

    for row in cond_rows:
        target_qid = _strip(_find_child_text_any_ns(row, "qid")) or _strip(row.get("qid"))
        source_qid = _strip(_find_child_text_any_ns(row, "cqid")) or _strip(row.get("cqid"))
        if not target_qid or not source_qid:
            continue
        cfieldname = _strip(_find_child_text_any_ns(row, "cfieldname")) or _strip(row.get("cfieldname"))
        op = (
            _strip(_find_child_text_any_ns(row, "method"))
            or _strip(_find_child_text_any_ns(row, "op"))
            or _strip(_find_child_text_any_ns(row, "operator"))
            or "=="
        )
        val = _strip(_find_child_text_any_ns(row, "value")) or _strip(_find_child_text_any_ns(row, "cvalue"))
        conditions.append({"qid": target_qid, "cqid": source_qid, "cfieldname": cfieldname, "op": op, "value": val})

    return groups, questions, conditions


def build_order_flow(groups: Dict[str, Group], questions: Dict[str, Question]) -> List[Tuple[str, str, str]]:
    """Create edges representing the default sequential flow between questions."""
    group_list = sorted(groups.values(), key=lambda g: (g.order, _as_int(g.gid, 10**9), g.gid))

    q_by_gid: Dict[str, List[Question]] = {}
    for q in questions.values():
        q_by_gid.setdefault(q.gid, []).append(q)
    for lst in q_by_gid.values():
        lst.sort(key=lambda q: (q.order, _as_int(q.qid, 10**9), q.qid))

    known_gids = {g.gid for g in group_list}
    ordered_questions: List[Question] = []
    for g in group_list:
        ordered_questions.extend(q_by_gid.get(g.gid, []))
    for gid in sorted(set(q_by_gid.keys()) - known_gids):
        ordered_questions.extend(q_by_gid[gid])

    return [(f"q_{a.qid}", f"q_{b.qid}", "") for a, b in zip(ordered_questions, ordered_questions[1:])]


def simplify_cfieldname(cfieldname: str, cqid: str, questions: Dict[str, Question]) -> str:
    """Shorten an SGQ-style field name to a compact label."""
    s = (cfieldname or "").strip().lstrip("+")
    if "X" in s:
        tail = s.split("X")[-1]
        if cqid and tail.startswith(str(cqid)):
            tail = tail[len(str(cqid)):]
        if tail:
            return tail
    q = questions.get(str(cqid))
    return q.code if q else (s or f"Q{cqid}")


def format_rhs_value(val: str) -> str:
    """Normalize RHS values for condition labels."""
    v = (val or "").strip()
    if not v:
        return "?"
    if (v.startswith("'") and v.endswith("'")) or (v.startswith('"') and v.endswith('"')):
        return v
    if _IDENT_RE.fullmatch(v):
        return f"'{v}'"
    return v


def build_condition_edges(conditions: List[dict], questions: Dict[str, Question]) -> List[Tuple[str, str, str]]:
    edges: List[Tuple[str, str, str]] = []
    for c in conditions:
        lhs = simplify_cfieldname(c.get("cfieldname", ""), c["cqid"], questions)
        op = (c.get("op") or "==").strip()
        rhs = format_rhs_value(c.get("value", ""))
        edges.append((f"q_{c['cqid']}", f"q_{c['qid']}", f"if ({lhs} {op} {rhs})"))
    return edges


def extract_relevance_dependencies(questions: Dict[str, Question]) -> List[Tuple[str, str, str]]:
    code_to_qid: Dict[str, str] = {q.code: q.qid for q in questions.values()}

    brace_token = re.compile(r"\{([^}]+)\}")
    sgq_token = re.compile(r"\b\d+X\d+X\d+\b")
    code_token = re.compile(r"\b[A-Za-z][A-Za-z0-9_]*\b")

    reserved = {
        "and", "or", "not", "in", "eq", "ne", "lt", "lte", "gt", "gte",
        "sum", "count", "is_empty", "is_null", "strlen", "if",
        "true", "false", "null",
        "NAOK", "NOK", "OK",
    }

    edges: List[Tuple[str, str, str]] = []
    for q in questions.values():
        if not q.relevance:
            continue
        expr = q.relevance
        refs: Set[str] = set()
        for m in brace_token.findall(expr):
            refs.add(m.strip().split(".")[0].strip())
        refs.update(sgq_token.findall(expr))
        for t in code_token.findall(expr):
            if t.lower() not in reserved and t in code_to_qid:
                refs.add(t)
        for r in sorted(refs):
            if r in code_to_qid:
                src_qid = code_to_qid[r]
                if src_qid != q.qid:
                    edges.append((f"q_{src_qid}", f"q_{q.qid}", "relevance"))
    return edges


def mermaid_escape(text: str, max_len: int = 80) -> str:
    """Sanitize short labels for Mermaid (collapse whitespace, limit length)."""
    text = _WS_RE.sub(" ", (text or "")).strip()
    if len(text) > max_len:
        text = text[: max_len - 1] + "…"
    return text.replace('"', "'")


def wrap_for_mermaid(
    s: str,
    line_width: int = 22,
    max_words_per_line: int = 2,
    max_lines: int = 4,
) -> str:
    """Wrap text for Mermaid node labels.

    Builds lines from words, enforcing both a per-line word limit and a
    character width. If the text exceeds ``max_lines``, the last line is
    suffixed with an ellipsis.
    """
    s = clean_human_text(s).replace('"', "'")
    words = s.split()
    if not words:
        return ""

    lines: List[str] = []
    cur: List[str] = []

    def flush() -> None:
        if cur:
            lines.append(" ".join(cur))
            cur.clear()

    for w in words:
        if len(lines) >= max_lines:
            # Mark truncation and stop
            if lines:
                lines[-1] += "…"
            break
        if not cur:
            cur.append(w)
        else:
            candidate = " ".join(cur + [w])
            if len(cur) >= max_words_per_line or len(candidate) > line_width:
                flush()
                if len(lines) >= max_lines:
                    lines[-1] += "…"
                    break
                cur.append(w)
            else:
                cur.append(w)
    else:
        flush()

    return "<br/>".join(lines)


def render_mermaid(
    groups: Dict[str, Group],
    questions: Dict[str, Question],
    edges: List[Tuple[str, str, str]],
    include_groups: bool = True,
) -> str:
    q_by_gid: Dict[str, List[Question]] = {}
    for q in questions.values():
        q_by_gid.setdefault(q.gid, []).append(q)
    for lst in q_by_gid.values():
        lst.sort(key=lambda x: (x.order, _as_int(x.qid, 10**9), x.qid))

    group_list = sorted(groups.values(), key=lambda g: (g.order, _as_int(g.gid, 10**9), g.gid))
    known_gids = {g.gid for g in group_list}
    unknown_gids = sorted(set(q_by_gid.keys()) - known_gids)

    lines: List[str] = [
        "flowchart TD",
        "  classDef mandatory fill:#ffe0e0,stroke:#cc0000,stroke-width:2px,color:#000",
    ]

    def node_line(q: Question) -> str:
        code = q.code + " *" if q.mandatory else q.code
        label = f"{code}<br/>{wrap_for_mermaid(q.text)}"
        return f'  q_{q.qid}["{label}"]'

    if include_groups and groups:
        for g in group_list:
            if g.gid not in q_by_gid:
                continue
            g_title = mermaid_escape(g.title, max_len=60) or f"Group {g.gid}"
            lines.append(f'  subgraph g_{g.gid}["{g_title}"]')
            for q in q_by_gid[g.gid]:
                lines.append(node_line(q))
            lines.append("  end")
        for gid in unknown_gids:
            lines.append(f'  subgraph g_{gid}["Group {gid}"]')
            for q in q_by_gid[gid]:
                lines.append(node_line(q))
            lines.append("  end")
    else:
        for q in sorted(questions.values(), key=lambda x: (x.gid, x.order, _as_int(x.qid, 10**9), x.qid)):
            lines.append(node_line(q))

    seen: Set[Tuple[str, str, str]] = set()
    for a, b, lbl in edges:
        key = (a, b, lbl)
        if key in seen:
            continue
        seen.add(key)
        if lbl:
            lines.append(f'  {a} -->|"{mermaid_escape(lbl, max_len=70)}"| {b}')
        else:
            lines.append(f"  {a} --> {b}")

    mandatory_ids = [f"q_{q.qid}" for q in questions.values() if q.mandatory]
    if mandatory_ids:
        lines.append("  class " + ",".join(mandatory_ids) + " mandatory")

    return "\n".join(lines) + "\n"


def render_markdown(mermaid: str, questions: Dict[str, Question]) -> str:
    """The Markdown view: the same diagram, in a file that renders in a browser.

    Written by this script rather than kept by hand -- the previous copy had
    drifted a generator version behind the .mmd it was pasted from.
    """
    n_questions = len(questions)
    n_mandatory = sum(1 for q in questions.values() if q.mandatory)
    return f"""# Survey Flowchart

The questionnaire as administered: question order, one subgraph per dimension
group, and every conditional edge LimeSurvey enforces. {n_questions} closed
questions, {n_mandatory} of them mandatory (marked `*`); conditional routing
means no respondent sees all of them.

Generated, together with `limesurvey_survey_668719.mmd`, from the `.lss`
export. Regenerate both after any change to the instrument, from
`replication/`:

```bash
python scripts/limesurvey2mermaid.py \\
    -o data/limesurvey_survey_668719.mmd \\
    --md data/SurveyFlowchart.md
```

Two conditions are worth reading off the diagram. Role selection is
multi-select, and the `SQ002` (Technical) sub-answer of `contextrole` gates the
technical branches, so one respondent may see several. And `doesgscope` (CO2e
scope differentiation) is shown only to respondents whose `dometrics` answer
already includes `SQ003`, that is, only to those who track CO2e.

```mermaid
{mermaid.rstrip()}
```
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("lss", type=Path, nargs="?", default=DEFAULT_LSS, help="Path to LimeSurvey .lss export file")
    ap.add_argument("-o", "--out", type=Path, default=None, help="Output .mmd file (default: <lss>.mmd alongside input)")
    ap.add_argument(
        "--mode",
        choices=["flow", "deps"],
        default="flow",
        help="flow = default order flow + conditions; deps = dependency graph from relevance/conditions only",
    )
    ap.add_argument("--no-groups", action="store_true", help="Do not create Mermaid subgraphs per LimeSurvey group")
    ap.add_argument("--md", type=Path, default=None,
                    help="Also write the rendered Markdown view (embeds the diagram)")
    args = ap.parse_args()

    if not args.lss.exists():
        print(f"File not found: {args.lss}", file=sys.stderr)
        return 2

    groups, questions, conditions = parse_lss(args.lss)

    edges: List[Tuple[str, str, str]] = []
    if args.mode == "flow":
        edges.extend(build_order_flow(groups, questions))
        edges.extend(build_condition_edges(conditions, questions))
        edges.extend(extract_relevance_dependencies(questions))
    else:
        edges.extend(build_condition_edges(conditions, questions))
        edges.extend(extract_relevance_dependencies(questions))

    mermaid = render_mermaid(groups, questions, edges, include_groups=(not args.no_groups))

    out = args.out or args.lss.with_suffix(".mmd")
    out.write_text(mermaid, encoding="utf-8")
    print(f"Wrote Mermaid to: {out}")

    if args.md:
        args.md.write_text(render_markdown(mermaid, questions), encoding="utf-8")
        print(f"Wrote Markdown to: {args.md}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
