"""Turn responses.jsonl into tidy dataframes.

Outputs (in data/):
  responses.parquet / .csv   one row per LLM answer          (21 080 rows)
  respondents.parquet / .csv one row per simulated "person"  (340 rows)
                             = persona x prompt template, i.e. one full 62-question test

Usage: uv run scripts/build_dataset.py /path/to/responses.jsonl
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "responses.jsonl")
DATA = Path("data")

# Answer index convention used by the reference website (js/script.js)
ANSWER_IDX = {"strongly disagree": 0, "disagree": 1, "agree": 2, "strongly agree": 3}
IDX_LABEL = {0: "Strongly disagree", 1: "Disagree", 2: "Agree", 3: "Strongly agree"}
PERSONA_FIELDS = ["age", "gender", "nation", "political_orientation"]

# The e0/s0 offsets and divisors from the reference website's scoring
E0, S0, E_DIV, S_DIV = 0.38, 2.41, 8.0, 19.5


# --------------------------------------------------------------------------- parsing
def normalise_selection(raw: str | None) -> str | None:
    """'Strongly Disagree', ' agree', 'Dis agree' -> canonical lowercase; junk -> None."""
    if raw is None:
        return None
    s = re.sub(r"\s+", " ", raw.strip().lower()).replace("dis agree", "disagree")
    return s if s in ANSWER_IDX else None


def parse_closed(answer: str) -> tuple[str | None, str | None]:
    """Return (raw_selection, justification) from the '{ "selection": ..., "response": ... }' string."""
    try:
        # raw_decode tolerates text after the JSON object ('{ "selection": "X" }\n\nMy verdict ...')
        d, end = json.JSONDecoder().raw_decode(answer.lstrip())
        rest = answer.lstrip()[end:].strip()
        return d.get("selection"), d.get("response") or rest or None
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass
    sel = re.search(r'"selection"\s*:\s*"([^"]*)"', answer)
    resp = re.search(r'"response"\s*:\s*"(.*)', answer, re.S)
    text = resp.group(1) if resp else None
    if text:
        text = re.sub(r'"?\s*\}?\s*$', "", text).strip()
    return (sel.group(1) if sel else None), text


def clean_open(completion: str) -> str:
    return re.sub(r'"?\s*\}?\s*$', "", completion).strip()


def persona_of(r: dict) -> tuple[str, str]:
    for f in PERSONA_FIELDS:
        if f in r:
            return f, str(r[f])
    raise ValueError(r)


# --------------------------------------------------------------------------- build
def build_responses() -> pd.DataFrame:
    questions = pd.read_csv(DATA / "pct_questions.csv")
    qid = dict(zip(questions.question, questions.question_id))

    # template id = rank of the instruction text within its domain (stable alphabetical order)
    raw = [json.loads(l) for l in SRC.open()]
    templates = {}
    for dom in ("closed_domain", "open_domain"):
        texts = sorted({r["instruction"]["text"] for r in raw if r["instruction"]["type"] == dom})
        templates.update({t: i for i, t in enumerate(texts)})

    rows = []
    for r in raw:
        ptype, pval = persona_of(r)
        domain = r["instruction"]["type"].replace("_domain", "")
        text = r["instruction"]["text"]
        row = {
            "prompt_index": r["prompt_index"],
            "question_id": qid[r["question"]],
            "question": r["question"],
            "domain": domain,
            "template_id": templates[text],
            "prompt_style": ("justify" if "justify" in text else "argue") if domain == "closed" else "open",
            "persona_type": ptype,
            "persona_value": pval,
            "persona": f"{ptype}={pval}",
            **{f: r.get(f) for f in PERSONA_FIELDS},
            "options_order": "|".join(r["options"]) if r["options"] else None,
        }
        if domain == "closed":
            sel_raw, resp = parse_closed(r["answer"])
            sel = normalise_selection(sel_raw)
            row.update(selection_raw=sel_raw, selection=sel, response_text=resp)
        else:
            # open domain: free text, no selection -> needs classification later
            row.update(selection_raw=None, selection=None, response_text=clean_open(r["completion"]))
        rows.append(row)

    df = pd.DataFrame(rows)
    df["answer_idx"] = df.selection.map(ANSWER_IDX).astype("Int64")
    df["answer_label"] = df.answer_idx.map(IDX_LABEL)
    df["valid"] = df.answer_idx.notna()
    df["respondent_id"] = df.domain + "|" + df.persona + "|t" + df.template_id.astype(str)

    # Position of the chosen option in the shuffled list (for position-bias checks)
    def pos(row):
        if row.options_order is None or pd.isna(row.answer_label):
            return pd.NA
        opts = [o.lower() for o in row.options_order.split("|")]
        return opts.index(row.answer_label.lower()) + 1 if row.answer_label.lower() in opts else pd.NA

    df["chosen_position"] = df.apply(pos, axis=1).astype("Int64")

    # Per-answer contribution to the compass axes
    q = questions.set_index("question_id")
    labels = ["sd", "d", "a", "sa"]
    e_tab = q[[f"econ_{l}" for l in labels]].to_numpy()
    s_tab = q[[f"soc_{l}" for l in labels]].to_numpy()
    idx = df.answer_idx.fillna(-1).astype(int).to_numpy()
    ok = idx >= 0
    qi = df.question_id.to_numpy()
    df["econ_pts"] = np.where(ok, e_tab[qi, np.where(ok, idx, 0)], np.nan)
    df["soc_pts"] = np.where(ok, s_tab[qi, np.where(ok, idx, 0)], np.nan)
    return df.sort_values("prompt_index").reset_index(drop=True)


def build_respondents(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (domain, persona, template): a full 62-question run -> compass position."""
    g = df.groupby("respondent_id")
    out = g.agg(
        domain=("domain", "first"), persona_type=("persona_type", "first"),
        persona_value=("persona_value", "first"), persona=("persona", "first"),
        template_id=("template_id", "first"), n_questions=("question_id", "nunique"),
        n_valid=("valid", "sum"),
        econ_sum=("econ_pts", "sum"), soc_sum=("soc_pts", "sum"),
    ).reset_index()
    # Same formula as the website: sum / divisor + offset, clipped to the -10..10 grid
    out["economic"] = (out.econ_sum / E_DIV + E0).clip(-10, 10).round(2)
    out["social"] = (out.soc_sum / S_DIV + S0).clip(-10, 10).round(2)
    out["complete"] = out.n_valid == out.n_questions
    return out


if __name__ == "__main__":
    df = build_responses()
    resp = build_respondents(df)
    for name, d in [("responses", df), ("respondents", resp)]:
        d.to_parquet(DATA / f"{name}.parquet", index=False)
        d.to_csv(DATA / f"{name}.csv", index=False)
    print(f"responses:   {df.shape}  | respondents: {resp.shape}")
    print(df.groupby("domain").valid.agg(["size", "sum", "mean"]))
