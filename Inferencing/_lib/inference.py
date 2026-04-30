"""
_lib/inference.py
-----------------
Prompt template, item formatter, single-item inference, and corruption checker.
"""

import re
import pandas as pd

from . import config
from .auth import get_client

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a third-year medical student taking a licensing exam. You have solid but imperfect knowledge — you know core concepts well but sometimes struggle with rare diseases, subtle distinctions between similar answer choices, or complex multi-step reasoning.

For each question, reason through it as a student would: recall what you know, consider all options, and acknowledge uncertainty where it exists.

Rules (strictly follow):
1. Inside <thinking> tags, reason through the question as a medical student: recall relevant knowledge, weigh the options, and note where you are unsure or find the choices close. Write as a flowing paragraph — do NOT reproduce the answer choices as a lettered list (no "A) ...", "B) ...", etc.).
2. Inside <answer> tags write only the single letter of your best answer choice (e.g. A, B, C ...). Nothing else.
3. Inside <confidence> tags write a single integer from 1 to 5 reflecting how confident you are in your answer. Nothing else.
   1 = guessing, very unsure
   2 = leaning toward this answer but quite uncertain
   3 = moderately confident
   4 = fairly confident, minor doubt
   5 = very confident

Output format:
<thinking>[student reasoning paragraph]</thinking>
<answer>[single letter]</answer>
<confidence>[1-5]</confidence>"""


# ── Item formatting ───────────────────────────────────────────────────────────
def format_item(row: pd.Series) -> str:
    """Convert a source CSV row into the prompt text sent to the model."""
    lines = [str(row["ItemStem_Text"]).strip()]
    for label, col in zip(config.ANSWER_LABELS, config.ANSWER_COLS):
        val = row.get(col)
        if pd.notna(val) and str(val).strip():
            lines.append(f"{label}) {str(val).strip()}")
    return "\n".join(lines)


# ── Single-item inference ─────────────────────────────────────────────────────
def get_cot_answer(item_text: str, model_cfg: dict) -> dict:
    """
    Run zero-shot CoT inference on one item.

    Returns dict with keys: thinking, answer, confidence, raw.
    """
    client = get_client(model_cfg)
    response = client.chat.completions.create(
        model=model_cfg["model_id"],
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": item_text},
        ],
        temperature=0,
        max_tokens=config.MAX_OUTPUT_TOKENS,
    )
    raw = response.choices[0].message.content or ""

    thinking_m   = re.search(r"<thinking>(.*?)</thinking>",    raw, re.DOTALL)
    answer_m     = re.search(r"<answer>(.*?)</answer>",         raw, re.DOTALL)
    confidence_m = re.search(r"<confidence>(.*?)</confidence>", raw, re.DOTALL)

    thinking_raw   = thinking_m.group(1).strip() if thinking_m else ""
    # Collapse embedded newlines → spaces so valid long-paragraph responses
    # pass the is_corrupted() check without needing another repair pass.
    thinking_clean = " ".join(thinking_raw.split("\n")).strip()

    return {
        "thinking":   thinking_clean,
        "answer":     answer_m.group(1).strip()      if answer_m     else raw.strip(),
        "confidence": confidence_m.group(1).strip()  if confidence_m else "",
        "raw":        raw,
    }


# ── Corruption detection ──────────────────────────────────────────────────────
def is_corrupted(row: pd.Series) -> bool:
    """
    Return True if a result row is considered corrupted:
      - answer is not a single letter A-J
      - confidence is missing / empty
      - thinking contains a newline  (indicates truncation or structured listing)
    """
    a = str(row["answer"]).strip()
    c = str(row["confidence"]).strip()
    t = str(row["thinking"]) if pd.notna(row.get("thinking")) else ""

    return (
        a not in config.VALID_LETTERS
        or c in {"nan", "", "None"}
        or "\n" in t
    )
