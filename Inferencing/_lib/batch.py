"""
_lib/batch.py
-------------
Core batch-inference logic.

run(input_csv, output_csv, model_cfg)
  - If the output file does not exist  → full inference from scratch.
  - If the output file already exists  → find MISSING items (in source but
    absent from output) AND CORRUPTED items, infer them, and loop until
    every item is present and clean.

Checkpointing: the output CSV is written after every single item so that
a mid-run crash loses at most one item of work.
"""

import os
import time
import pandas as pd

from . import config
from .inference import format_item, get_cot_answer, is_corrupted


# ── Public entry point ────────────────────────────────────────────────────────
def run(input_csv: str, output_csv: str, model_cfg: dict) -> None:
    """
    Ensure *output_csv* contains a clean result for every row in *input_csv*.

    On the first invocation (no output file) this is a full inference run.
    On subsequent invocations it detects and repairs missing / corrupted rows,
    retrying until the file is completely clean.
    """
    src     = pd.read_csv(input_csv, encoding="utf-8-sig")
    all_ids = sorted(src["ItemNum"].tolist())
    src_idx = src.set_index("ItemNum")

    # Create an empty output file if this is a fresh run
    if not os.path.exists(output_csv):
        print(f"  Output file not found — starting fresh inference "
              f"({len(all_ids)} items).")
        pd.DataFrame(columns=["ItemNum", "thinking", "answer", "confidence"]).to_csv(
            output_csv, index=False, encoding="utf-8-sig"
        )

    # ── Main loop: keep going until every item is present AND clean ───────────
    pass_num = 0
    while True:
        out          = pd.read_csv(output_csv, encoding="utf-8-sig")
        present_ids  = set(out["ItemNum"].tolist())

        missing_ids   = [i for i in all_ids if i not in present_ids]
        corrupted_ids = out[out.apply(is_corrupted, axis=1)]["ItemNum"].tolist()
        problem_ids   = sorted(set(missing_ids + corrupted_ids))

        if not problem_ids:
            msg = (f"  All {len(all_ids)} items already present and clean."
                   if pass_num == 0
                   else f"  All {len(all_ids)} items present and clean "
                        f"after {pass_num} pass(es).")
            print(msg)
            break

        pass_num += 1

        # Summarise what needs doing
        if missing_ids:
            shown = missing_ids[:10]
            tail  = f"... (+{len(missing_ids)-10} more)" if len(missing_ids) > 10 else ""
            print(f"  {len(missing_ids)} missing:   {shown}{tail}")
        if corrupted_ids:
            shown = sorted(corrupted_ids)[:10]
            tail  = f"... (+{len(corrupted_ids)-10} more)" if len(corrupted_ids) > 10 else ""
            print(f"  {len(corrupted_ids)} corrupted: {shown}{tail}")

        label = "pass-1 (initial)" if pass_num == 1 else f"repair pass {pass_num}"
        print(f"\n  [{label}] Processing {len(problem_ids)} item(s)...")

        # Drop corrupted rows before re-inferring so they don't block the check
        if corrupted_ids:
            out_clean = out[~out["ItemNum"].isin(corrupted_ids)]
            out_clean.to_csv(output_csv, index=False, encoding="utf-8-sig")

        _infer_items(problem_ids, src_idx, output_csv, model_cfg)


# ── Internal helper ───────────────────────────────────────────────────────────
def _infer_items(
    item_ids: list,
    src_idx:  pd.DataFrame,
    output_csv: str,
    model_cfg:  dict,
) -> None:
    """Infer *item_ids* one by one, checkpointing to *output_csv* after each."""
    total = len(item_ids)
    for i, item_num in enumerate(item_ids, start=1):
        if item_num not in src_idx.index:
            print(f"  [{i}/{total}] ItemNum {item_num} — NOT FOUND in source, skipping.")
            continue

        row       = src_idx.loc[item_num]
        item_text = format_item(row)

        print(f"  [{i}/{total}] ItemNum {item_num} ... ", end="", flush=True)
        try:
            result = get_cot_answer(item_text, model_cfg)
            record = {
                "ItemNum":    item_num,
                "thinking":   result["thinking"],
                "answer":     result["answer"],
                "confidence": result["confidence"],
            }
            print(f"-> {result['answer']} (confidence: {result['confidence']})")
        except Exception as exc:
            print(f"ERROR: {exc}")
            record = {
                "ItemNum":    item_num,
                "thinking":   "",
                "answer":     f"ERROR: {exc}",
                "confidence": "",
            }

        # Checkpoint: append the new record and save immediately
        existing = pd.read_csv(output_csv, encoding="utf-8-sig")
        updated  = pd.concat([existing, pd.DataFrame([record])], ignore_index=True)
        updated.to_csv(output_csv, index=False, encoding="utf-8-sig")

        time.sleep(config.REQUEST_DELAY)
