import re
import time
import os
import pandas as pd
import google.auth
import google.auth.transport.requests
from openai import OpenAI

# ── Configuration ─────────────────────────────────────────────────────────────
PROJECT_ID = "project-8c9c1880-369b-4376-b02"

# Models to run. Each entry defines its own MaaS endpoint and region so that
# Gemma (global endpoint) and Llama (regional endpoint) both work correctly.
MODELS = {
    "gemma": {
        "model_id": "google/gemma-4-26b-a4b-it-maas",
        "endpoint": "aiplatform.googleapis.com",
        "region":   "global",
    },
    "llama": {
        "model_id": "meta/llama-3.3-70b-instruct-maas",
        "endpoint": "us-central1-aiplatform.googleapis.com",
        "region":   "us-central1",
    },
}

# Max output tokens per request (CoT paragraph + single-letter answer)
MAX_OUTPUT_TOKENS = 512

# Seconds to wait between requests (avoids hitting quota limits)
REQUEST_DELAY = 1.0

# Answer-choice labels in column order
ANSWER_LABELS = list("ABCDEFGHIJ")
ANSWER_COLS   = [f"Answer__{l}" for l in ANSWER_LABELS]

# Paths
DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "Data", "Our")
TRAIN_CSV = os.path.join(DATA_DIR, "train_final.csv")
TEST_CSV  = os.path.join(DATA_DIR, "test_final.csv")

# ── Auth ───────────────────────────────────────────────────────────────────────
# ADC credentials are loaded once; token is refreshed automatically when needed.
credentials, _ = google.auth.default(
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)
_auth_request = google.auth.transport.requests.Request()


def _get_client(model_cfg: dict) -> OpenAI:
    """Return an OpenAI client with a refreshed bearer token for the given model."""
    if not credentials.valid:
        credentials.refresh(_auth_request)
    base_url = (
        f"https://{model_cfg['endpoint']}/v1/projects/{PROJECT_ID}"
        f"/locations/{model_cfg['region']}/endpoints/openapi"
    )
    return OpenAI(base_url=base_url, api_key=credentials.token)


# ── Zero-shot CoT prompt ───────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a medical expert. Your task is to answer the given medical exam question by reasoning through it step by step.

Rules (strictly follow):
1. Inside <thinking> tags, work through the clinical reasoning: consider the stem, eliminate incorrect options, and explain why the correct option is right. Keep it focused — a short paragraph, no repetition.
2. Inside <answer> tags write only the single letter of the correct answer choice (e.g. A, B, C ...). Nothing else.

Output format:
<thinking>[clinical reasoning paragraph]</thinking>
<answer>[single letter]</answer>"""


# ── Item formatting ────────────────────────────────────────────────────────────
def format_item(row: pd.Series) -> str:
    """Convert a CSV row into a prompt string for the model."""
    lines = [str(row["ItemStem_Text"]).strip()]
    for label, col in zip(ANSWER_LABELS, ANSWER_COLS):
        val = row.get(col)
        if pd.notna(val) and str(val).strip():
            lines.append(f"{label}) {str(val).strip()}")
    return "\n".join(lines)


# ── Single-item inference ──────────────────────────────────────────────────────
def get_cot_answer(item_text: str, model_cfg: dict) -> dict:
    """
    Run zero-shot CoT inference on one medical exam item.

    Returns
    -------
    dict with keys:
        "thinking"          - CoT reasoning from <thinking> tags
        "answer"            - predicted answer letter from <answer> tags
        "raw"               - full model output (for debugging)
        "prompt_tokens"     - input token count
        "completion_tokens" - output token count
    """
    client = _get_client(model_cfg)
    response = client.chat.completions.create(
        model=model_cfg["model_id"],
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": item_text},
        ],
        temperature=0,
        max_tokens=MAX_OUTPUT_TOKENS,
    )

    raw = response.choices[0].message.content or ""

    thinking_match = re.search(r"<thinking>(.*?)</thinking>", raw, re.DOTALL)
    answer_match   = re.search(r"<answer>(.*?)</answer>",     raw, re.DOTALL)

    usage = response.usage.model_dump() if response.usage else {}
    return {
        "thinking":          thinking_match.group(1).strip() if thinking_match else "",
        "answer":            answer_match.group(1).strip()   if answer_match   else raw.strip(),
        "raw":               raw,
        "prompt_tokens":     usage.get("prompt_tokens", ""),
        "completion_tokens": usage.get("completion_tokens", ""),
    }


# ── Batch inference with checkpointing ────────────────────────────────────────
def run_batch(input_csv: str, output_csv: str, model_cfg: dict) -> pd.DataFrame:
    """
    Run get_cot_answer on every row of input_csv and save results to output_csv.
    Already-processed ItemNums are skipped so the job can be safely restarted
    after a failure (checkpoint behaviour).

    Parameters
    ----------
    input_csv  : path to train_final.csv or test_final.csv
    output_csv : path where results are written incrementally
    model_cfg  : entry from MODELS dict

    Returns
    -------
    DataFrame with all results (previously completed + newly added).
    """
    df = pd.read_csv(input_csv)

    if os.path.exists(output_csv):
        done_df  = pd.read_csv(output_csv)
        done_ids = set(done_df["ItemNum"].tolist())
        print(f"Resuming — {len(done_ids)} items already done, "
              f"{len(df) - len(done_ids)} remaining.")
    else:
        done_df  = pd.DataFrame()
        done_ids = set()

    rows_to_process = df[~df["ItemNum"].isin(done_ids)]
    new_records = []

    for i, (_, row) in enumerate(rows_to_process.iterrows(), start=1):
        item_num  = row["ItemNum"]
        item_text = format_item(row)

        print(f"[{i}/{len(rows_to_process)}] ItemNum {item_num} ... ", end="", flush=True)

        try:
            result = get_cot_answer(item_text, model_cfg)
            record = {
                "ItemNum":           item_num,
                "thinking":          result["thinking"],
                "answer":            result["answer"],
                "prompt_tokens":     result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
            }
            print(f"-> {result['answer']}")
        except Exception as e:
            print(f"ERROR: {e}")
            record = {
                "ItemNum":           item_num,
                "thinking":          "",
                "answer":            f"ERROR: {e}",
                "prompt_tokens":     "",
                "completion_tokens": "",
            }

        new_records.append(record)

        # Checkpoint after every item so progress is never lost
        checkpoint_df = pd.DataFrame(new_records)
        combined      = pd.concat([done_df, checkpoint_df], ignore_index=True)
        combined.to_csv(output_csv, index=False, encoding="utf-8-sig")

        time.sleep(REQUEST_DELAY)

    print(f"\nDone. Results saved to {output_csv}")
    return pd.read_csv(output_csv)


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for model_name, model_cfg in MODELS.items():
        train_out = os.path.join(DATA_DIR, f"train_results_{model_name}.csv")
        test_out  = os.path.join(DATA_DIR, f"test_results_{model_name}.csv")

        print(f"\n{'='*60}")
        print(f"Model : {model_name}  ({model_cfg['model_id']})")
        print(f"{'='*60}")

        print("\n--- TRAIN set ---")
        run_batch(TRAIN_CSV, train_out, model_cfg)

        print("\n--- TEST set ---")
        run_batch(TEST_CSV, test_out, model_cfg)
