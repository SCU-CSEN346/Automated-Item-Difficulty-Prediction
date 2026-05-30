"""
demo.py — Inference demo for the ModernBERT + LoRA Dual-Gate model
         (best checkpoint: diff_r0.302_tau0.168__rtime_r0.610_tau0.471)

Usage
-----
# Batch mode — runs on the full test set and reports metrics
python demo.py

# Single-question mode — show one question by its ItemNum
python demo.py --item 552

# Batch mode but skip metric computation (no ground-truth needed)
python demo.py --no-labels
"""

import argparse
import os
import textwrap

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, TaskType, get_peft_model
from scipy.stats import kendalltau, pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from transformers import AutoModel, AutoTokenizer

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))
MODEL_PT     = os.path.join(_HERE, "Results",
               "diff_r0.302_tau0.168__rtime_r0.610_tau0.471", "model.pt")
TRAIN_CSV    = os.path.join(_HERE, "Data", "Our", "train_final_extended.csv")
TEST_CSV     = os.path.join(_HERE, "Data", "Our", "test_final_extended.csv")

# ── Hyper-parameters (must match training) ────────────────────────────────────
MODEL_NAME   = "answerdotai/ModernBERT-base"
MAX_LEN      = 512    # Q-stream: question + answer choices
MAX_LEN_LLM  = 1024   # LLM-stream: chain-of-thought
BATCH_SIZE   = 8
_CONF_COLS   = ["gemma_confidence", "llama_confidence",
                "gemma_correct",    "llama_correct"]
STR_COLS     = [
    "Answer__A", "Answer__B", "Answer__C", "Answer__D", "Answer__E",
    "Answer__F", "Answer__G", "Answer__H", "Answer__I", "Answer__J",
    "Answer_Key", "Answer_Text", "all_input", "q_answers_input",
    "answers_input", "q_a_input", "llms_a_input", "q_llms_a_input",
    "gemma_thinking", "llama_thinking",
]


# ── Model definition (identical to training notebook) ─────────────────────────

class DualGateModel(torch.nn.Module):
    """Dual-stream ModernBERT encoder with two independent regression heads.

    Stream Q   : encodes q_answers_input  (question + choices, MAX_LEN=512)
    Stream LLM : encodes llms_a_input     (LLM chain-of-thought, MAX_LEN=1024)
    Scalars    : gemma_confidence, llama_confidence, gemma_correct, llama_correct

    Fusion: cat([mean_pool_q, mean_pool_llm, scalars]) → [B, 2H+4]
    Heads:  head_difficulty    → normalised difficulty   [0, 1]
            head_response_time → normalised response time [0, 1]
    """

    NUM_CONF = 4

    def __init__(self, encoder: torch.nn.Module, hidden_size: int):
        super().__init__()
        self.encoder            = encoder
        in_features             = hidden_size * 2 + self.NUM_CONF
        self.head_difficulty    = torch.nn.Linear(in_features, 1)
        self.head_response_time = torch.nn.Linear(in_features, 1)

    def _encode(self, input_ids, attention_mask):
        out    = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state                # [B, T, H]
        mask   = attention_mask.unsqueeze(-1).float() # [B, T, 1]
        summed = (hidden * mask).sum(dim=1)           # [B, H]
        counts = mask.sum(dim=1).clamp(min=1e-9)      # [B, 1]
        return summed / counts                        # [B, H]

    def forward(self, input_ids_q, attention_mask_q,
                input_ids_llm, attention_mask_llm, conf):
        pool_q   = self._encode(input_ids_q,   attention_mask_q)
        pool_llm = self._encode(input_ids_llm, attention_mask_llm)
        fused    = torch.cat([pool_q, pool_llm, conf], dim=-1)
        pred_diff  = torch.sigmoid(self.head_difficulty(fused)).squeeze(-1)
        pred_rtime = torch.sigmoid(self.head_response_time(fused)).squeeze(-1)
        return pred_diff, pred_rtime


def build_model(device: torch.device) -> DualGateModel:
    print("Loading ModernBERT-base + LoRA adapters …")
    base = AutoModel.from_pretrained(MODEL_NAME)
    lora_cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=32,
        lora_alpha=32,
        target_modules="all-linear",
        lora_dropout=0.1,
        bias="none",
    )
    encoder = get_peft_model(base, lora_cfg)
    model   = DualGateModel(encoder, base.config.hidden_size)
    state   = torch.load(MODEL_PT, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"Loaded weights from: {MODEL_PT}\n")
    return model


# ── Scalers ───────────────────────────────────────────────────────────────────

def fit_scalers(train_df: pd.DataFrame):
    """Fit MinMaxScalers on the training set — same as training notebook."""

    def _fit(series):
        sc  = MinMaxScaler()
        arr = np.array(series, dtype=np.float32).reshape(-1, 1)
        sc.fit(arr)
        return sc

    sc_diff  = _fit(train_df["Difficulty"])
    sc_rtime = _fit(train_df["Response_Time"])
    sc_gemma = _fit(train_df["gemma_confidence"].fillna(0.5))
    sc_llama = _fit(train_df["llama_confidence"].fillna(0.5))
    return sc_diff, sc_rtime, sc_gemma, sc_llama


# ── Tokenisation ──────────────────────────────────────────────────────────────

def encode_texts(tokenizer, texts, max_len: int):
    enc = tokenizer(
        [str(t) for t in texts],
        add_special_tokens=True,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    return enc["input_ids"], enc["attention_mask"]


# ── Pre-processing (mirrors training notebook) ────────────────────────────────

def preprocess(df: pd.DataFrame, sc_diff, sc_rtime, sc_gemma, sc_llama,
               has_labels: bool):
    df = df.copy()
    df[STR_COLS] = df[STR_COLS].astype(str)

    # Confidence scores — transform with training scalers
    df["gemma_confidence"] = sc_gemma.transform(
        df["gemma_confidence"].fillna(0.5).values.reshape(-1, 1)
    ).flatten()
    df["llama_confidence"] = sc_llama.transform(
        df["llama_confidence"].fillna(0.5).values.reshape(-1, 1)
    ).flatten()

    # Correctness flags
    df["gemma_correct"] = (
        df["gemma_answer"].fillna("").str.strip().str.upper() ==
        df["Answer_Key"].fillna("").str.strip().str.upper()
    ).astype(np.float32)
    df["llama_correct"] = (
        df["llama_answer"].fillna("").str.strip().str.upper() ==
        df["Answer_Key"].fillna("").str.strip().str.upper()
    ).astype(np.float32)

    # Ground-truth labels (scaled) — only when available
    if has_labels:
        labels_diff  = torch.tensor(
            sc_diff.transform(
                np.array(df["Difficulty"],    dtype=np.float32).reshape(-1, 1)
            ).flatten(), dtype=torch.float32)
        labels_rtime = torch.tensor(
            sc_rtime.transform(
                np.array(df["Response_Time"], dtype=np.float32).reshape(-1, 1)
            ).flatten(), dtype=torch.float32)
    else:
        labels_diff  = torch.zeros(len(df), dtype=torch.float32)
        labels_rtime = torch.zeros(len(df), dtype=torch.float32)

    return df, labels_diff, labels_rtime


# ── Batch inference ───────────────────────────────────────────────────────────

def run_batch_inference(model, tokenizer, df, labels_diff, labels_rtime,
                        sc_diff, sc_rtime, device, has_labels: bool):
    ids_q,   mask_q   = encode_texts(tokenizer, df.q_answers_input.values, MAX_LEN)
    ids_llm, mask_llm = encode_texts(tokenizer, df.llms_a_input.values,    MAX_LEN_LLM)
    conf              = torch.tensor(
        df[_CONF_COLS].values.astype(np.float32), dtype=torch.float32)
    item_nums         = torch.tensor(df.ItemNum.values, dtype=torch.long)

    dataset = TensorDataset(ids_q, mask_q, ids_llm, mask_llm,
                            conf, item_nums, labels_diff, labels_rtime)
    loader  = DataLoader(dataset, sampler=SequentialSampler(dataset),
                         batch_size=BATCH_SIZE)

    all_diff, all_rtime, all_items = [], [], []
    use_amp = device.type == "cuda"

    print(f"Running inference on {len(df)} items …")
    with torch.no_grad():
        for ids_q_b, mask_q_b, ids_llm_b, mask_llm_b, conf_b, items_b, *_ in loader:
            ids_q_b,   mask_q_b   = ids_q_b.to(device),   mask_q_b.to(device)
            ids_llm_b, mask_llm_b = ids_llm_b.to(device), mask_llm_b.to(device)
            conf_b = conf_b.to(device)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                pd_diff, pd_rtime = model(ids_q_b, mask_q_b,
                                          ids_llm_b, mask_llm_b, conf_b)
            all_diff.extend(pd_diff.cpu().numpy())
            all_rtime.extend(pd_rtime.cpu().numpy())
            all_items.extend(items_b.numpy())

    preds_diff  = sc_diff.inverse_transform(
        np.array(all_diff).reshape(-1, 1)).flatten()
    preds_rtime = sc_rtime.inverse_transform(
        np.array(all_rtime).reshape(-1, 1)).flatten()

    return np.array(all_items, dtype=int), preds_diff, preds_rtime


# ── Metric reporting ──────────────────────────────────────────────────────────

def report_metrics(true_diff, true_rtime, preds_diff, preds_rtime):
    def _show(true, pred, name):
        rmse     = np.sqrt(mean_squared_error(true, pred))
        r, rp     = pearsonr(true, pred)
        tau, taup   = kendalltau(true, pred)
        print(f"  [{name}]")
        print(f"    RMSE      : {rmse:.4f}")
        print(f"    Pearson r : {r:.4f} (p = {rp:.2e})")
        print(f"    Kendall τ : {tau:.4f} (p = {taup:.2e})")

    print("\n=== Test-Set Metrics ===")
    _show(true_diff,  preds_diff,  "Difficulty")
    _show(true_rtime, preds_rtime, "Response Time")


# ── Single-item display ───────────────────────────────────────────────────────

def show_single(item_num: int, test_df: pd.DataFrame,
                preds_diff: np.ndarray, preds_rtime: np.ndarray,
                item_nums: np.ndarray, has_labels: bool):
    mask = item_nums == item_num
    if not mask.any():
        print(f"ItemNum {item_num} not found in test set.")
        return

    idx      = np.where(mask)[0][0]
    row      = test_df[test_df["ItemNum"] == item_num].iloc[0]
    p_diff   = preds_diff[idx]
    p_rtime  = preds_rtime[idx]

    # Build answer list (skip empty/NaN entries)
    answer_letters = list("ABCDEFGHIJ")
    answers = []
    for letter in answer_letters:
        col = f"Answer__{letter}"
        val = str(row.get(col, "")).strip()
        if val and val.lower() not in ("nan", "none", ""):
            marker = " ◀ correct" if letter == str(row["Answer_Key"]).strip().upper() else ""
            answers.append(f"  ({letter}) {val}{marker}")

    divider = "─" * 72

    print(f"\n{divider}")
    print(f"  Item #{item_num}  |  Exam: {row.get('EXAM','?')}  |  Type: {row.get('ItemType','?')}")
    print(divider)
    print("\nQUESTION:")
    for line in textwrap.wrap(str(row["ItemStem_Text"]), width=70):
        print(f"  {line}")
    print("\nANSWER CHOICES:")
    print("\n".join(answers))

    print(f"\n{'─'*72}")
    print("LLM REASONING SIGNALS")
    print(f"{'─'*72}")

    gemma_correct = str(row.get("gemma_answer", "")).strip().upper() == \
                    str(row.get("Answer_Key",   "")).strip().upper()
    llama_correct = str(row.get("llama_answer", "")).strip().upper() == \
                    str(row.get("Answer_Key",   "")).strip().upper()

    print(f"  Gemma  → answered {str(row.get('gemma_answer','?')).strip().upper()}"
          f"  (confidence: {row.get('gemma_confidence', '?')})  "
          f"{'✓ correct' if gemma_correct else '✗ wrong'}")
    print(f"  Llama  → answered {str(row.get('llama_answer','?')).strip().upper()}"
          f"  (confidence: {row.get('llama_confidence', '?')})  "
          f"{'✓ correct' if llama_correct else '✗ wrong'}")

    agreement = "AGREE" if gemma_correct == llama_correct else "DISAGREE"
    print(f"  LLM agreement: {agreement}")

    print(f"\n{'─'*72}")
    print("MODEL PREDICTIONS")
    print(f"{'─'*72}")
    print(f"  Difficulty score  : {p_diff:.4f}  (0=easy … 1=hard)")
    print(f"  Response time     : {p_rtime:.1f} seconds")

    if has_labels and "Difficulty" in test_df.columns:
        t_diff  = float(row["Difficulty"])
        t_rtime = float(row["Response_Time"])
        print(f"\n  Ground truth difficulty   : {t_diff:.4f}")
        print(f"  Ground truth response time: {t_rtime:.1f} seconds")
        print(f"  Difficulty  error : {abs(p_diff  - t_diff):.4f}")
        print(f"  ResponseTime error: {abs(p_rtime - t_rtime):.1f} seconds")

    print(divider)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Demo for Dual-Gate difficulty predictor")
    parser.add_argument("--item",      type=int,  default=None,
                        help="Show detailed output for a specific ItemNum")
    parser.add_argument("--no-labels", action="store_true",
                        help="Skip metric computation (test CSV has no ground-truth)")
    parser.add_argument("--top",       type=int,  default=10,
                        help="Number of predictions to show in the summary table (default: 10)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}\n")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading datasets …")
    train_df = pd.read_csv(TRAIN_CSV)
    test_df  = pd.read_csv(TEST_CSV)
    train_df[STR_COLS] = train_df[STR_COLS].astype(str)

    has_labels = (not args.no_labels and
                  "Difficulty"    in test_df.columns and
                  "Response_Time" in test_df.columns)
    print(f"Train samples : {len(train_df):,}")
    print(f"Test  samples : {len(test_df):,}")
    print(f"Ground-truth labels available: {has_labels}\n")

    # ── Fit scalers on training set ───────────────────────────────────────────
    print("Fitting MinMaxScalers …")
    sc_diff, sc_rtime, sc_gemma, sc_llama = fit_scalers(train_df)

    # ── Pre-process test set ──────────────────────────────────────────────────
    test_df_proc, labels_diff, labels_rtime = preprocess(
        test_df, sc_diff, sc_rtime, sc_gemma, sc_llama, has_labels)

    # ── Build model & load weights ────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = build_model(device)

    # ── Inference ─────────────────────────────────────────────────────────────
    # Predictions: inverse_transform with train-only scaler (correct model scale)
    item_nums, preds_diff, preds_rtime = run_batch_inference(
        model, tokenizer, test_df_proc,
        labels_diff, labels_rtime,
        sc_diff, sc_rtime, device, has_labels)

    # ── Metrics ───────────────────────────────────────────────────────────────
    if has_labels:
        # Ground truth read directly from CSV — original units, no scaler round-trip
        true_diff  = test_df["Difficulty"].values.astype(np.float32)
        true_rtime = test_df["Response_Time"].values.astype(np.float32)
        report_metrics(true_diff, true_rtime, preds_diff, preds_rtime)

    # ── Summary table ─────────────────────────────────────────────────────────
    results = pd.DataFrame({
        "ItemNum":             item_nums,
        "Pred_Difficulty":     preds_diff.round(4),
        "Pred_ResponseTime_s": preds_rtime.round(1),
    })
    if has_labels:
        results["True_Difficulty"]     = true_diff.round(4)
        results["True_ResponseTime_s"] = true_rtime.round(1)

    print(f"\n=== Predictions (first {args.top} items) ===")
    print(results.head(args.top).to_string(index=False))

    # ── Single-item deep-dive ─────────────────────────────────────────────────
    if args.item is not None:
        show_single(args.item, test_df, preds_diff, preds_rtime, item_nums, has_labels)


if __name__ == "__main__":
    main()
