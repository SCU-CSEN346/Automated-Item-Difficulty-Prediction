# ModernBERT + LoRA — Planned Upgrades

Baseline: single-encoder, single-stream (`all_input`), CLS pooling, MSE loss.  
Original paper best: BERT + LLM-augmented features → Pearson r = 0.181, Kendall τ = 0.106.

---

## Upgrade Roadmap

### 🔴 High Impact

- [x] **1. Dual-encoder (separate streams)**  
  Encode `q_answers_input` (question + choices) and `llms_a_input` (LLM reasoning) with the *same* shared LoRA-adapted ModernBERT in two separate forward passes. Concatenate the two CLS vectors → `Linear(2H, 1)` regression heads.  
  _Rationale: prevents LLM chain-of-thought from drowning out the question signal in a single 512-token window. The original paper shows BERT + LLM features beats question-only, but only when the model can properly attend to each._

- [ ] **2. Asymmetric MAX_LEN (LLM stream at 1024)**  
  `q_answers_input` stays at 512; `llms_a_input` (gemma_thinking + llama_thinking + Answer_Text) gets 1024. ModernBERT natively supports up to 8192 tokens via RoPE — no architecture change needed.  
  _Rationale: LLM chain-of-thought is often 300–600 tokens on its own; truncating at 512 discards the conclusion of the reasoning._

- [ ] **3. LLM confidence scores as scalar features**  
  Append `gemma_confidence` and `llama_confidence` (already in the dataset) as a 2-dim numeric vector. Concatenate to the fused CLS representation before the regression heads: `Linear(2H + 2, 1)`.  
  _Rationale: agreement + high confidence → easy item; disagreement or low confidence → hard item. Strong proxy for difficulty that requires no additional compute._

---

### 🟡 Medium Impact

- [ ] **4. Mean pooling instead of CLS-only**  
  Replace `out.last_hidden_state[:, 0, :]` with an attention-mask-weighted mean over all token positions.  
  _Rationale: CLS pooling can miss information spread across long sequences; mean pooling is consistently better for regression on longer inputs._

- [ ] **5. Huber loss instead of MSE**  
  Replace `nn.MSELoss()` with `nn.HuberLoss(delta=0.1)`.  
  _Rationale: difficulty and response time distributions have outliers; Huber loss down-weights their gradient contribution and typically improves rank correlation (Kendall τ) even when RMSE is similar._

- [ ] **6. LLM answer correctness as a binary feature**  
  Derive `gemma_correct` and `llama_correct` (1 if the LLM's answer matches `Answer_Key`, else 0). Append alongside confidence scores.  
  _Rationale: direct proxy — if both LLMs answer correctly the item is likely easy; if both fail it is hard._

---

### 🟢 Lower / Exploratory

- [ ] **7. Ensemble (question-only + LLM-augmented)**  
  Train two separate models (one on `q_answers_input`, one on `llms_a_input`) and average their predictions.  
  _Rationale: hedges against one stream dominating; often gives +0.02–0.05 on Kendall τ in regression ensembles._

- [ ] **8. Additional LLMs**  
  The original paper used 3 LLMs (Falcon, Meditron, Mistral); our dataset has 2 (Gemma, Llama). Adding a third (e.g., Qwen or Phi) to the inference pipeline would enrich the `llms_a_input` signal.  
  _Requires re-running inference — out of scope until core upgrades are validated._

---

## Progress Log

| # | Upgrade | Status | Notes |
|---|---|---|---|
| 1 | Dual-encoder | ✅ done | Architecture + all training/test cells |
| 2 | Asymmetric MAX_LEN | ✅ done | LLM stream at 1024 tokens |
| 3 | Confidence scalars | ✅ done | gemma+llama conf → [B,2], concat to CLS |
| 4 | Mean pooling | ⏳ | After #1 |
| 5 | Huber loss | ⏳ | Quick swap, any time |
| 6 | Correctness flag | ⏳ | After #3 |
| 7 | Ensemble | ⏳ | After #1–4 |
| 8 | More LLMs | ⏳ | Future work |
