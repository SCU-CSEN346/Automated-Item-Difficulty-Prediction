# Automated-Item-Difficulty-Prediction

We are doing Option 2 for the project.
This repository contains notebook-based experiments for predicting medical exam item characteristics from question text and answer options.

Primary prediction targets:
- `Difficulty` (item difficulty)
- `Response_Time` (time needed to answer)

The experiments use transformer-based models (BERT and GPT variants) on the BEA 2024 extended task/test data.

## Repository Layout

```
.
|- Data/
|  |- Original Paper/
|  |  |- BEA 2024 Task Data Extended Shuffled - BEA 2024 Task Data Extended Shuffled.csv
|  |  `- BEA 2024 Test Data Extended - Sheet 1.csv
|  `- Our/
|     |- train_final.csv / test_final.csv        ← pre-processed splits
|     |- train_inference_gemma.csv / test_inference_gemma.csv
|     `- train_inference_llama.csv / test_inference_llama.csv
|- Model/
|  |- Original Paper/
|  |  |- Difficulty/
|  |  |  |- 6_Difficulty_BEA_2024_BERT.ipynb
|  |  |  |- 8_Difficulty_BEA_2024_GPT.ipynb
|  |  |  `- 12_Difficulty_BEA_2024_GPT.ipynb
|  |  `- Response Time/
|  |     |- 2_Time_BEA_2024_BERT.ipynb
|  |     |- 6_Time_BEA_2024_BERT.ipynb
|  |     `- 8_Time_BEA_2024_GPT.ipynb
|  `- Our/
|     |- ModernBert_LoRA_Dualgate.ipynb          ← our model (training + eval)
|     |- Unibuc-FMI_DualGate_predictions.csv     ← competition submission format
|     `- UPGRADES.md                             ← upgrade roadmap + progress log
|- Inferencing/
|  |- run_inference.py
|  |- sanity_check.py
|  `- _lib/  (auth, batch, config, inference)
`- Results/
   |- diff_r0.231_tau0.144__rtime_r0.535_tau0.434/
   |- diff_r0.262_tau0.145__rtime_r0.628_tau0.490/
   |- diff_r0.262_tau0.165__rtime_r0.536_tau0.415/
   |- diff_r0.274_tau0.171__rtime_r0.597_tau0.460/
   `- diff_r0.302_tau0.168__rtime_r0.610_tau0.471/
      Each folder contains: metrics.txt + the notebook snapshot at that run.
```


## Model Description Overview: ModernBERT and LoRA DualGate
Our system, DualGate, has significantly improved in performance compared to the original UnibucLLM submission to the BEA 2024 Shared Task. It uses a dual-stream encoder with parameter-efficient fine-tuning, scalar feature injection, and joint multi-task learning framework. 

To deal with the limitations of the original work that stifled performance such as the small LLMs, lack of structured reasoning, and overfitting, we use stronger models such as Gemma 4 and Lamma 3.3 for zero-shot Chain-of-Thought feature generation.

DualGate uses a shared ModernBERT encoder to process the inputs question text which has a 512 token max and LLM-generated reasoning logs which has a 1024 token max. We use mean pooling to create two 768-dimensional vectors(the standard embedding dimension of the ModernBERT-base encoder). These combine with four metadata features (the normalized confidence scores and correctness labels for each) into a single 1540-dimensional vector. This vector feeds into two regression heads with sigmoid activations. 

Since the 466 sample data set is small and could overfit, we use LoRA (r=32) on all linear layers which reduces training to 4.34% of the parameters. Lastly, we train the model jointly using weighted Huber loss delta=0.1 to take advantage of the link between difficulty and response time.

Finally, to combine the strengths of these inputs to hopefully improve performance, we used a weighted ensemble. We set the ensemble weight alpha to 0.85 to give more weight to the structured LLM reasoning in the final prediction since it is more accurate in predictions.

## Data

Main dataset columns used across notebooks include:
- Item text and answer options: `ItemStem_Text`, `Answer__A` ... `Answer__J`
- Metadata: `ItemType`, `EXAM`, `step_integer`
- Model inputs: `all_input`, `q_answers_input`, `answers_input`, `q_a_input`, `llms_a_input`, `q_llms_a_input`
- Targets: `Difficulty`, `Response_Time`

## Notebook Guide

### Original Paper Notebooks

Difficulty notebooks (`Model/Original Paper/Difficulty/`):
- `6_Difficulty_BEA_2024_BERT.ipynb`: BERT-based difficulty regression pipeline
- `8_Difficulty_BEA_2024_GPT.ipynb`: GPT-based difficulty regression pipeline
- `12_Difficulty_BEA_2024_GPT.ipynb`: GPT-based difficulty experiment variant

Response-time notebooks (`Model/Original Paper/Response Time/`):
- `2_Time_BEA_2024_BERT.ipynb`: BERT-based response-time regression pipeline
- `6_Time_BEA_2024_BERT.ipynb`: BERT-based response-time experiment variant
- `8_Time_BEA_2024_GPT.ipynb`: GPT-based response-time regression pipeline

### Our Model Notebook

- `Model/Our/ModernBert_LoRA_Dualgate.ipynb`: ModernBERT + LoRA dual-gate architecture for joint difficulty and response-time prediction. See [Our Model](#our-model-modernbert--lora-dualgate) below for details and results.

## Environment and Dependencies

> **Python version:** Use **Python 3.10**. The notebooks use `DataFrame.append()` which was removed in pandas 2.0, and pandas 1.x wheels are not available for Python 3.11+. Python 3.10 is required to install `pandas<2.0` locally.

The notebooks were authored for Google Colab and include inline package installs.

Common dependencies:
- `transformers`
- `torch`
- `torchmetrics`
- `tensorflow` (used for GPU checks in notebook setup)
- `scikit-learn`
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `gspread`, `google-auth`
- `fasttext`, `huggingface_hub`

## How to Run

### Option 1: Run as-is in Google Colab (recommended)

1. Open any notebook.
2. Run setup/import cells (including `pip install` cells).
3. Authenticate when prompted by:
	- `from google.colab import auth`
	- `auth.authenticate_user()`
4. Ensure your Google account can access sheets named:
	- `BEA 2024 Task Data Extended Shuffled`
	- `BEA 2024 Test Data Extended`
5. Run the remaining cells in order.

### Option 2: Run with local CSV files

The current notebooks load data from Google Sheets with `gspread`. To use local files in `Data/`, replace the sheet-loading cells with `pandas.read_csv`.

Example replacement:

```python
import pandas as pd

dataset = pd.read_csv(
	 "Data/BEA 2024 Task Data Extended Shuffled - BEA 2024 Task Data Extended Shuffled.csv"
)
test_dataset = pd.read_csv(
	 "Data/BEA 2024 Test Data Extended - Sheet 1.csv"
)
```

## LLM Chain-of-Thought Inference (Our Addition)

As an extension to the original notebook experiments we added an LLM-based inference pipeline that generates a **chain-of-thought answer and confidence score** for every item using large language models served through Google Vertex AI MaaS.  The LLM output is then used as an additional feature set when predicting difficulty and response time.

### Models

| Key | Model | Endpoint |
|-----|-------|----------|
| `gemma` | `google/gemma-4-26b-a4b-it-maas` | `aiplatform.googleapis.com` (global) |
| `llama` | `meta/llama-3.3-70b-instruct-maas` | `us-central1-aiplatform.googleapis.com` |

Each model is prompted to act as a third-year medical student and respond with:
- `<thinking>` — free-text chain-of-thought reasoning
- `<answer>` — a single letter (A–J)
- `<confidence>` — integer 1–5

### Inference Output

The pipeline writes one CSV per model per split:

```
Data/Our/
  train_final.csv              ← pre-processed training items
  test_final.csv               ← pre-processed test items
  train_inference_gemma.csv    ← Gemma CoT answers for train set
  test_inference_gemma.csv     ← Gemma CoT answers for test set
  train_inference_llama.csv    ← Llama CoT answers for train set
  test_inference_llama.csv     ← Llama CoT answers for test set
```

Columns in each inference CSV: `ItemNum`, `thinking`, `answer`, `confidence`.

### Running Inference

```bash
# from the workspace root (venv must be active)
python run_inference.py
```

The script is fully resumable.  If a result CSV already exists it checks for missing or corrupted rows and re-infers only those, looping until every item is clean.  Progress is checkpointed after every single item.

A row is considered **corrupted** if:
- `answer` is not a valid letter A–J
- `confidence` is empty / missing
- `thinking` contains an embedded newline (indicates the model was cut off before finishing)

### Configuration

All tunable settings live in `Inferencing/_lib/config.py`:

| Setting | Default | Purpose |
|---------|---------|---------|
| `MAX_OUTPUT_TOKENS` | `2048` | Token budget per request |
| `REQUEST_DELAY` | `1.0 s` | Delay between API calls |
| `MODELS` | see above | Which models to run |

### Inferencing Module Layout

```
Inferencing/
  run_inference.py       ← entry point (also at repo root)
  _lib/
    config.py            ← paths, model defs, constants
    auth.py              ← Google ADC credentials + OpenAI-compatible client
    inference.py         ← prompt template, item formatter, single-item inference
    batch.py             ← resume / repair loop with per-item checkpointing
  sanity_check.py        ← quick smoke-test on a small sample
  test_questions.csv     ← sample items used by sanity_check.py
```

### Prerequisites

```bash
pip install openai google-auth pandas
gcloud auth application-default login   # sets up Google ADC credentials
```

> **Before running:** open `Inferencing/_lib/config.py` and replace the `PROJECT_ID` value with your own Google Cloud project ID:
>
> ```python
> PROJECT_ID = "your-gcp-project-id-here"
> ```
>
> Make sure the Google account you authenticate with (`gcloud auth application-default login`) has the **Vertex AI User** role on that project, and that the Vertex AI API is enabled.

## Our Model: ModernBERT + LoRA DualGate

Our model (`Model/Our/ModernBert_LoRA_Dualgate.ipynb`) improves on the original paper's best result (Pearson r = 0.181, Kendall τ = 0.106 for difficulty) using the following architecture:

- **Dual-encoder:** `q_answers_input` (question + choices, max 512 tokens) and `llms_a_input` (Gemma + Llama CoT reasoning, max 1024 tokens) are encoded in separate forward passes through the same LoRA-adapted ModernBERT. The two pooled representations are concatenated.
- **Mean pooling:** Attention-mask-weighted mean over all token positions (instead of CLS-only).
- **Scalar features:** `gemma_confidence`, `llama_confidence`, `gemma_correct`, `llama_correct` appended to the fused representation before the regression heads.
- **Huber loss** (`delta=0.1`) for outlier-robust training.
- **Joint prediction:** separate linear heads for `Difficulty` and `Response_Time` trained simultaneously.

See `Model/Our/UPGRADES.md` for the full upgrade roadmap and progress log.

### Results

All runs evaluated on the held-out test set. Folder names encode the key metrics of that run.

| Run | Diff RMSE | Diff r | Diff τ | RT RMSE | RT r | RT τ |
|-----|:---------:|:------:|:------:|:-------:|:----:|:----:|
| diff_r0.231_tau0.144__rtime_r0.535_tau0.434 | 0.3147 | 0.2308 | 0.1441 | 29.485 | 0.5350 | 0.4340 |
| diff_r0.262_tau0.145__rtime_r0.628_tau0.490 | 0.3064 | 0.2620 | 0.1449 | 24.591 | 0.6282 | **0.4899** |
| diff_r0.262_tau0.165__rtime_r0.536_tau0.415 | 0.3052 | 0.2617 | 0.1649 | 26.849 | 0.5364 | 0.4155 |
| diff_r0.274_tau0.171__rtime_r0.597_tau0.460 | 0.3079 | 0.2742 | **0.1708** | 25.411 | 0.5971 | 0.4599 |
| diff_r0.302_tau0.168__rtime_r0.610_tau0.471 | **0.3024** | **0.3025** | 0.1678 | **25.102** | 0.6104 | 0.4714 |
| *Original paper best* | *0.3078* | *0.181* | *0.106* | *27.0160* | *0.5503* | *0.4355* |

Best single-run result: **Difficulty r = 0.3025**, **RT r = 0.6282** — representing a **+67%** improvement in difficulty Pearson r over the original paper baseline.

## Notes

- Notebooks perform manual fold splitting and regression-style training loops.
- Several notebooks normalize targets with `MinMaxScaler` before training.
- GPU is expected for practical training speed.

## Suggested Next Steps

1. Move shared preprocessing/training utilities into Python modules.
2. Standardize notebook naming to match model/task/fold consistently.
3. Implement the remaining upgrades from `Model/Our/UPGRADES.md` (ensemble of question-only + LLM-augmented streams, additional LLMs).

### Team Member Contributions
##Michael Suo:

##Bojing (Shirley) Yu:

##Andrew Le:


