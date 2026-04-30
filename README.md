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
|  |- BEA 2024 Task Data Extended Shuffled - BEA 2024 Task Data Extended Shuffled.csv
|  `- BEA 2024 Test Data Extended - Sheet 1.csv
|- Difficulty/
|  |- 6_Difficulty_BEA_2024_BERT.ipynb
|  |- 8_Difficulty_BEA_2024_GPT.ipynb
|  `- 12_Difficulty_BEA_2024_GPT.ipynb
`- Response Time/
	|- 2_Time_BEA_2024_BERT.ipynb
	|- 6_Time_BEA_2024_BERT.ipynb
	`- 8_Time_BEA_2024_GPT.ipynb
```

## Data

Main dataset columns used across notebooks include:
- Item text and answer options: `ItemStem_Text`, `Answer__A` ... `Answer__J`
- Metadata: `ItemType`, `EXAM`, `step_integer`
- Model inputs: `all_input`, `q_answers_input`, `answers_input`, `q_a_input`, `llms_a_input`, `q_llms_a_input`
- Targets: `Difficulty`, `Response_Time`

## Notebook Guide

Difficulty notebooks:
- `Difficulty/6_Difficulty_BEA_2024_BERT.ipynb`: BERT-based difficulty regression pipeline
- `Difficulty/8_Difficulty_BEA_2024_GPT.ipynb`: GPT-based difficulty regression pipeline
- `Difficulty/12_Difficulty_BEA_2024_GPT.ipynb`: GPT-based difficulty experiment variant

Response-time notebooks:
- `Response Time/2_Time_BEA_2024_BERT.ipynb`: BERT-based response-time regression pipeline
- `Response Time/6_Time_BEA_2024_BERT.ipynb`: BERT-based response-time experiment variant
- `Response Time/8_Time_BEA_2024_GPT.ipynb`: GPT-based response-time regression pipeline

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

## Notes

- Notebooks perform manual fold splitting and regression-style training loops.
- Several notebooks normalize targets with `MinMaxScaler` before training.
- GPU is expected for practical training speed.

## Suggested Next Cleanup

If you plan to maintain this project long-term, useful next steps are:
1. Move shared preprocessing/training utilities into Python modules.
2. Add a single requirements file for local reproducibility.
3. Standardize notebook naming to match model/task/fold consistently.