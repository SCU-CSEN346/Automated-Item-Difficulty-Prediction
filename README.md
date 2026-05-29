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
|- .gitattributes
|- .gitignore
|- Inferencing/
|  |- _lib/
|  |  |- __init__.py
|  |  |- auth.py
|  |  |- batch.py
|  |  |- config.py
|  |  `- inference.py
|  |- inference_test_questions.csv
|  |- run_inference.py
|  `- sanity_check.py
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
|     |- ModernBert_LoRA_Dualgate.ipynb
|     |- ModernBert_LoRA_Dualgate_(Final_Ensemble).ipynb
|     |- Unibuc-FMI_DualGate_predictions.csv
|     `- UPGRADES.md
|- Results/
|  |- diff_r0.208_tau0.134 __rtime_r0.535_tau0.434/
|  |- diff_r0.231_tau0.144__rtime_r0.535_tau0.434/
|  |- diff_r0.262_tau0.145__rtime_r0.628_tau0.490/
|  |- diff_r0.262_tau0.165__rtime_r0.536_tau0.415/
|  |- diff_r0.274_tau0.171__rtime_r0.597_tau0.460/
|  `- diff_r0.302_tau0.168__rtime_r0.610_tau0.471/
|     Each folder contains: metrics.txt + the notebook snapshot at that run.
|- README.md
|- requirements-jupyter.txt
`- requirements.txt
```


## Model Description Overview: ModernBERT and LoRA DualGate
Our system, DualGate, has significantly improved in performance compared to the original UnibucLLM submission to the BEA 2024 Shared Task. It uses a dual-stream encoder with parameter-efficient fine-tuning (LoRA) over ModernBERT and jointly predicts both Difficulty and Response_Time.

To deal with the limitations of the original work that stifled performance such as the small LLMs, lack of structured reasoning, and overfitting, we use stronger models such as Gemma 4 and Lamma 3.3 for chain-of-thought inference. We also refine how these signals are fused with the original question-and-answer text.

DualGate uses a shared ModernBERT encoder to process the inputs question text which has a 512 token max and LLM-generated reasoning logs which has a 1024 token max. We use mean pooling to create two 768-dimensional embeddings and combine them with scalar features such as confidence and answer correctness.

Since the 466 sample data set is small and could overfit, we use LoRA (r=32) on all linear layers which reduces training to 4.34% of the parameters. Lastly, we train the model jointly using weighted Huber losses for both prediction targets.

Finally, to combine the strengths of these inputs to hopefully improve performance, we used a weighted ensemble. We set the ensemble weight alpha to 0.85 to give more weight to the structured LLM reasoning stream while still retaining the direct question/answer signal.

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

- `Model/Our/ModernBert_LoRA_Dualgate.ipynb`: ModernBERT + LoRA dual-gate architecture for joint difficulty and response-time prediction. See [Our Model](#our-model-modernbert--lora-dualgate) below for details.

## Environment and Dependencies

> **Python version:** Use **Python 3.10**. The notebooks use `DataFrame.append()` which was removed in pandas 2.0, and pandas 1.x wheels are not available for Python 3.11+. Python 3.10 is required to run the notebooks without refactoring those cells.

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

As an extension to the original notebook experiments we added an LLM-based inference pipeline that generates a **chain-of-thought answer and confidence score** for every item using large language models served through Google Vertex AI MaaS. The LLM output is then used as an additional feature set when predicting difficulty and response time.

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

The script is fully resumable.  If a result CSV already exists it checks for missing or corrupted rows and re-infers only those, looping until every item is clean.  Progress is checkpointed after every successful item.

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

- **Dual-encoder:** `q_answers_input` (question + choices, max 512 tokens) and `llms_a_input` (Gemma + Llama CoT reasoning, max 1024 tokens) are encoded in separate forward passes through the same LoRA-adapted ModernBERT backbone.
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

# Team Member Contributions

## Michael Suo:
1. Paper: Wrote introduction section on Overleaf
2. Paper: Rewrite Results section after final model
3. Paper: Wrote Analysis and Motivation subsection in Paper Evaluation
4. Paper: Add table 2 for demostrate our Dataset composition
5. Paper: Add reprodcution results as table 3
6. Paper: Finalize table 4 with our final resutls
7. Paper: Draft first version of related work
8. Repo: Create and maintain repo file structure
9. Code: Craete inference pipline
10. Code: Write inference sanity-check and prompt template
11. Code: Inference from Gemma 4 and Llama 3.3 for CoT stream
12. Data: Fix original author data encoding and formatting issue
13. Data: Generate Final data for our model
14. ENV: Build Ubuntu env on computer for team's development
15. ENV: Help build up env for original paper reproduction
16. Code: Upgrade our model with mean pooling methods
17. Code: Replace MSE loss function with Huber loss
18. Code: Add LLM correctness label as scalar inputs
19. Presentation: Final presentation slides and script (11-13)
20. Presentation: Hold for Project Status presentation
21. Presentation: Help writing Project Idea presentation script

## Bojing (Shirley) Yu:
1. Paper: Wrote Abstract section
2. Paper: Editted Related work section
3. Paper: Wrote Methods section (opening paragraph + 4.1-4.4)
4. Paper: Draft first version of Results section (routine submission)
5. Paper: Draw model architecture diagram
6. Paper: Wrote Baseline Reproduction and Verification subsection in Paper Evaluation section
7. Paper: Wrote Ethics section
8. Paper: Wrote Conlusion section
9. Code: Wrote first version of Dulgate
10. Code: Baseline reproduction
11. Code: Created a potential upgrade list
12. Code: Upgrate 1 (Dual encoder)
13. Code: Upgrate 2 (Asymmetric MAX_LEN)
14. Code: Upgrate 3 (LLM confidence score as scalar features)
15. Idea Presentation: Did slides and wrote the script
16. Project Status Presentation: Did slides and wrote the script
17. Final Presentation: Did majority of the slides from scratch and wrote script for the team (1-10)
18. Poster Presentation: Created poster for poster presentation

## Andrew Le:
1. Paper: Started introduction and related work.
2. Paper: Rewrote parts of experimental setup and introduction.
3. Paper: Wrote parts of Parameter-Efficient Fine-Tuning and Multi-Task Learning in Related Work.
4. Paper: Evaluation Metrics
5. Paper: Feature set table
6. Paper: Weighted Ensemble section
7. Paper: Weighted ensemble results
8. Code: Baseline reproduction (outdated)
9. Code: Upgrade 7: ensemble method
10. Presentation: Edited final presentation slides
11. Presentation: Edited idea presentation slides
12. Other: Created group discord
