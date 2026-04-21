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

## Notes

- Notebooks perform manual fold splitting and regression-style training loops.
- Several notebooks normalize targets with `MinMaxScaler` before training.
- GPU is expected for practical training speed.

## Suggested Next Cleanup

If you plan to maintain this project long-term, useful next steps are:
1. Move shared preprocessing/training utilities into Python modules.
2. Add a single requirements file for local reproducibility.
3. Standardize notebook naming to match model/task/fold consistently.