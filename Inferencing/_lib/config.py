"""
_lib/config.py
--------------
Central configuration for paths, model definitions, and constants.
All other modules import from here — do not scatter magic strings elsewhere.
"""

import os

# ── Project / API ──────────────────────────────────────────────────────────────
PROJECT_ID = "project-8c9c1880-369b-4376-b02"

# Models to run.  Add or remove entries here to change which models are run.
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

# ── Inference settings ────────────────────────────────────────────────────────
# 2048 tokens to give complex CoT questions enough room to finish
MAX_OUTPUT_TOKENS = 2048

# Seconds between API requests (avoids quota errors)
REQUEST_DELAY = 1.0

# ── Answer schema ─────────────────────────────────────────────────────────────
ANSWER_LABELS = list("ABCDEFGHIJ")
ANSWER_COLS   = [f"Answer__{l}" for l in ANSWER_LABELS]
VALID_LETTERS = set(ANSWER_LABELS)

# ── Paths (resolved from this file's location) ────────────────────────────────
_LIB_DIR  = os.path.dirname(os.path.abspath(__file__))   # Inferencing/_lib
_INF_DIR  = os.path.dirname(_LIB_DIR)                    # Inferencing
_ROOT_DIR = os.path.dirname(_INF_DIR)                    # workspace root

DATA_DIR  = os.path.join(_ROOT_DIR, "Data", "Our")
TRAIN_CSV = os.path.join(DATA_DIR, "train_final.csv")
TEST_CSV  = os.path.join(DATA_DIR, "test_final.csv")
