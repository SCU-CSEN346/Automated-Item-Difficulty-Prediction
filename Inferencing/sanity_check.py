"""Quick sanity check: run both models on test_questions.csv"""
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _lib import config
from _lib.batch import run

SANITY_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_questions.csv")
OUT_DIR    = os.path.dirname(os.path.abspath(__file__))

for model_name, model_cfg in config.MODELS.items():
    out_csv = os.path.join(OUT_DIR, f"sanity_{model_name}.csv")
    print(f"\n{'='*60}")
    print(f"Model : {model_name}  ({model_cfg['model_id']})")
    print(f"{'='*60}")
    run(SANITY_CSV, out_csv, model_cfg)
    df = pd.read_csv(out_csv)
    print("\nResults preview:")
    print(df[["ItemNum", "answer", "thinking"]].to_string(index=False))
