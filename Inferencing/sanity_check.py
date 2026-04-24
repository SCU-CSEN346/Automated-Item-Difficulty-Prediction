"""Quick sanity check: run both models on test_questions.csv"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
import get_LLM_result as g

SANITY_CSV = os.path.join(os.path.dirname(__file__), "test_questions.csv")
OUT_DIR    = os.path.dirname(__file__)

for model_name, model_cfg in g.MODELS.items():
    out_csv = os.path.join(OUT_DIR, f"sanity_{model_name}.csv")
    print(f"\n{'='*60}")
    print(f"Model : {model_name}  ({model_cfg['model_id']})")
    print(f"{'='*60}")
    df = g.run_batch(SANITY_CSV, out_csv, model_cfg)
    print("\nResults preview:")
    print(df[["ItemNum", "answer", "thinking"]].to_string(index=False))
