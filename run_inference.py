"""
run_inference.py
----------------
Single entry point for LLM inference.

Usage
-----
    python run_inference.py

Behaviour
---------
For each configured model and each dataset (train / test):

  • If the result CSV does not exist
      → runs full inference from scratch.

  • If the result CSV already exists
      → checks for missing items  (rows in source but absent from output)
         and corrupted items       (bad answer, missing confidence, or
                                    truncated thinking with embedded newlines)
      → re-infers all problem items and repeats until everything is clean.

Progress is checkpointed after every single item so a crash loses at most
one item of work.  Re-running the script safely picks up where it left off.

Configuration
-------------
Edit  Inferencing/_lib/config.py  to change:
  • Which models to run  (MODELS dict)
  • Token budget          (MAX_OUTPUT_TOKENS)
  • Request delay         (REQUEST_DELAY)
  • Source / output paths (DATA_DIR, TRAIN_CSV, TEST_CSV)
"""

import os
import sys

# Make the _lib package importable regardless of where the script is run from
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "Inferencing"))

from _lib import config          # noqa: E402
from _lib.batch import run       # noqa: E402


def main() -> None:
    for model_name, model_cfg in config.MODELS.items():
        train_out = os.path.join(config.DATA_DIR, f"train_inference_{model_name}.csv")
        test_out  = os.path.join(config.DATA_DIR, f"test_inference_{model_name}.csv")

        print(f"\n{'='*60}")
        print(f"Model : {model_name}  ({model_cfg['model_id']})")
        print(f"{'='*60}")

        print("\n--- TRAIN ---")
        run(config.TRAIN_CSV, train_out, model_cfg)

        print("\n--- TEST ---")
        run(config.TEST_CSV, test_out, model_cfg)


if __name__ == "__main__":
    main()
