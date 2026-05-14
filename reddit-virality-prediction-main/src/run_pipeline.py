"""Run the complete Reddit virality project pipeline."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_step(command: list[str]) -> None:
    print("\n" + "=" * 80)
    print("Running:", " ".join(command))
    print("=" * 80)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    python = sys.executable
    run_step([python, "src/collect_reddit_public_api.py", "--limit", "100"])
    run_step([python, "src/preprocess.py"])
    run_step([python, "src/doc2vec_tsne.py"])
    run_step([python, "src/train_classifier.py"])
    print("\nPipeline complete. Check data/processed and outputs/ for results.")


if __name__ == "__main__":
    main()
