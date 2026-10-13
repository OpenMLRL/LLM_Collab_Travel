"""Run the two-role Travel IAC baseline."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from single_turn.train.train_actor_critic import main

if __name__ == "__main__":
    main("iac")
