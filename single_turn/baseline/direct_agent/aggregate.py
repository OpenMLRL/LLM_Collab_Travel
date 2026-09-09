"""Aggregate two FINISHED runs per method; refuse missing, mixed-protocol or duplicate seeds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from .protocol import ALGORITHMS, METRICS, eval_key, fingerprint


def aggregate(paths: list[Path], expected_seeds=(42, 43)) -> dict:
    grouped = {a: {} for a in ALGORITHMS}
    protocol = set()
    training_protocol = {a: set() for a in ALGORITHMS}
    for folder in paths:
        completed = json.loads((folder / "completed.json").read_text())
        result = json.loads((folder / "eval_final.json").read_text())
        metadata = json.loads((folder / "protocol.json").read_text())
        if completed["status"] != "finished" or result["phase"] != "final":
            raise ValueError(f"Incomplete run: {folder}")
        if result["count"] != metadata["settings"]["eval_samples"]:
            raise ValueError(f"Partial evaluation: {folder}")
        alg, seed = result["algorithm"], result["seed"]
        if completed["algorithm"] != alg or completed["seed"] != seed:
            raise ValueError(f"Mismatched run records: {folder}")
        if alg not in grouped or seed not in expected_seeds or seed in grouped[alg]:
            raise ValueError(f"Unexpected algorithm/seed or duplicate run: {folder}")
        protocol.add(result["protocol_sha256"])
        training_protocol[alg].add(fingerprint({k: v for k, v in metadata["settings"].items()
            if k not in ("seed", "output_dir", "actor_device", "reward_device", "wandb_enabled")}))
        grouped[alg][seed] = result["metrics"]
    if len(protocol) != 1:
        raise ValueError("Runs use different models/data/evaluation protocols")
    for alg, runs in grouped.items():
        if set(runs) != set(expected_seeds):
            raise ValueError(f"{alg}: need seeds {expected_seeds}, found {sorted(runs)}")
        if len(training_protocol[alg]) != 1:
            raise ValueError(f"{alg}: the two seeds used different training configurations")
    summary = {}
    for alg, runs in grouped.items():
        summary[alg] = {}
        for metric in METRICS:
            key = eval_key(metric)
            values = [runs[s][key] for s in expected_seeds]
            deltas = [runs[s][key] - grouped["raw"][s][key] for s in expected_seeds]
            summary[alg][metric] = {"mean": statistics.mean(values), "sample_std": statistics.stdev(values),
                                   "paired_delta_from_raw": statistics.mean(deltas), "per_seed": values}
    return {"seeds": list(expected_seeds), "runs_per_method": len(expected_seeds),
            "protocol_sha256": next(iter(protocol)), "results": summary,
            "caveat": "Two seeds are preliminary evidence; SD is not a confidence interval."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(args.runs)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("method\tfinal_pass_mean\treward_mean\tpaired_final_pass_gain")
    for alg, row in result["results"].items():
        print(f"{alg}\t{row['final_pass']['mean']:.4f}\t{row['reward']['mean']:.4f}"
              f"\t{row['final_pass']['paired_delta_from_raw']:.4f}")


if __name__ == "__main__":
    main()
