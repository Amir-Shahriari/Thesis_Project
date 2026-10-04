"""Suite B — Transfer benchmark script.

Usage:
    uv run python scripts/run_transfer_benchmark.py --seed 42
"""
import argparse
import json
import pathlib
import time

from qwarm.eval.transfer_benchmark import run_transfer_benchmark
from qwarm.utils.seeding import set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_global_seed(args.seed)
    out_dir = pathlib.Path("runs") / f"transfer_{int(time.time())}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = run_transfer_benchmark(
        train_grid_config={"grid_width": 50, "grid_height": 50, "extra_edges": 2, "seed": args.seed},
        test_grid_configs=[
            {"grid_width": 50, "grid_height": 50, "extra_edges": 2, "seed": 99},
            {"grid_width": 75, "grid_height": 75, "extra_edges": 3, "seed": 99},
            {"grid_width": 60, "grid_height": 40, "extra_edges": 2, "seed": 99},
        ],
        n_train_iterations=10,
        episodes_per_iteration=200,
        seed=args.seed,
    )

    out_path = out_dir / "transfer_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Transfer results saved to {out_path}")
    for agent_name, grid_results in results.items():
        for grid_key, metrics in grid_results.items():
            print(f"  {agent_name} on {grid_key}: goal_reach={metrics['goal_reach_rate']:.2f}")


if __name__ == "__main__":
    main()
