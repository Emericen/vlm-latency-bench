#!/usr/bin/env python3
"""
Script to run benchmarks across multiple models automatically.
Usage: python run_benchmarks.py
"""

import subprocess
import sys
from pathlib import Path

# Models to benchmark
MODELS = [
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen/Qwen2.5-VL-32B-Instruct",
    "Qwen/Qwen2.5-VL-32B-Instruct-AWQ",
]

# Configuration
BASE_URL = "http://192.222.54.163:443/v1"
RUNS = 8
TURNS = 8
OUTPUT_FILE = "all_models_benchmark.csv"


def run_benchmark(model, is_first=False):
    """Run benchmark for a single model."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {model}")
    print(f"{'='*60}")

    if not is_first:
        print(f"\n📋 MANUAL STEP REQUIRED:")
        print(f"   1. SSH to GPU server")
        print(f"   2. cd inference_experiment/")
        print(f'   3. Run: make stop && make run MODEL="{model}"')
        print(f"   4. Wait for server to start (check: make logs)")
        print(f"   5. Verify server: make health")

        input(f"\n⏸️  Press Enter when {model} server is ready...")

    # fmt: off
    cmd = [
        sys.executable, "scripts/s1_benchmark_models.py",
        "--model", model,
        "--base-url", BASE_URL,
        "--runs", str(RUNS),
        "--turns", str(TURNS),
        "--output", OUTPUT_FILE
    ]
    # fmt: on

    try:
        subprocess.run(cmd, check=True)
        print(f"✓ Completed: {model}")
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed: {model} (exit code: {e.returncode})")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠ Interrupted during: {model}")
        return False

    return True


def main():
    """Run benchmarks for all models."""
    print(f"Starting benchmark suite for {len(MODELS)} models")
    print(f"Config: {RUNS} runs × {TURNS} turns per model")
    print(f"Output: {OUTPUT_FILE}")

    # Clear previous results
    output_path = Path(OUTPUT_FILE)
    if output_path.exists():
        output_path.unlink()
        print(f"Cleared previous results from {OUTPUT_FILE}")

    completed = 0
    failed = []

    for i, model in enumerate(MODELS, 1):
        print(f"\nProgress: {i}/{len(MODELS)}")

        if run_benchmark(model, is_first=(i == 1)):
            completed += 1
        else:
            failed.append(model)

    # Summary
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"Completed: {completed}/{len(MODELS)} models")

    if failed:
        print(f"Failed: {', '.join(failed)}")

    if completed > 0:
        print(f"Results saved to: {OUTPUT_FILE}")
        print("\nTo analyze results:")
        print(
            f"  python -c \"import pandas as pd; df = pd.read_csv('{OUTPUT_FILE}'); print(df.groupby('model')[['ttft', 'total_time']].agg(['mean', 'std']).round(3))\""
        )


if __name__ == "__main__":
    main()
