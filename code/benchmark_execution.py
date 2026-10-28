"""Measure execution settings without reducing a final experiment's workload.

Example screening run (one split per dataset, not final report evidence):
    python benchmark_execution.py --configs 2x2 3x2 --runs 1 --out runtime_optimization_screen

Each configuration is a fresh subprocess and output directory. The elapsed time
includes Python startup, fits, evaluation and plots; verification is timed separately.
Use --runs 10 to measure a complete assignment experiment for a selected setting.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


def compare_outputs(result, reference):
    """Compare saved arrays, including splits and normalization, on shared seeds."""
    details = []

    def compare_array(label, actual, expected):
        same = actual.dtype == expected.dtype and np.array_equal(actual, expected)
        row = {"file": label, "exact_match": same}
        if actual.shape == expected.shape:
            row["different_entries"] = int(np.count_nonzero(actual != expected))
            row["max_absolute_difference"] = float(np.max(np.abs(actual - expected)))
        details.append(row)

    for folder in sorted(result.glob("*/seed_*")):
        other = reference / folder.relative_to(result)
        for path in sorted(folder.glob("*.npy")):
            expected = other / path.name
            if not expected.is_file():
                details.append({"file": str(path.relative_to(result)), "status": "missing_reference"})
                continue
            compare_array(str(path.relative_to(result)), np.load(path), np.load(expected))
        for path in sorted(folder.glob("*.npz")):
            expected = other / path.name
            if not expected.is_file():
                details.append({"file": str(path.relative_to(result)), "status": "missing_reference"})
                continue
            with np.load(path) as actual, np.load(expected) as target:
                if set(actual.files) != set(target.files):
                    details.append({"file": str(path.relative_to(result)), "status": "different_array_keys"})
                    continue
                for key in sorted(actual.files):
                    compare_array(f"{path.relative_to(result)}:{key}", actual[key], target[key])
    return {"compared_arrays": len(details),
            "all_exact": bool(details) and all(row.get("exact_match", False) for row in details),
            "details": details}


def run_configuration(code, output, workers, threads, runs, reference=None, datasets=None):
    output.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-u", "run_evaluation.py", "--runs", str(runs),
               "--workers", str(workers), "--threads", str(threads), "--out", str(output.resolve())]
    if datasets is not None:
        command.extend(["--datasets", *datasets])
    record = {"workers": workers, "threads_per_worker": threads, "runs_per_dataset": runs,
              "datasets": datasets or ["FashionMNIST0.3", "FashionMNIST0.6", "CIFAR"],
              "fresh_output": True, "command": command,
              "scope": "Process startup, all fits, evaluation, tables and plots; excludes installation and subsequent verification",
              "started_at_utc": datetime.now(timezone.utc).isoformat()}
    metadata = output / "benchmark_process.json"
    started = time.perf_counter()
    with (output / "benchmark.log").open("w") as log:
        process = subprocess.Popen(command, cwd=code, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        record["process_id"] = process.pid
        metadata.write_text(json.dumps(record, indent=2) + "\n")
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(f"[{workers}w/{threads}t] {line}", end="", flush=True)
        record["exit_code"] = process.wait()
    record["wall_seconds"] = time.perf_counter() - started
    record["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    metadata.write_text(json.dumps(record, indent=2) + "\n")
    if record["exit_code"]:
        raise RuntimeError(f"Benchmark failed: {output}")
    record["completed_splits"] = len(list(output.glob("*/seed_*/metrics.json")))
    record["expected_splits"] = len(record["datasets"]) * runs
    if record["completed_splits"] != record["expected_splits"]:
        raise RuntimeError(f"Incomplete benchmark: {output}")
    check_started = time.perf_counter()
    with (output / "verification.log").open("w") as log:
        checked = subprocess.run([sys.executable, "verify_evaluation.py", "--results", str(output.resolve())],
                                 cwd=code, stdout=log, stderr=subprocess.STDOUT)
    record["verification_seconds"] = time.perf_counter() - check_started
    record["verification_exit_code"] = checked.returncode
    if reference is not None:
        record["reference_comparison"] = compare_outputs(output, reference)
    metadata.write_text(json.dumps(record, indent=2) + "\n")
    print(f"MEASURED {workers}w/{threads}t: {record['wall_seconds']:.2f}s; "
          f"verification exit={checked.returncode}", flush=True)
    if checked.returncode:
        raise RuntimeError(f"Verification failed: {output}")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", default=["2x2", "3x2"])
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--datasets", nargs="+", choices=["FashionMNIST0.3", "FashionMNIST0.6", "CIFAR"])
    args = parser.parse_args()
    configurations = []
    for value in args.configs:
        try:
            workers, threads = map(int, value.split("x"))
            if min(workers, threads, args.runs) < 1:
                raise ValueError
        except ValueError:
            parser.error("Configs must be positive WORKERSxTHREADS values; runs must be positive")
        configurations.append((workers, threads))
    code = Path(__file__).resolve().parent
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    for workers, threads in configurations:
        records.append(run_configuration(code, args.out / f"w{workers}_t{threads}", workers, threads,
                                         args.runs, args.reference, args.datasets))
        (args.out / "comparison.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
