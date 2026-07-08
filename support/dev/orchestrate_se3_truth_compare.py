#!/usr/bin/env python3
"""
Run SE(3) smoothing on probe/tanker navcalcs CSVs, then run truth comparison.

This is a thin orchestration wrapper around:

    - apply_se3_factor_graph_navcalcs.py
    - truth_compare_navcalcs.py

The probe and tanker smoothing steps are independent, so they can be launched in
parallel. After both finish successfully, the truth-comparison script is invoked
using the resulting CSVs and the provided TSPI file.

Example:
    python orchestrate_se3_truth_compare.py \
        --probe-csv navcalcs_probe.csv \
        --tanker-csv navcalcs_tanker.csv \
        --novatel-txt relative_tspi.txt \
        --se3-arg=--inflate-measurement-covariances \
        --se3-arg=--fg-cov-alpha-trans --se3-arg=25 \
        --se3-arg=--fg-cov-alpha-rot --se3-arg=9
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path


HERE = Path(__file__).resolve().parent
APPLY_SE3_SCRIPT = HERE / "apply_se3_factor_graph_navcalcs.py"
TRUTH_COMPARE_SCRIPT = HERE / "truth_compare_navcalcs.py"


@dataclass(frozen=True)
class CommandResult:
    label: str
    command: list[str]
    returncode: int
    stdout: str
    stderr: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run apply_se3_factor_graph_navcalcs on probe and tanker CSVs, then "
            "run truth_compare_navcalcs on the resulting files."
        )
    )
    parser.add_argument("--probe-csv", type=Path, required=True, help="Probe/receiver navcalcs CSV.")
    parser.add_argument("--tanker-csv", type=Path, required=True, help="Tanker/target navcalcs CSV.")
    parser.add_argument("--novatel-txt", type=Path, required=True, help="Whitespace-delimited NovAtel/TSPI text file.")
    parser.add_argument(
        "--probe-output",
        type=Path,
        default=None,
        help="Optional output path for the smoothed probe CSV. Default overwrites --probe-csv.",
    )
    parser.add_argument(
        "--tanker-output",
        type=Path,
        default=None,
        help="Optional output path for the smoothed tanker CSV. Default overwrites --tanker-csv.",
    )
    parser.add_argument(
        "--truth-out",
        type=Path,
        default=None,
        help="Optional output directory for truth_compare_navcalcs.py.",
    )
    parser.add_argument(
        "--python-exe",
        type=Path,
        default=Path(sys.executable),
        help="Python interpreter used to launch the subordinate scripts. Default is the current interpreter.",
    )
    parser.add_argument(
        "--serial",
        action="store_true",
        help="Run probe/tanker smoothing serially instead of in parallel.",
    )
    parser.add_argument(
        "--se3-arg",
        action="append",
        default=[],
        help="Argument forwarded to both apply_se3_factor_graph_navcalcs.py runs. Repeat as needed.",
    )
    parser.add_argument(
        "--probe-se3-arg",
        action="append",
        default=[],
        help="Argument forwarded only to the probe apply_se3_factor_graph_navcalcs.py run. Repeat as needed.",
    )
    parser.add_argument(
        "--tanker-se3-arg",
        action="append",
        default=[],
        help="Argument forwarded only to the tanker apply_se3_factor_graph_navcalcs.py run. Repeat as needed.",
    )
    parser.add_argument(
        "--truth-arg",
        action="append",
        default=[],
        help="Argument forwarded to truth_compare_navcalcs.py. Repeat as needed.",
    )
    return parser


def normalize_paths(args: argparse.Namespace) -> argparse.Namespace:
    args.probe_csv = args.probe_csv.resolve()
    args.tanker_csv = args.tanker_csv.resolve()
    args.novatel_txt = args.novatel_txt.resolve()
    args.python_exe = args.python_exe.resolve()
    args.probe_output = args.probe_output.resolve() if args.probe_output is not None else args.probe_csv
    args.tanker_output = args.tanker_output.resolve() if args.tanker_output is not None else args.tanker_csv
    if args.truth_out is not None:
        args.truth_out = args.truth_out.resolve()
    return args


def ensure_inputs_exist(args: argparse.Namespace) -> None:
    required = {
        "probe CSV": args.probe_csv,
        "tanker CSV": args.tanker_csv,
        "NovAtel/TSPI text": args.novatel_txt,
        "apply_se3 script": APPLY_SE3_SCRIPT,
        "truth_compare script": TRUTH_COMPARE_SCRIPT,
        "python executable": args.python_exe,
    }
    missing = [f"{label}: {path}" for label, path in required.items() if not path.exists()]
    if missing:
        raise SystemExit("Missing required paths:\n  " + "\n  ".join(missing))


def run_command(label: str, command: list[str]) -> CommandResult:
    proc = subprocess.run(
        command,
        cwd=str(HERE),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return CommandResult(
        label=label,
        command=command,
        returncode=int(proc.returncode),
        stdout=proc.stdout,
        stderr=proc.stderr,
    )


def print_result(result: CommandResult) -> None:
    print(f"\n--- {result.label} ---")
    print("Command:")
    print("  " + subprocess.list2cmdline(result.command))
    print(f"Exit code: {result.returncode}")
    if result.stdout.strip():
        print("\nstdout:")
        print(result.stdout.rstrip())
    if result.stderr.strip():
        print("\nstderr:")
        print(result.stderr.rstrip())


def build_apply_se3_command(
    python_exe: Path,
    csv_path: Path,
    output_path: Path,
    shared_args: list[str],
    method_args: list[str],
) -> list[str]:
    cmd = [str(python_exe), str(APPLY_SE3_SCRIPT), "--csv", str(csv_path), "--output", str(output_path)]
    cmd.extend(shared_args)
    cmd.extend(method_args)
    return cmd


def build_truth_compare_command(
    python_exe: Path,
    probe_csv: Path,
    tanker_csv: Path,
    novatel_txt: Path,
    out_dir: Path | None,
    truth_args: list[str],
) -> list[str]:
    cmd = [
        str(python_exe),
        str(TRUTH_COMPARE_SCRIPT),
        "--probe-csv",
        str(probe_csv),
        "--tanker-csv",
        str(tanker_csv),
        "--novatel-txt",
        str(novatel_txt),
    ]
    if out_dir is not None:
        cmd.extend(["--out", str(out_dir)])
    cmd.extend(truth_args)
    return cmd


def main() -> None:
    args = normalize_paths(build_parser().parse_args())
    ensure_inputs_exist(args)

    probe_cmd = build_apply_se3_command(
        args.python_exe,
        args.probe_csv,
        args.probe_output,
        list(args.se3_arg),
        list(args.probe_se3_arg),
    )
    tanker_cmd = build_apply_se3_command(
        args.python_exe,
        args.tanker_csv,
        args.tanker_output,
        list(args.se3_arg),
        list(args.tanker_se3_arg),
    )

    print("Starting SE(3) smoothing.")
    print(f"  probe:  {args.probe_csv}")
    print(f"  tanker: {args.tanker_csv}")
    print(f"  mode:   {'serial' if args.serial else 'parallel'}")

    if args.serial:
        se3_results = [
            run_command("apply_se3 probe", probe_cmd),
            run_command("apply_se3 tanker", tanker_cmd),
        ]
    else:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(run_command, "apply_se3 probe", probe_cmd),
                executor.submit(run_command, "apply_se3 tanker", tanker_cmd),
            ]
            se3_results = [future.result() for future in futures]

    failed = [result for result in se3_results if result.returncode != 0]
    for result in se3_results:
        print_result(result)
    if failed:
        raise SystemExit(f"\nSE(3) smoothing failed for {len(failed)} run(s); skipping truth comparison.")

    truth_cmd = build_truth_compare_command(
        args.python_exe,
        args.probe_output,
        args.tanker_output,
        args.novatel_txt,
        args.truth_out,
        list(args.truth_arg),
    )

    print("\nStarting truth comparison.")
    truth_result = run_command("truth_compare", truth_cmd)
    print_result(truth_result)
    if truth_result.returncode != 0:
        raise SystemExit("\ntruth_compare_navcalcs.py failed.")

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
