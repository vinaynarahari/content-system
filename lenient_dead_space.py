#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_THRESHOLD = 0.015
DEFAULT_EXIT_RATIO = 0.4
DEFAULT_RELEASE = 0.35
DEFAULT_LEAD_IN = 0.20
DEFAULT_TAIL_OUT = 0.80
DEFAULT_MIN_CUT = 0.90
DEFAULT_MIN_CLIP = 0.60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a conservative dead-space cut that only removes sustained "
            "near-silence and leaves broader clips for manual review."
        )
    )
    parser.add_argument("input", type=Path, help="Input media file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output path. Defaults to <input>_lenient.fcpxml for Final Cut Pro exports.",
    )
    parser.add_argument(
        "--export",
        default="final-cut-pro",
        help="auto-editor export mode. Defaults to final-cut-pro.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=(
            "Enter threshold for detecting live audio. Lower values are more lenient. "
            f"Defaults to {DEFAULT_THRESHOLD:.3f}."
        ),
    )
    parser.add_argument(
        "--exit-ratio",
        type=float,
        default=DEFAULT_EXIT_RATIO,
        help=(
            "Close a clip when the level falls below threshold * exit-ratio for "
            f"long enough. Defaults to {DEFAULT_EXIT_RATIO:.2f}."
        ),
    )
    parser.add_argument(
        "--release",
        type=float,
        default=DEFAULT_RELEASE,
        help=f"Seconds of near-silence required before closing a clip. Defaults to {DEFAULT_RELEASE:.2f}s.",
    )
    parser.add_argument(
        "--lead-in",
        type=float,
        default=DEFAULT_LEAD_IN,
        help=f"Seconds to keep before each clip. Defaults to {DEFAULT_LEAD_IN:.2f}s.",
    )
    parser.add_argument(
        "--tail-out",
        type=float,
        default=DEFAULT_TAIL_OUT,
        help=f"Seconds to keep after each clip. Defaults to {DEFAULT_TAIL_OUT:.2f}s.",
    )
    parser.add_argument(
        "--min-cut",
        type=float,
        default=DEFAULT_MIN_CUT,
        help=f"Merge silent gaps shorter than this many seconds. Defaults to {DEFAULT_MIN_CUT:.2f}s.",
    )
    parser.add_argument(
        "--min-clip",
        type=float,
        default=DEFAULT_MIN_CLIP,
        help=f"Drop clips shorter than this many seconds. Defaults to {DEFAULT_MIN_CLIP:.2f}s.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Analyze and print the chosen settings without exporting.",
    )
    return parser.parse_args()


def default_output_path(input_path: Path, export_mode: str) -> Path:
    suffix = ".fcpxml" if export_mode == "final-cut-pro" else input_path.suffix
    return input_path.with_name(f"{input_path.stem}_lenient{suffix}")


def smart_script_path() -> Path:
    return Path(__file__).with_name("smart_dead_space.py")


def build_command(args: argparse.Namespace, input_path: Path, output_path: Path | None) -> list[str]:
    command = [
        "python3",
        str(smart_script_path()),
        str(input_path),
        "--export",
        args.export,
        "--enter-threshold",
        f"{args.threshold:.6f}",
        "--exit-ratio",
        str(args.exit_ratio),
        "--release",
        str(args.release),
        "--lead-in",
        str(args.lead_in),
        "--tail-out",
        str(args.tail_out),
        "--min-cut",
        str(args.min_cut),
        "--min-clip",
        str(args.min_clip),
    ]
    if output_path is not None:
        command.extend(["-o", str(output_path)])
    if args.preview:
        command.append("--preview")
    return command


def main() -> int:
    args = parse_args()
    script_path = smart_script_path()
    if not script_path.exists():
        print(f"Missing dependency: {script_path}", file=sys.stderr)
        return 1

    input_path = args.input.expanduser().resolve()
    output_path = None
    if not args.preview:
        output_path = args.output.expanduser().resolve() if args.output else default_output_path(input_path, args.export)

    command = build_command(args=args, input_path=input_path, output_path=output_path)
    completed = subprocess.run(command, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
