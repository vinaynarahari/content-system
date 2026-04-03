from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from statistics import mean, median
from typing import Iterable

DEFAULT_TARGET_SECONDS = 60.0
DEFAULT_EXIT_RATIO = 0.34
DEFAULT_RELEASE_SECONDS = 0.10
DEFAULT_LEAD_IN_SECONDS = 0.07
DEFAULT_TAIL_OUT_SECONDS = 0.40
DEFAULT_MIN_CUT_SECONDS = 0.30
DEFAULT_MIN_CLIP_SECONDS = 0.20
DEFAULT_SEARCH_MIN = 0.020
DEFAULT_SEARCH_MAX = 0.350
DEFAULT_COARSE_STEP = 0.005
DEFAULT_FINE_STEP = 0.0005
DEFAULT_CACHE_DIR = Path("analysis")


@dataclass(frozen=True)
class DetectorSettings:
    enter_threshold: float
    exit_threshold: float
    release_frames: int
    lead_in_frames: int
    tail_out_frames: int
    min_cut_frames: int
    min_clip_frames: int


@dataclass(frozen=True)
class DetectionResult:
    settings: DetectorSettings
    frame_rate: float
    total_frames: int
    kept_frames: int
    clip_count: int
    cut_count: int
    clip_lengths_seconds: list[float]
    cut_lengths_seconds: list[float]
    cut_ranges: list[str]

    @property
    def input_seconds(self) -> float:
        return self.total_frames / self.frame_rate

    @property
    def output_seconds(self) -> float:
        return self.kept_frames / self.frame_rate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find dead space with a two-threshold gate, then export explicit "
            "cut ranges through auto-editor."
        )
    )
    parser.add_argument("input", type=Path, help="Input media file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output path. Defaults to <input>_smart.fcpxml for Final Cut Pro exports.",
    )
    parser.add_argument(
        "--export",
        default="final-cut-pro",
        help="auto-editor export mode. Defaults to final-cut-pro.",
    )
    parser.add_argument(
        "--target-seconds",
        type=float,
        default=DEFAULT_TARGET_SECONDS,
        help=(
            "Target kept duration in seconds when auto-tuning the enter threshold. "
            f"Defaults to {DEFAULT_TARGET_SECONDS:.1f}."
        ),
    )
    parser.add_argument(
        "--enter-threshold",
        type=float,
        help="Set the enter threshold directly instead of auto-tuning it.",
    )
    parser.add_argument(
        "--exit-threshold",
        type=float,
        help="Set the exit threshold directly. Defaults to enter-threshold * --exit-ratio.",
    )
    parser.add_argument(
        "--exit-ratio",
        type=float,
        default=DEFAULT_EXIT_RATIO,
        help=(
            "When --exit-threshold is omitted, use enter-threshold * exit-ratio. "
            f"Defaults to {DEFAULT_EXIT_RATIO:.2f}."
        ),
    )
    parser.add_argument(
        "--release",
        type=float,
        default=DEFAULT_RELEASE_SECONDS,
        help=(
            "How long audio must stay below the exit threshold before a clip closes. "
            f"Defaults to {DEFAULT_RELEASE_SECONDS:.2f}s."
        ),
    )
    parser.add_argument(
        "--lead-in",
        type=float,
        default=DEFAULT_LEAD_IN_SECONDS,
        help=f"Seconds to keep before each detected clip. Defaults to {DEFAULT_LEAD_IN_SECONDS:.2f}s.",
    )
    parser.add_argument(
        "--tail-out",
        type=float,
        default=DEFAULT_TAIL_OUT_SECONDS,
        help=f"Seconds to keep after each detected clip. Defaults to {DEFAULT_TAIL_OUT_SECONDS:.2f}s.",
    )
    parser.add_argument(
        "--min-cut",
        type=float,
        default=DEFAULT_MIN_CUT_SECONDS,
        help=f"Merge cut gaps shorter than this many seconds. Defaults to {DEFAULT_MIN_CUT_SECONDS:.2f}s.",
    )
    parser.add_argument(
        "--min-clip",
        type=float,
        default=DEFAULT_MIN_CLIP_SECONDS,
        help=f"Drop clips shorter than this many seconds. Defaults to {DEFAULT_MIN_CLIP_SECONDS:.2f}s.",
    )
    parser.add_argument(
        "--search-min",
        type=float,
        default=DEFAULT_SEARCH_MIN,
        help="Minimum enter threshold to try during auto-tuning.",
    )
    parser.add_argument(
        "--search-max",
        type=float,
        default=DEFAULT_SEARCH_MAX,
        help="Maximum enter threshold to try during auto-tuning.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Directory for cached auto-editor level dumps and reports.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Analyze and print the chosen settings without exporting.",
    )
    parser.add_argument(
        "--force-recompute-levels",
        action="store_true",
        help="Ignore any cached levels file and recompute it.",
    )
    return parser.parse_args()


def run_command(command: list[str], capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=True,
        capture_output=capture_output,
        text=True,
    )


def format_duration(seconds: float) -> str:
    minutes = int(seconds // 60)
    remainder = seconds - (minutes * 60)
    return f"{minutes}:{remainder:05.2f}"


def timebase_from_info(input_path: Path) -> float:
    result = run_command(["auto-editor", "info", str(input_path)], capture_output=True)
    match = re.search(r"recommendedTimebase:\s+([0-9]+/[0-9]+|[0-9.]+)", result.stdout)
    if not match:
        raise ValueError("Could not find recommendedTimebase in auto-editor info output.")
    return float(Fraction(match.group(1)))


def default_output_path(input_path: Path, export_mode: str) -> Path:
    suffix = ".fcpxml" if export_mode == "final-cut-pro" else input_path.suffix
    return input_path.with_name(f"{input_path.stem}_smart{suffix}")


def levels_cache_path(cache_dir: Path, input_path: Path) -> Path:
    return cache_dir / f"{input_path.stem}.levels.txt"


def report_path(cache_dir: Path, input_path: Path) -> Path:
    return cache_dir / f"{input_path.stem}.dead-space-report.json"


def load_levels(cache_path: Path) -> list[float]:
    values: list[float] = []
    for line in cache_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("@"):
            continue
        values.append(float(line))
    if not values:
        raise ValueError(f"No level values found in {cache_path}.")
    return values


def get_levels(input_path: Path, cache_dir: Path, force_recompute: bool) -> tuple[Path, list[float]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = levels_cache_path(cache_dir, input_path)

    if force_recompute or not cache_path.exists():
        result = run_command(["auto-editor", "levels", str(input_path)], capture_output=True)
        cache_path.write_text(result.stdout)

    return cache_path, load_levels(cache_path)


def frames_from_seconds(seconds: float, frame_rate: float) -> int:
    return max(0, round(seconds * frame_rate))


def iter_runs(mask: list[bool]) -> Iterable[tuple[int, int, bool]]:
    length = len(mask)
    index = 0
    while index < length:
        value = mask[index]
        next_index = index + 1
        while next_index < length and mask[next_index] == value:
            next_index += 1
        yield index, next_index, value
        index = next_index


def hysteresis_mask(levels: list[float], enter_threshold: float, exit_threshold: float, release_frames: int) -> list[bool]:
    active = False
    below_count = 0
    mask = [False] * len(levels)

    for index, value in enumerate(levels):
        if active:
            mask[index] = True
            if value < exit_threshold:
                below_count += 1
                if below_count >= release_frames:
                    active = False
            else:
                below_count = 0
            continue

        if value >= enter_threshold:
            active = True
            below_count = 0
            mask[index] = True

    return mask


def apply_margin(mask: list[bool], lead_in_frames: int, tail_out_frames: int) -> list[bool]:
    expanded = mask[:]
    total = len(mask)

    for start, stop, is_kept in iter_runs(mask):
        if not is_kept:
            continue
        left = max(0, start - lead_in_frames)
        right = min(total, stop + tail_out_frames)
        expanded[left:right] = [True] * (right - left)

    return expanded


def smooth_mask(mask: list[bool], min_cut_frames: int, min_clip_frames: int) -> list[bool]:
    smoothed = mask[:]
    changed = True

    while changed:
        changed = False
        for start, stop, is_kept in list(iter_runs(smoothed)):
            if is_kept:
                continue
            if stop - start < min_cut_frames:
                smoothed[start:stop] = [True] * (stop - start)
                changed = True

        for start, stop, is_kept in list(iter_runs(smoothed)):
            if not is_kept:
                continue
            if stop - start < min_clip_frames:
                smoothed[start:stop] = [False] * (stop - start)
                changed = True

    return smoothed


def detect_segments(levels: list[float], frame_rate: float, settings: DetectorSettings) -> DetectionResult:
    mask = hysteresis_mask(
        levels=levels,
        enter_threshold=settings.enter_threshold,
        exit_threshold=settings.exit_threshold,
        release_frames=settings.release_frames,
    )
    mask = apply_margin(
        mask=mask,
        lead_in_frames=settings.lead_in_frames,
        tail_out_frames=settings.tail_out_frames,
    )
    mask = smooth_mask(
        mask=mask,
        min_cut_frames=settings.min_cut_frames,
        min_clip_frames=settings.min_clip_frames,
    )

    clips: list[tuple[int, int]] = []
    cuts: list[tuple[int, int]] = []
    for start, stop, is_kept in iter_runs(mask):
        if is_kept:
            clips.append((start, stop))
        else:
            cuts.append((start, stop))

    if not clips:
        raise ValueError("This configuration would cut the entire video.")

    cut_ranges = [
        f"{start / frame_rate:.3f}s,{stop / frame_rate:.3f}s"
        for start, stop in cuts
        if stop > start
    ]

    return DetectionResult(
        settings=settings,
        frame_rate=frame_rate,
        total_frames=len(mask),
        kept_frames=sum(mask),
        clip_count=len(clips),
        cut_count=len(cuts),
        clip_lengths_seconds=[(stop - start) / frame_rate for start, stop in clips],
        cut_lengths_seconds=[(stop - start) / frame_rate for start, stop in cuts],
        cut_ranges=cut_ranges,
    )


def make_settings(
    enter_threshold: float,
    exit_threshold: float | None,
    exit_ratio: float,
    frame_rate: float,
    release: float,
    lead_in: float,
    tail_out: float,
    min_cut: float,
    min_clip: float,
) -> DetectorSettings:
    computed_exit = exit_threshold if exit_threshold is not None else enter_threshold * exit_ratio
    return DetectorSettings(
        enter_threshold=enter_threshold,
        exit_threshold=computed_exit,
        release_frames=max(1, frames_from_seconds(release, frame_rate)),
        lead_in_frames=frames_from_seconds(lead_in, frame_rate),
        tail_out_frames=frames_from_seconds(tail_out, frame_rate),
        min_cut_frames=frames_from_seconds(min_cut, frame_rate),
        min_clip_frames=frames_from_seconds(min_clip, frame_rate),
    )


def frange(start: float, stop: float, step: float) -> Iterable[float]:
    count = 0
    value = start
    while value <= stop + (step / 10):
        yield round(value, 6)
        count += 1
        value = start + (count * step)


def choose_settings(args: argparse.Namespace, levels: list[float], frame_rate: float) -> DetectionResult:
    if args.enter_threshold is not None:
        settings = make_settings(
            enter_threshold=args.enter_threshold,
            exit_threshold=args.exit_threshold,
            exit_ratio=args.exit_ratio,
            frame_rate=frame_rate,
            release=args.release,
            lead_in=args.lead_in,
            tail_out=args.tail_out,
            min_cut=args.min_cut,
            min_clip=args.min_clip,
        )
        return detect_segments(levels, frame_rate, settings)

    if args.search_min >= args.search_max:
        raise ValueError("--search-min must be smaller than --search-max.")

    coarse_results: list[tuple[float, float, DetectionResult]] = []
    for threshold in frange(args.search_min, args.search_max, DEFAULT_COARSE_STEP):
        settings = make_settings(
            enter_threshold=threshold,
            exit_threshold=args.exit_threshold,
            exit_ratio=args.exit_ratio,
            frame_rate=frame_rate,
            release=args.release,
            lead_in=args.lead_in,
            tail_out=args.tail_out,
            min_cut=args.min_cut,
            min_clip=args.min_clip,
        )
        result = detect_segments(levels, frame_rate, settings)
        error = abs(result.output_seconds - args.target_seconds)
        coarse_results.append((error, threshold, result))

    coarse_results.sort(key=lambda item: item[0])
    _, coarse_threshold, _ = coarse_results[0]

    fine_start = max(args.search_min, coarse_threshold - DEFAULT_COARSE_STEP)
    fine_stop = min(args.search_max, coarse_threshold + DEFAULT_COARSE_STEP)

    fine_results: list[tuple[float, float, DetectionResult]] = []
    for threshold in frange(fine_start, fine_stop, DEFAULT_FINE_STEP):
        settings = make_settings(
            enter_threshold=threshold,
            exit_threshold=args.exit_threshold,
            exit_ratio=args.exit_ratio,
            frame_rate=frame_rate,
            release=args.release,
            lead_in=args.lead_in,
            tail_out=args.tail_out,
            min_cut=args.min_cut,
            min_clip=args.min_clip,
        )
        result = detect_segments(levels, frame_rate, settings)
        error = abs(result.output_seconds - args.target_seconds)
        fine_results.append((error, threshold, result))

    fine_results.sort(key=lambda item: (item[0], item[2].clip_count))
    return fine_results[0][2]


def write_report(report_file: Path, input_path: Path, levels_file: Path, result: DetectionResult) -> None:
    payload = {
        "input": str(input_path),
        "levels_file": str(levels_file),
        "frame_rate": result.frame_rate,
        "input_seconds": result.input_seconds,
        "output_seconds": result.output_seconds,
        "clip_count": result.clip_count,
        "cut_count": result.cut_count,
        "settings": asdict(result.settings),
        "clip_stats": {
            "smallest": min(result.clip_lengths_seconds),
            "median": median(result.clip_lengths_seconds),
            "average": mean(result.clip_lengths_seconds),
            "largest": max(result.clip_lengths_seconds),
        },
        "cut_stats": {
            "smallest": min(result.cut_lengths_seconds),
            "median": median(result.cut_lengths_seconds),
            "average": mean(result.cut_lengths_seconds),
            "largest": max(result.cut_lengths_seconds),
        },
        "cut_ranges": result.cut_ranges,
    }
    report_file.write_text(json.dumps(payload, indent=2))


def export_cut_ranges(input_path: Path, output_path: Path, export_mode: str, cut_ranges: list[str]) -> None:
    command = [
        "auto-editor",
        str(input_path),
        "--edit",
        "none",
        "--export",
        export_mode,
        "-o",
        str(output_path),
        "--no-open",
    ]
    for cut_range in cut_ranges:
        command.extend(["--cut", cut_range])
    run_command(command)


def print_summary(result: DetectionResult, input_path: Path, output_path: Path | None, levels_file: Path, report_file: Path) -> None:
    settings = result.settings
    print(f"Input:        {input_path}")
    print(f"Levels:       {levels_file}")
    print(f"Report:       {report_file}")
    if output_path is not None:
        print(f"Output:       {output_path}")
    print(f"Frame Rate:   {result.frame_rate:.3f}")
    print(f"Input Length: {format_duration(result.input_seconds)}")
    print(f"Output Length:{format_duration(result.output_seconds)}")
    print(f"Kept:         {result.output_seconds / result.input_seconds:.2%}")
    print()
    print("Detector:")
    print(f"  enter-threshold: {settings.enter_threshold:.4f}")
    print(f"  exit-threshold:  {settings.exit_threshold:.4f}")
    print(f"  release:         {settings.release_frames / result.frame_rate:.2f}s")
    print(f"  lead-in:         {settings.lead_in_frames / result.frame_rate:.2f}s")
    print(f"  tail-out:        {settings.tail_out_frames / result.frame_rate:.2f}s")
    print(f"  min-cut:         {settings.min_cut_frames / result.frame_rate:.2f}s")
    print(f"  min-clip:        {settings.min_clip_frames / result.frame_rate:.2f}s")
    print()
    print("Segments:")
    print(f"  clips:           {result.clip_count}")
    print(f"  cuts:            {result.cut_count}")
    print(f"  smallest clip:   {min(result.clip_lengths_seconds):.2f}s")
    print(f"  median clip:     {median(result.clip_lengths_seconds):.2f}s")
    print(f"  average clip:    {mean(result.clip_lengths_seconds):.2f}s")
    print(f"  largest clip:    {max(result.clip_lengths_seconds):.2f}s")


def validate_args(args: argparse.Namespace) -> None:
    if not args.input.exists():
        raise FileNotFoundError(f"Input file does not exist: {args.input}")
    if args.exit_ratio <= 0:
        raise ValueError("--exit-ratio must be greater than zero.")
    if args.target_seconds <= 0:
        raise ValueError("--target-seconds must be greater than zero.")


def main() -> int:
    args = parse_args()

    try:
        validate_args(args)
        input_path = args.input.expanduser().resolve()
        output_path = args.output.expanduser().resolve() if args.output else default_output_path(input_path, args.export)

        frame_rate = timebase_from_info(input_path)
        levels_file, levels = get_levels(
            input_path=input_path,
            cache_dir=args.cache_dir,
            force_recompute=args.force_recompute_levels,
        )
        result = choose_settings(args=args, levels=levels, frame_rate=frame_rate)

        report_file = report_path(args.cache_dir, input_path)
        write_report(report_file, input_path, levels_file, result)
        print_summary(
            result=result,
            input_path=input_path,
            output_path=None if args.preview else output_path,
            levels_file=levels_file,
            report_file=report_file,
        )

        if args.preview:
            return 0

        export_cut_ranges(
            input_path=input_path,
            output_path=output_path,
            export_mode=args.export,
            cut_ranges=result.cut_ranges,
        )
        return 0
    except subprocess.CalledProcessError as error:
        if error.stderr:
            print(error.stderr.strip(), file=sys.stderr)
        elif error.stdout:
            print(error.stdout.strip(), file=sys.stderr)
        else:
            print(str(error), file=sys.stderr)
        return error.returncode or 1
    except Exception as error:  # noqa: BLE001
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
