#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
COMPLETED_DIR = SCRIPT_DIR / "Completed"

SOURCES = {
    "code": COMPLETED_DIR / "tech_code_surface.mov",
    "training": COMPLETED_DIR / "tech_model_training.mov",
    "sorting": COMPLETED_DIR / "tech_sorting_trace.mov",
    "agent": COMPLETED_DIR / "tech_agent_flow.mov",
}

SEQUENCES = {
    "stim_cut_01": [
        ("code", 1.0),
        ("training", 3.8),
        ("sorting", 1.8),
        ("agent", 5.2),
        ("code", 8.4),
    ],
    "stim_cut_02": [
        ("training", 1.2),
        ("code", 4.6),
        ("agent", 7.4),
        ("sorting", 4.8),
        ("training", 9.6),
    ],
    "stim_cut_03": [
        ("agent", 1.4),
        ("sorting", 6.2),
        ("code", 6.8),
        ("training", 7.2),
        ("agent", 10.0),
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate short 2-second-cut stim montages from tech clips.")
    parser.add_argument(
        "--segment-seconds",
        type=float,
        default=2.0,
        help="Length of each cut segment.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        choices=sorted(SEQUENCES.keys()),
        help="Render only the selected sequence ids.",
    )
    parser.add_argument(
        "--portrait",
        action="store_true",
        help="Render into a 1080x1920 portrait canvas with padding instead of cropping.",
    )
    return parser.parse_args()


def render_sequence(name, sequence, segment_seconds, portrait=False):
    suffix = "_vertical" if portrait else ""
    output_path = COMPLETED_DIR / f"{name}{suffix}.mov"
    input_order = []
    command = ["ffmpeg", "-y"]
    for source_key in SOURCES:
        input_order.append(source_key)
        command.extend(["-i", str(SOURCES[source_key])])

    filter_parts = []
    concat_inputs = []
    source_index = {name: index for index, name in enumerate(input_order)}

    for segment_index, (source_name, start_time) in enumerate(sequence):
        input_index = source_index[source_name]
        label = f"v{segment_index}"
        filter_parts.append(
            f"[{input_index}:v]trim=start={start_time}:end={start_time + segment_seconds},"
            f"setpts=PTS-STARTPTS[{label}]"
        )
        concat_inputs.append(f"[{label}]")

    concat_graph = "".join(concat_inputs) + f"concat=n={len(sequence)}:v=1:a=0"
    if portrait:
        concat_graph += ",scale=1080:1920:force_original_aspect_ratio=decrease"
        concat_graph += ",pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black"
    concat_graph += ",setsar=1,format=yuv422p10le[vout]"
    filter_parts.append(concat_graph)
    command.extend(
        [
            "-filter_complex",
            ";".join(filter_parts),
            "-map",
            "[vout]",
            "-an",
            "-c:v",
            "prores_ks",
            "-profile:v",
            "3",
            "-pix_fmt",
            "yuv422p10le",
            "-colorspace",
            "1",
            "-color_primaries",
            "1",
            "-color_trc",
            "1",
            str(output_path),
        ]
    )

    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path


def main():
    args = parse_args()
    selected = args.only or sorted(SEQUENCES.keys())

    for name in selected:
        print(f"Rendering {name}")
        render_sequence(name, SEQUENCES[name], args.segment_seconds, portrait=args.portrait)

    print("Done.")


if __name__ == "__main__":
    main()
