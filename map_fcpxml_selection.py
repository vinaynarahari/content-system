from __future__ import annotations

import argparse
import copy
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


def parse_time(value: str) -> float:
    value = value.strip()
    if not value.endswith("s"):
        raise ValueError(f"Unsupported time format: {value}")
    value = value[:-1]
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        return float(numerator) / float(denominator)
    return float(value)


def format_time(seconds: float) -> str:
    frames = round(seconds * 30)
    return f"{frames}/30s"


@dataclass(frozen=True)
class Clip:
    index: int
    offset: float
    start: float
    duration: float

    @property
    def end(self) -> float:
        return self.start + self.duration


@dataclass(frozen=True)
class Overlap:
    keep_index: int
    keep_start: float
    keep_end: float
    source_start: float
    source_end: float

    @property
    def duration(self) -> float:
        return self.source_end - self.source_start


def load_fcpxml_clips(path: Path) -> tuple[ET.ElementTree, ET.Element, ET.Element, list[Clip]]:
    tree = ET.parse(path)
    root = tree.getroot()
    sequence = root.find(".//sequence")
    if sequence is None:
        raise ValueError(f"No <sequence> found in {path}")
    spine = sequence.find("spine")
    if spine is None:
        raise ValueError(f"No <spine> found in {path}")

    clips: list[Clip] = []
    for index, node in enumerate(spine.findall("asset-clip"), start=1):
        clips.append(
            Clip(
                index=index,
                offset=parse_time(node.get("offset", "0s")),
                start=parse_time(node.attrib["start"]),
                duration=parse_time(node.attrib["duration"]),
            )
        )
    return tree, sequence, spine, clips


def keep_ranges_from_report(path: Path) -> tuple[float, list[tuple[float, float]]]:
    report = json.loads(path.read_text())
    total_duration = float(report["input_seconds"])
    cut_ranges = []
    for item in report["cut_ranges"]:
        start, end = item.split(",", 1)
        cut_ranges.append((parse_time(start), parse_time(end)))

    keep_ranges: list[tuple[float, float]] = []
    cursor = 0.0
    for cut_start, cut_end in cut_ranges:
        if cut_start > cursor:
            keep_ranges.append((cursor, cut_start))
        cursor = cut_end
    if total_duration > cursor:
        keep_ranges.append((cursor, total_duration))
    return total_duration, keep_ranges


def overlaps_for_clips(source_clips: list[Clip], keep_ranges: list[tuple[float, float]]) -> list[dict]:
    results = []
    for clip in source_clips:
        overlaps: list[Overlap] = []
        for keep_index, (keep_start, keep_end) in enumerate(keep_ranges, start=1):
            overlap_start = max(clip.start, keep_start)
            overlap_end = min(clip.end, keep_end)
            if overlap_end > overlap_start:
                overlaps.append(
                    Overlap(
                        keep_index=keep_index,
                        keep_start=keep_start,
                        keep_end=keep_end,
                        source_start=overlap_start,
                        source_end=overlap_end,
                    )
                )
        if overlaps:
            results.append(
                {
                    "clip": clip,
                    "overlaps": overlaps,
                }
            )
    return results


def write_report(path: Path, source_path: Path, report_path: Path, overlaps: list[dict]) -> None:
    payload = {
        "source_fcpxml": str(source_path),
        "selector_report": str(report_path),
        "used_take_count": len(overlaps),
        "used_takes": [
            {
                "take_index": item["clip"].index,
                "source_start_seconds": round(item["clip"].start, 3),
                "source_end_seconds": round(item["clip"].end, 3),
                "take_duration_seconds": round(item["clip"].duration, 3),
                "used_duration_seconds": round(sum(part.duration for part in item["overlaps"]), 3),
                "used_parts": [
                    {
                        "keep_index": part.keep_index,
                        "keep_source_start_seconds": round(part.keep_start, 3),
                        "keep_source_end_seconds": round(part.keep_end, 3),
                        "overlap_start_seconds": round(part.source_start, 3),
                        "overlap_end_seconds": round(part.source_end, 3),
                        "overlap_duration_seconds": round(part.duration, 3),
                    }
                    for part in item["overlaps"]
                ],
            }
            for item in overlaps
        ],
    }
    path.write_text(json.dumps(payload, indent=2))


def build_output_tree(
    source_tree: ET.ElementTree,
    source_sequence: ET.Element,
    overlaps: list[dict],
    project_name: str,
    mode: str,
) -> ET.ElementTree:
    source_root = source_tree.getroot()
    resources = source_root.find("resources")
    library = source_root.find("library")
    event = library.find("event") if library is not None else None
    if resources is None or library is None or event is None:
        raise ValueError("Source FCPXML is missing required top-level nodes.")

    root = ET.Element(source_root.tag, source_root.attrib)
    root.append(copy.deepcopy(resources))

    library_node = ET.SubElement(root, "library")
    event_node = ET.SubElement(library_node, "event", event.attrib)
    project_node = ET.SubElement(event_node, "project", {"name": project_name})

    sequence_node = ET.SubElement(project_node, "sequence", source_sequence.attrib)
    spine_node = ET.SubElement(sequence_node, "spine")

    offset = 0.0
    for item in overlaps:
        clip = item["clip"]
        if mode == "full":
            node = ET.SubElement(
                spine_node,
                "asset-clip",
                {
                    "offset": format_time(offset),
                    "duration": format_time(clip.duration),
                    "tcFormat": "NDF",
                    "start": format_time(clip.start),
                    "name": "IMG_7969",
                    "ref": "r2",
                },
            )
            offset += clip.duration
            continue

        for overlap in item["overlaps"]:
            ET.SubElement(
                spine_node,
                "asset-clip",
                {
                    "offset": format_time(offset),
                    "duration": format_time(overlap.duration),
                    "tcFormat": "NDF",
                    "start": format_time(overlap.source_start),
                    "name": "IMG_7969",
                    "ref": "r2",
                },
            )
            offset += overlap.duration

    return ET.ElementTree(root)


def indent_xml(node: ET.Element, level: int = 0) -> None:
    whitespace = "\n" + ("  " * level)
    child_whitespace = "\n" + ("  " * (level + 1))
    if len(node):
        if not node.text or not node.text.strip():
            node.text = child_whitespace
        for child in node:
            indent_xml(child, level + 1)
        if not node[-1].tail or not node[-1].tail.strip():
            node[-1].tail = whitespace
    if level and (not node.tail or not node.tail.strip()):
        node.tail = whitespace


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Map kept ranges from a dead-space report back onto a longer FCPXML "
            "timeline and emit filtered timelines."
        )
    )
    parser.add_argument("source_fcpxml", type=Path, help="Longer source FCPXML to filter.")
    parser.add_argument("selector_report", type=Path, help="JSON report with cut_ranges to invert into keep ranges.")
    parser.add_argument("--report-out", type=Path, required=True, help="Path for the overlap report JSON.")
    parser.add_argument("--full-out", type=Path, required=True, help="Path for the full-takes filtered FCPXML.")
    parser.add_argument("--trimmed-out", type=Path, required=True, help="Path for the exact-overlap filtered FCPXML.")
    args = parser.parse_args()

    source_tree, source_sequence, _, source_clips = load_fcpxml_clips(args.source_fcpxml)
    _, keep_ranges = keep_ranges_from_report(args.selector_report)
    overlaps = overlaps_for_clips(source_clips, keep_ranges)

    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.full_out.parent.mkdir(parents=True, exist_ok=True)
    args.trimmed_out.parent.mkdir(parents=True, exist_ok=True)

    write_report(args.report_out, args.source_fcpxml, args.selector_report, overlaps)

    full_tree = build_output_tree(
        source_tree=source_tree,
        source_sequence=source_sequence,
        overlaps=overlaps,
        project_name=f"{args.source_fcpxml.stem}_used_takes",
        mode="full",
    )
    indent_xml(full_tree.getroot())
    full_tree.write(args.full_out, encoding="utf-8", xml_declaration=True)

    trimmed_tree = build_output_tree(
        source_tree=source_tree,
        source_sequence=source_sequence,
        overlaps=overlaps,
        project_name=f"{args.source_fcpxml.stem}_trimmed_to_recovered56",
        mode="trimmed",
    )
    indent_xml(trimmed_tree.getroot())
    trimmed_tree.write(args.trimmed_out, encoding="utf-8", xml_declaration=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
