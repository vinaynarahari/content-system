from __future__ import annotations

import argparse
import copy
import json
import string
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path


TRANSLATION_TABLE = str.maketrans({char: " " for char in string.punctuation if char != "'"})


def parse_time(value: str) -> float:
    value = value.strip()
    if not value.endswith("s"):
        raise ValueError(f"Unsupported time value: {value}")
    value = value[:-1]
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        return float(numerator) / float(denominator)
    return float(value)


def format_time(seconds: float) -> str:
    frames = round(seconds * 30)
    return f"{frames}/30s"


def normalize_text(text: str) -> str:
    return " ".join(text.lower().translate(TRANSLATION_TABLE).split())


def token_similarity(left: str, right: str) -> float:
    left_tokens = left.split()
    right_tokens = right.split()
    if not left_tokens or not right_tokens:
        return 0.0

    left_set = set(left_tokens)
    right_set = set(right_tokens)
    jaccard = len(left_set & right_set) / len(left_set | right_set)
    ratio = SequenceMatcher(None, left, right).ratio()

    prefix_matches = 0
    for left_token, right_token in zip(left_tokens, right_tokens):
        if left_token != right_token:
            break
        prefix_matches += 1
    prefix_score = prefix_matches / max(len(left_tokens), len(right_tokens), 1)
    return max(jaccard, ratio, prefix_score)


@dataclass(frozen=True)
class TimelineClip:
    index: int
    offset: float
    start: float
    duration: float

    @property
    def end(self) -> float:
        return self.offset + self.duration


@dataclass(frozen=True)
class CaptionTitle:
    index: int
    offset: float
    duration: float
    text: str
    normalized: str

    @property
    def end(self) -> float:
        return self.offset + self.duration


@dataclass(frozen=True)
class CaptionMatch:
    final_title: CaptionTitle
    source_title: CaptionTitle
    score: float


@dataclass(frozen=True)
class SourceRange:
    clip_index: int
    take_select_start: float
    take_select_end: float
    source_start: float
    source_end: float

    @property
    def duration(self) -> float:
        return self.source_end - self.source_start


def load_sequence_and_clips(fcpxml_path: Path) -> tuple[ET.ElementTree, ET.Element, list[TimelineClip]]:
    tree = ET.parse(fcpxml_path)
    root = tree.getroot()
    sequence = root.find(".//sequence")
    if sequence is None:
        raise ValueError(f"No <sequence> found in {fcpxml_path}")
    spine = sequence.find("spine")
    if spine is None:
        raise ValueError(f"No <spine> found in {fcpxml_path}")

    clips: list[TimelineClip] = []
    for index, node in enumerate(spine.findall("asset-clip"), start=1):
        clips.append(
            TimelineClip(
                index=index,
                offset=parse_time(node.get("offset", "0s")),
                start=parse_time(node.attrib["start"]),
                duration=parse_time(node.attrib["duration"]),
            )
        )
    return tree, sequence, clips


def load_titles(fcpxml_path: Path) -> list[CaptionTitle]:
    root = ET.parse(fcpxml_path).getroot()
    gap = root.find(".//gap")
    gap_start = parse_time(gap.get("start", "0s")) if gap is not None else 0.0

    titles: list[CaptionTitle] = []
    for index, node in enumerate(root.findall(".//title"), start=1):
        text = " ".join("".join(node.itertext()).split())
        offset = parse_time(node.attrib["offset"]) - gap_start
        duration = parse_time(node.attrib["duration"])
        titles.append(
            CaptionTitle(
                index=index,
                offset=offset,
                duration=duration,
                text=text,
                normalized=normalize_text(text),
            )
        )
    return titles


def align_titles(final_titles: list[CaptionTitle], source_titles: list[CaptionTitle]) -> list[CaptionMatch]:
    final_count = len(final_titles)
    source_count = len(source_titles)
    negative_infinity = -1e18

    previous = [0.0] * (source_count + 1)
    choices = [[0] * (source_count + 1) for _ in range(final_count + 1)]

    for final_index in range(1, final_count + 1):
        current = [negative_infinity] * (source_count + 1)
        current[0] = previous[0] - 1.1
        choices[final_index][0] = 2

        final_title = final_titles[final_index - 1]
        for source_index in range(1, source_count + 1):
            source_title = source_titles[source_index - 1]
            similarity = token_similarity(final_title.normalized, source_title.normalized)

            match_score = previous[source_index - 1] + (similarity * 4.0 - 1.85)
            skip_source_score = current[source_index - 1] - 0.06
            skip_final_score = previous[source_index] - 1.1

            best_score, best_choice = max(
                (skip_source_score, 0),
                (match_score, 1),
                (skip_final_score, 2),
                key=lambda item: item[0],
            )
            current[source_index] = best_score
            choices[final_index][source_index] = best_choice

        previous = current

    matches: list[CaptionMatch] = []
    final_index = final_count
    source_index = source_count
    while final_index > 0 and source_index > 0:
        choice = choices[final_index][source_index]
        if choice == 1:
            final_title = final_titles[final_index - 1]
            source_title = source_titles[source_index - 1]
            score = token_similarity(final_title.normalized, source_title.normalized)
            matches.append(
                CaptionMatch(
                    final_title=final_title,
                    source_title=source_title,
                    score=score,
                )
            )
            final_index -= 1
            source_index -= 1
        elif choice == 0:
            source_index -= 1
        else:
            final_index -= 1

    matches.reverse()
    return matches


def build_timeline_windows(
    matches: list[CaptionMatch],
    *,
    pre_pad: float,
    post_pad: float,
    merge_gap: float,
    min_score: float,
) -> list[tuple[float, float, list[CaptionMatch]]]:
    windows: list[tuple[float, float, list[CaptionMatch]]] = []
    for match in matches:
        if match.score < min_score:
            continue
        start = max(0.0, match.source_title.offset - pre_pad)
        end = match.source_title.end + post_pad
        if not windows:
            windows.append((start, end, [match]))
            continue

        last_start, last_end, bucket = windows[-1]
        if start <= last_end + merge_gap:
            windows[-1] = (last_start, max(last_end, end), bucket + [match])
        else:
            windows.append((start, end, [match]))
    return windows


def map_timeline_window_to_source(
    window_start: float,
    window_end: float,
    source_clips: list[TimelineClip],
) -> list[SourceRange]:
    mapped: list[SourceRange] = []
    for clip in source_clips:
        overlap_start = max(window_start, clip.offset)
        overlap_end = min(window_end, clip.end)
        if overlap_end <= overlap_start:
            continue

        source_start = clip.start + (overlap_start - clip.offset)
        source_end = clip.start + (overlap_end - clip.offset)
        mapped.append(
            SourceRange(
                clip_index=clip.index,
                take_select_start=overlap_start,
                take_select_end=overlap_end,
                source_start=source_start,
                source_end=source_end,
            )
        )
    return mapped


def build_candidate_fcpxml(
    source_tree: ET.ElementTree,
    source_sequence: ET.Element,
    mapped_ranges: list[SourceRange],
    project_name: str,
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
    for mapped_range in mapped_ranges:
        duration = mapped_range.duration
        ET.SubElement(
            spine_node,
            "asset-clip",
            {
                "offset": format_time(offset),
                "duration": format_time(duration),
                "tcFormat": "NDF",
                "start": format_time(mapped_range.source_start),
                "name": "IMG_7969",
                "ref": "r2",
            },
        )
        offset += duration
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


def write_report(
    report_path: Path,
    matches: list[CaptionMatch],
    windows: list[tuple[float, float, list[CaptionMatch]]],
    mapped_ranges: list[SourceRange],
    min_score: float,
) -> None:
    payload = {
        "match_count": len(matches),
        "min_score_for_candidate": min_score,
        "matches": [
            {
                "final_title_index": match.final_title.index,
                "final_text": match.final_title.text,
                "final_offset_seconds": round(match.final_title.offset, 3),
                "final_duration_seconds": round(match.final_title.duration, 3),
                "source_title_index": match.source_title.index,
                "source_text": match.source_title.text,
                "source_offset_seconds": round(match.source_title.offset, 3),
                "source_duration_seconds": round(match.source_title.duration, 3),
                "score": round(match.score, 3),
            }
            for match in matches
        ],
        "merged_windows": [
            {
                "window_index": index,
                "take_select_start_seconds": round(start, 3),
                "take_select_end_seconds": round(end, 3),
                "duration_seconds": round(end - start, 3),
                "final_title_indices": [match.final_title.index for match in bucket],
                "source_title_indices": [match.source_title.index for match in bucket],
            }
            for index, (start, end, bucket) in enumerate(windows, start=1)
        ],
        "mapped_source_ranges": [
            {
                "range_index": index,
                "take_index": mapped_range.clip_index,
                "take_select_start_seconds": round(mapped_range.take_select_start, 3),
                "take_select_end_seconds": round(mapped_range.take_select_end, 3),
                "source_start_seconds": round(mapped_range.source_start, 3),
                "source_end_seconds": round(mapped_range.source_end, 3),
                "duration_seconds": round(mapped_range.duration, 3),
            }
            for index, mapped_range in enumerate(mapped_ranges, start=1)
        ],
    }
    report_path.write_text(json.dumps(payload, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Align caption chunks from a short flattened export back onto a longer "
            "captioned take-select timeline and emit a candidate reconstruction."
        )
    )
    parser.add_argument("source_take_fcpxml", type=Path, help="Take-select FCPXML with asset clips.")
    parser.add_argument("source_caption_fcpxml", type=Path, help="Captioned FCPXML for the take-select.")
    parser.add_argument("final_caption_fcpxml", type=Path, help="Captioned FCPXML for the flattened short export.")
    parser.add_argument("--report-out", type=Path, required=True, help="Path for the JSON alignment report.")
    parser.add_argument("--candidate-out", type=Path, required=True, help="Path for the candidate FCPXML.")
    parser.add_argument("--pre-pad", type=float, default=0.12, help="Seconds of padding before each matched title.")
    parser.add_argument("--post-pad", type=float, default=0.18, help="Seconds of padding after each matched title.")
    parser.add_argument("--merge-gap", type=float, default=0.35, help="Merge adjacent matched windows separated by this many seconds or less.")
    parser.add_argument("--min-score", type=float, default=0.0, help="Only use matched captions at or above this similarity score when building the candidate.")
    args = parser.parse_args()

    source_tree, source_sequence, source_clips = load_sequence_and_clips(args.source_take_fcpxml)
    source_titles = load_titles(args.source_caption_fcpxml)
    final_titles = load_titles(args.final_caption_fcpxml)

    matches = align_titles(final_titles, source_titles)
    windows = build_timeline_windows(
        matches,
        pre_pad=args.pre_pad,
        post_pad=args.post_pad,
        merge_gap=args.merge_gap,
        min_score=args.min_score,
    )

    mapped_ranges: list[SourceRange] = []
    for window_start, window_end, _ in windows:
        mapped_ranges.extend(map_timeline_window_to_source(window_start, window_end, source_clips))

    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.candidate_out.parent.mkdir(parents=True, exist_ok=True)
    write_report(args.report_out, matches, windows, mapped_ranges, args.min_score)

    candidate_tree = build_candidate_fcpxml(
        source_tree=source_tree,
        source_sequence=source_sequence,
        mapped_ranges=mapped_ranges,
        project_name=f"{args.source_take_fcpxml.stem}_caption_aligned_candidate",
    )
    indent_xml(candidate_tree.getroot())
    candidate_tree.write(args.candidate_out, encoding="utf-8", xml_declaration=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
