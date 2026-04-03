from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import unquote, urlparse

TIMEBASE = 30000
FRAME_TICKS = 1001
DEFAULT_WORDS_PER_POP = 3
DEFAULT_MAX_GAP_SECONDS = 0.55
DEFAULT_MODEL = "large-v3"
DEFAULT_EFFECT_ID = "r-custom-title"
DEFAULT_GAP_NAME = "Titles Gap"
DEFAULT_CAPTION_MODE = "classic"
DEFAULT_VARIATION_RATE = 0.28
DEFAULT_CAPTION_SEED = 17
DEFAULT_HOOK_WINDOW_SECONDS = 3.0
DEFAULT_CLASSIC_CAPTION_LEAD_SECONDS = 0.0
DEFAULT_VARIABLE_CAPTION_LEAD_SECONDS = 0.02
DEFAULT_CAPTION_CLEARANCE_SECONDS = 0.0
DEFAULT_FACE_SAMPLE_SECONDS = 0.8
USER_SAFE_POSITION_X = 150.0
USER_SAFE_POSITION_Y = 375.0

CUSTOM_TITLE_UID = ".../Titles.localized/Build In:Out.localized/Custom.localized/Custom.moti"
FONT = "Helvetica"
FONT_COLOR = "1 0.546278 0 1"
FOCUS_FONT_COLOR = "1 0.921569 0.298039 1"
HOOK_FONT_COLOR = "1 1 1 1"
POSITION = "0 18.3883"
SCREEN_WIDTH = 1080
SCREEN_HEIGHT = 1920
MARGIN = 100
MAX_FONT = 78
MIN_FONT = 36
CHAR_WIDTH_RATIO = 0.58
GAP_START_TICKS = 107892 * 1001

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "but",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "here",
    "hers",
    "him",
    "his",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "ours",
    "she",
    "so",
    "than",
    "that",
    "the",
    "their",
    "them",
    "there",
    "these",
    "they",
    "this",
    "those",
    "to",
    "up",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
    "you",
    "your",
    "yours",
}
FILLER_WORDS = {
    "actually",
    "basically",
    "just",
    "kind",
    "like",
    "literally",
    "maybe",
    "really",
    "right",
    "sort",
    "stuff",
    "thing",
    "things",
    "uh",
    "um",
    "well",
    "yeah",
    "youknow",
}
HOOK_WORDS = {
    "best",
    "breakdown",
    "easy",
    "fast",
    "fix",
    "hack",
    "hacks",
    "hard",
    "mistake",
    "mistakes",
    "never",
    "nobody",
    "proof",
    "reason",
    "reasons",
    "results",
    "secret",
    "secrets",
    "simple",
    "stop",
    "truth",
    "viral",
    "wrong",
}

MEDIA_TAGS = {"asset-clip", "video"}
TIMELINE_TAGS = {
    "asset-clip",
    "audio",
    "audition",
    "clip",
    "gap",
    "mc-clip",
    "ref-clip",
    "spine",
    "sync-clip",
    "video",
}


@dataclass(frozen=True)
class AssetResource:
    resource_id: str
    path: Path
    start_ticks: int
    duration_ticks: int


@dataclass(frozen=True)
class ClipUse:
    asset_id: str
    media_path: Path
    sequence_offset_ticks: int
    source_in_ticks: int
    duration_ticks: int
    tag: str


@dataclass(frozen=True)
class TimedWord:
    text: str
    start_ticks: int
    end_ticks: int


@dataclass(frozen=True)
class TitleChunk:
    text: str
    start_ticks: int
    duration_ticks: int
    style_name: str = "classic"
    font_color: str = FONT_COLOR
    font_scale: float = 1.0
    position: str = POSITION
    alignment: str = "center"


@dataclass(frozen=True)
class FaceObservation:
    time_ticks: int
    center_x: float
    center_y: float
    width: float
    height: float


@dataclass(frozen=True)
class LayoutContext:
    width: int
    height: int
    faces: list[FaceObservation]


@dataclass(frozen=True)
class FrameBounds:
    width: int
    height: int


def dynamic_font_size(text: str, scale: float = 1.0) -> int:
    usable = SCREEN_WIDTH - (2 * MARGIN)
    lines = [clean_display_text(line) for line in text.splitlines()] or [text]
    chars = max((len(line) for line in lines if line), default=0)
    if chars == 0:
        return MAX_FONT
    size = usable / (chars * CHAR_WIDTH_RATIO)
    return int(max(MIN_FONT, min(MAX_FONT, size * scale)))


def parse_ticks(time_str: str | None) -> int:
    if not time_str or time_str == "0s":
        return 0
    if not time_str.endswith("s"):
        raise ValueError(f"Unsupported time format: {time_str}")

    value = time_str[:-1]
    seconds = Fraction(value)
    return round(seconds * TIMEBASE)


def fmt_ticks(ticks: int) -> str:
    return f"{ticks}/{TIMEBASE}s"


def snap_ticks(ticks: int, frame: int = FRAME_TICKS) -> int:
    return round(ticks / frame) * frame


def snap_ticks_down(ticks: int, frame: int = FRAME_TICKS) -> int:
    return (ticks // frame) * frame


def snap_ticks_up(ticks: int, frame: int = FRAME_TICKS) -> int:
    return ((ticks + frame - 1) // frame) * frame


def seconds_to_ticks(seconds: float) -> int:
    return round(seconds * TIMEBASE)


def ticks_to_seconds(ticks: int) -> float:
    return ticks / TIMEBASE


def clean_display_word(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def clean_display_text(text: str) -> str:
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_display_multiline(lines: list[str]) -> str:
    return "\n".join(clean_display_text(line) for line in lines if clean_display_text(line))


def normalize_token(text: str) -> str:
    return re.sub(r"(^[^A-Za-z0-9']+|[^A-Za-z0-9']+$)", "", text).lower()


def stable_ratio(seed: int, text: str) -> float:
    digest = hashlib.sha1(f"{seed}:{text}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(0xFFFFFFFFFFFF)


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def normalized_to_position(
    norm_x: float,
    norm_y: float,
    width: int = SCREEN_WIDTH,
    height: int = SCREEN_HEIGHT,
) -> str:
    x_pixels = (norm_x - 0.5) * width
    y_pixels = (0.5 - norm_y) * height
    x_units = x_pixels / (width / 100.0)
    y_units = y_pixels / (height / 100.0)
    return f"{x_units:.4f} {y_units:.4f}"


def clamp_position_string(position: str, bounds: FrameBounds) -> str:
    parts = position.split()
    if len(parts) != 2:
        return position

    try:
        x = float(parts[0])
        y = float(parts[1])
    except ValueError:
        return position

    safe_x_units = USER_SAFE_POSITION_X / (bounds.width / 100.0)
    safe_y_units = USER_SAFE_POSITION_Y / (bounds.height / 100.0)
    x = clamp(x, -safe_x_units, safe_x_units)
    y = clamp(y, -safe_y_units, safe_y_units)
    return f"{x:.4f} {y:.4f}"


def ends_sentence(text: str) -> bool:
    return bool(re.search(r'[.!?]["\')\]]*$', text))


def make_parent_map(root: ET.Element) -> dict[int, ET.Element]:
    parent_map: dict[int, ET.Element] = {}
    for parent in root.iter():
        for child in parent:
            parent_map[id(child)] = parent
    return parent_map


def extract_asset_src(asset: ET.Element) -> str | None:
    if asset.get("src"):
        return asset.get("src")

    for media_rep in asset.findall("media-rep"):
        src = media_rep.get("src")
        if src:
            return src
    return None


def resolve_media_path(src: str, xml_path: Path) -> Path:
    parsed = urlparse(src)
    if parsed.scheme == "file":
        if parsed.netloc and parsed.netloc != "localhost":
            path = Path(f"//{parsed.netloc}{unquote(parsed.path)}")
        else:
            path = Path(unquote(parsed.path))
    else:
        path = Path(src)
        if not path.is_absolute():
            path = xml_path.parent / path
    return path.expanduser().resolve()


def build_asset_resources(root: ET.Element, xml_path: Path) -> dict[str, AssetResource]:
    resources = root.find("resources")
    if resources is None:
        raise ValueError("FCPXML is missing a <resources> section.")

    assets: dict[str, AssetResource] = {}
    for asset in resources.findall("asset"):
        resource_id = asset.get("id")
        src = extract_asset_src(asset)
        if not resource_id or not src:
            continue

        path = resolve_media_path(src, xml_path)
        assets[resource_id] = AssetResource(
            resource_id=resource_id,
            path=path,
            start_ticks=parse_ticks(asset.get("start", "0s")),
            duration_ticks=parse_ticks(asset.get("duration", "0s")),
        )
    return assets


def get_sequence_frame_bounds(root: ET.Element, sequence: ET.Element | None) -> FrameBounds:
    if sequence is not None:
        format_id = sequence.get("format")
        if format_id:
            fmt = root.find(f".//format[@id='{format_id}']")
            if fmt is not None:
                width = int(fmt.get("width", SCREEN_WIDTH))
                height = int(fmt.get("height", SCREEN_HEIGHT))
                return FrameBounds(width=width, height=height)

    first_format = root.find(".//format")
    if first_format is not None:
        width = int(first_format.get("width", SCREEN_WIDTH))
        height = int(first_format.get("height", SCREEN_HEIGHT))
        return FrameBounds(width=width, height=height)

    return FrameBounds(width=SCREEN_WIDTH, height=SCREEN_HEIGHT)


def walk_timeline(
    node: ET.Element,
    parent_global_offset_ticks: int,
    parent_start_ticks: int,
    assets: dict[str, AssetResource],
    clips: list[ClipUse],
) -> None:
    node_offset_ticks = parse_ticks(node.get("offset", "0s"))
    node_global_offset_ticks = parent_global_offset_ticks + node_offset_ticks - parent_start_ticks
    node_start_ticks = parse_ticks(node.get("start", "0s"))
    node_duration_ticks = parse_ticks(node.get("duration", "0s"))

    ref = node.get("ref")
    if node.tag in MEDIA_TAGS and ref in assets and node_duration_ticks > 0:
        asset = assets[ref]
        source_in_ticks = node_start_ticks - asset.start_ticks
        if source_in_ticks < 0:
            source_in_ticks = 0

        clips.append(
            ClipUse(
                asset_id=asset.resource_id,
                media_path=asset.path,
                sequence_offset_ticks=node_global_offset_ticks,
                source_in_ticks=source_in_ticks,
                duration_ticks=node_duration_ticks,
                tag=node.tag,
            )
        )

    for child in node:
        if child.tag in TIMELINE_TAGS:
            walk_timeline(
                node=child,
                parent_global_offset_ticks=node_global_offset_ticks,
                parent_start_ticks=node_start_ticks,
                assets=assets,
                clips=clips,
            )


def collect_clip_uses(
    root: ET.Element,
    assets: dict[str, AssetResource],
) -> tuple[ET.Element, ET.Element, list[ClipUse]]:
    sequence = root.find(".//sequence")
    if sequence is None:
        raise ValueError("FCPXML does not contain a <sequence>.")

    spine = sequence.find("spine")
    if spine is None:
        raise ValueError("FCPXML sequence does not contain a <spine>.")

    clips: list[ClipUse] = []
    for child in spine:
        if child.tag in TIMELINE_TAGS:
            walk_timeline(
                node=child,
                parent_global_offset_ticks=0,
                parent_start_ticks=0,
                assets=assets,
                clips=clips,
            )

    clips.sort(key=lambda clip: (clip.sequence_offset_ticks, clip.source_in_ticks))
    return sequence, spine, clips


def ensure_effect_resource(root: ET.Element, effect_id: str) -> None:
    resources = root.find("resources")
    if resources is None:
        raise ValueError("FCPXML is missing a <resources> section.")

    existing_ids = {element.get("id") for element in resources if element.get("id")}
    if effect_id in existing_ids:
        return

    ET.SubElement(
        resources,
        "effect",
        {"id": effect_id, "name": "Custom", "uid": CUSTOM_TITLE_UID},
    )


def remove_existing_captions(root: ET.Element) -> int:
    parent_map = make_parent_map(root)
    removed = 0
    for caption in list(root.iter("caption")):
        parent = parent_map.get(id(caption))
        if parent is None:
            continue
        parent.remove(caption)
        removed += 1
    return removed


def remove_existing_generated_gap(spine: ET.Element, gap_name: str) -> int:
    removed = 0
    for child in list(spine):
        if child.tag == "gap" and child.get("name") == gap_name:
            spine.remove(child)
            removed += 1
    return removed


def get_sequence_duration_ticks(sequence: ET.Element, clips: list[ClipUse]) -> int:
    declared = parse_ticks(sequence.get("duration", "0s"))
    if declared > 0:
        return snap_ticks(declared)

    if not clips:
        return 0

    last_end = max(clip.sequence_offset_ticks + clip.duration_ticks for clip in clips)
    return snap_ticks(last_end)


def detect_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


def choose_backend(preferred: str) -> str:
    if preferred == "auto":
        try:
            import faster_whisper  # noqa: F401

            return "faster-whisper"
        except ImportError:
            pass

        try:
            import whisper  # noqa: F401

            return "whisper"
        except ImportError:
            pass

        raise RuntimeError(
            "No Whisper backend found. Install 'faster-whisper' or 'openai-whisper'."
        )

    return preferred


def extract_timed_words_from_segments(segments: Iterable[object]) -> list[TimedWord]:
    timed_words: list[TimedWord] = []

    for segment in segments:
        words = getattr(segment, "words", None) or []
        if words:
            for word in words:
                text = clean_display_word(getattr(word, "word", ""))
                start = getattr(word, "start", None)
                end = getattr(word, "end", None)
                if not text or start is None or end is None:
                    continue
                start_ticks = seconds_to_ticks(start)
                end_ticks = max(seconds_to_ticks(end), start_ticks + FRAME_TICKS)
                timed_words.append(TimedWord(text=text, start_ticks=start_ticks, end_ticks=end_ticks))
            continue

        text = clean_display_word(getattr(segment, "text", ""))
        start = getattr(segment, "start", None)
        end = getattr(segment, "end", None)
        if not text or start is None or end is None:
            continue

        tokens = text.split()
        if not tokens:
            continue

        seg_start = seconds_to_ticks(start)
        seg_end = max(seconds_to_ticks(end), seg_start + FRAME_TICKS)
        span = max(seg_end - seg_start, len(tokens) * FRAME_TICKS)
        step = span / len(tokens)

        for index, token in enumerate(tokens):
            token_start = seg_start + round(index * step)
            token_end = seg_start + round((index + 1) * step)
            token_end = max(token_end, token_start + FRAME_TICKS)
            timed_words.append(TimedWord(text=token, start_ticks=token_start, end_ticks=token_end))

    timed_words.sort(key=lambda word: (word.start_ticks, word.end_ticks))
    return timed_words


def distribute_segment_text(text: str, start: float, end: float) -> list[TimedWord]:
    cleaned_text = clean_display_word(text)
    if not cleaned_text:
        return []

    tokens = cleaned_text.split()
    if not tokens:
        return []

    seg_start = seconds_to_ticks(start)
    seg_end = max(seconds_to_ticks(end), seg_start + FRAME_TICKS)
    span = max(seg_end - seg_start, len(tokens) * FRAME_TICKS)
    step = span / len(tokens)

    words: list[TimedWord] = []
    for index, token in enumerate(tokens):
        token_start = seg_start + round(index * step)
        token_end = seg_start + round((index + 1) * step)
        token_end = max(token_end, token_start + FRAME_TICKS)
        words.append(TimedWord(text=token, start_ticks=token_start, end_ticks=token_end))

    return words


def build_faster_whisper_transcriber(
    model_name: str,
    device: str,
    compute_type: str | None,
    language: str | None,
) -> Callable[[Path], list[TimedWord]]:
    from faster_whisper import WhisperModel

    chosen_compute_type = compute_type or ("float16" if device == "cuda" else "int8")
    model = WhisperModel(model_name, device=device, compute_type=chosen_compute_type)

    def transcribe(media_path: Path) -> list[TimedWord]:
        segments, _ = model.transcribe(
            str(media_path),
            language=language,
            word_timestamps=True,
            vad_filter=True,
        )
        return extract_timed_words_from_segments(segments)

    return transcribe


def build_whisper_transcriber(
    model_name: str,
    device: str,
    language: str | None,
) -> Callable[[Path], list[TimedWord]]:
    import whisper

    model = whisper.load_model(model_name, device=device)

    def transcribe(media_path: Path) -> list[TimedWord]:
        result = model.transcribe(
            str(media_path),
            language=language,
            word_timestamps=True,
            verbose=False,
            fp16=device == "cuda",
        )

        timed_words: list[TimedWord] = []
        for segment in result.get("segments", []):
            words = segment.get("words") or []
            if words:
                for word in words:
                    text = clean_display_word(word.get("word", ""))
                    start = word.get("start")
                    end = word.get("end")
                    if not text or start is None or end is None:
                        continue
                    start_ticks = seconds_to_ticks(start)
                    end_ticks = max(seconds_to_ticks(end), start_ticks + FRAME_TICKS)
                    timed_words.append(
                        TimedWord(text=text, start_ticks=start_ticks, end_ticks=end_ticks)
                    )
                continue

            text = clean_display_word(segment.get("text", ""))
            start = segment.get("start")
            end = segment.get("end")
            if not text or start is None or end is None:
                continue
            timed_words.extend(distribute_segment_text(text, start, end))

        timed_words.sort(key=lambda word: (word.start_ticks, word.end_ticks))
        return timed_words

    return transcribe


def build_transcriber(
    backend: str,
    model_name: str,
    device: str,
    compute_type: str | None,
    language: str | None,
) -> Callable[[Path], list[TimedWord]]:
    if backend == "faster-whisper":
        return build_faster_whisper_transcriber(
            model_name=model_name,
            device=device,
            compute_type=compute_type,
            language=language,
        )
    if backend == "whisper":
        return build_whisper_transcriber(
            model_name=model_name,
            device=device,
            language=language,
        )
    raise ValueError(f"Unsupported backend: {backend}")


def compute_cache_path(
    cache_dir: Path,
    media_path: Path,
    backend: str,
    model_name: str,
    language: str | None,
) -> Path:
    stat = media_path.stat()
    key = "::".join(
        [
            str(media_path.resolve()),
            str(stat.st_size),
            str(stat.st_mtime_ns),
            backend,
            model_name,
            language or "auto",
        ]
    )
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.json"


def compute_clip_cache_path(
    cache_dir: Path,
    clip: ClipUse,
    backend: str,
    model_name: str,
    language: str | None,
) -> Path:
    stat = clip.media_path.stat()
    key = "::".join(
        [
            str(clip.media_path.resolve()),
            str(stat.st_size),
            str(stat.st_mtime_ns),
            str(clip.source_in_ticks),
            str(clip.duration_ticks),
            backend,
            model_name,
            language or "auto",
        ]
    )
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return cache_dir / "clips" / f"{digest}.json"


def compute_clip_audio_path(cache_dir: Path, clip: ClipUse) -> Path:
    stat = clip.media_path.stat()
    key = "::".join(
        [
            str(clip.media_path.resolve()),
            str(stat.st_size),
            str(stat.st_mtime_ns),
            str(clip.source_in_ticks),
            str(clip.duration_ticks),
        ]
    )
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return cache_dir / "_segments" / f"{digest}.wav"


def load_cached_words(cache_path: Path) -> list[TimedWord] | None:
    if not cache_path.exists():
        return None

    payload = json.loads(cache_path.read_text())
    return [TimedWord(**item) for item in payload.get("words", [])]


def save_cached_words(cache_path: Path, words: list[TimedWord]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"words": [asdict(word) for word in words]}
    cache_path.write_text(json.dumps(payload, indent=2))


def detect_layout_context(media_path: Path) -> LayoutContext:
    try:
        import cv2
    except ImportError:
        return LayoutContext(width=SCREEN_WIDTH, height=SCREEN_HEIGHT, faces=[])

    capture = cv2.VideoCapture(str(media_path))
    if not capture.isOpened():
        return LayoutContext(width=SCREEN_WIDTH, height=SCREEN_HEIGHT, faces=[])

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or SCREEN_WIDTH)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or SCREEN_HEIGHT)
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    duration_seconds = frame_count / fps if frame_count > 0 and fps > 0 else 0.0

    classifier = cv2.CascadeClassifier(
        str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
    )
    faces: list[FaceObservation] = []

    sample_time = 0.0
    max_samples = 180
    samples = 0

    while sample_time <= duration_seconds + 0.01 and samples < max_samples:
        capture.set(cv2.CAP_PROP_POS_MSEC, sample_time * 1000.0)
        success, frame = capture.read()
        if success and frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected = classifier.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(max(60, width // 10), max(60, height // 10)),
            )
            if len(detected) > 0:
                x, y, w, h = max(detected, key=lambda box: box[2] * box[3])
                faces.append(
                    FaceObservation(
                        time_ticks=seconds_to_ticks(sample_time),
                        center_x=(x + (w / 2)) / width,
                        center_y=(y + (h / 2)) / height,
                        width=w / width,
                        height=h / height,
                    )
                )
        sample_time += DEFAULT_FACE_SAMPLE_SECONDS
        samples += 1

    capture.release()
    return LayoutContext(width=width, height=height, faces=faces)


def nearest_face(context: LayoutContext | None, time_ticks: int) -> FaceObservation | None:
    if context is None or not context.faces:
        return None
    return min(context.faces, key=lambda face: abs(face.time_ticks - time_ticks))


def transcribe_media(
    media_path: Path,
    transcriber: Callable[[Path], list[TimedWord]],
    backend: str,
    model_name: str,
    language: str | None,
    cache_dir: Path | None,
) -> list[TimedWord]:
    cache_path = None
    if cache_dir is not None:
        cache_path = compute_cache_path(cache_dir, media_path, backend, model_name, language)
        cached = load_cached_words(cache_path)
        if cached is not None:
            print(f"Using cached transcript for {media_path.name}")
            return cached

    print(f"Transcribing {media_path.name} with {backend} ({model_name})")
    words = transcriber(media_path)

    if cache_path is not None:
        save_cached_words(cache_path, words)
    return words


def extract_clip_audio(
    clip: ClipUse,
    destination: Path,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination

    start_seconds = ticks_to_seconds(clip.source_in_ticks)
    duration_seconds = max(ticks_to_seconds(clip.duration_ticks), ticks_to_seconds(FRAME_TICKS))
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-ss",
        f"{start_seconds:.6f}",
        "-t",
        f"{duration_seconds:.6f}",
        "-i",
        str(clip.media_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(destination),
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required to extract timeline clip audio.") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else "unknown ffmpeg error"
        raise RuntimeError(f"ffmpeg failed while extracting {clip.media_path.name}: {stderr}") from exc

    return destination


def transcribe_clip_use(
    clip: ClipUse,
    transcriber: Callable[[Path], list[TimedWord]],
    backend: str,
    model_name: str,
    language: str | None,
    cache_dir: Path | None,
) -> list[TimedWord]:
    cache_path = None
    if cache_dir is not None:
        cache_path = compute_clip_cache_path(cache_dir, clip, backend, model_name, language)
        cached = load_cached_words(cache_path)
        if cached is not None:
            return cached

    segment_path = compute_clip_audio_path(cache_dir or Path("/tmp"), clip)
    extract_clip_audio(clip, segment_path)
    words = transcriber(segment_path)

    clip_words: list[TimedWord] = []
    for word in words:
        local_start = max(0, min(word.start_ticks, max(clip.duration_ticks - FRAME_TICKS, 0)))
        local_end = min(max(word.end_ticks, local_start + FRAME_TICKS), clip.duration_ticks)
        clip_words.append(
            TimedWord(
                text=word.text,
                start_ticks=clip.sequence_offset_ticks + local_start,
                end_ticks=clip.sequence_offset_ticks + max(local_end, local_start + FRAME_TICKS),
            )
        )

    if cache_path is not None:
        save_cached_words(cache_path, clip_words)
    return clip_words


def group_words(words: list[TimedWord], max_words: int, max_gap_ticks: int) -> list[list[TimedWord]]:
    groups: list[list[TimedWord]] = []
    current_words: list[TimedWord] = []

    for word in words:
        if current_words and (
            word.start_ticks - current_words[-1].end_ticks > max_gap_ticks
            or len(current_words) >= max_words
        ):
            groups.append(current_words)
            current_words = []

        current_words.append(word)

        if len(current_words) >= max_words or ends_sentence(word.text):
            groups.append(current_words)
            current_words = []

    if current_words:
        groups.append(current_words)

    return groups


def build_chunk(words: list[TimedWord]) -> TitleChunk:
    text = clean_display_text(" ".join(word.text for word in words)).upper()
    start_ticks = words[0].start_ticks
    end_ticks = max(words[-1].end_ticks, start_ticks + FRAME_TICKS)
    duration_ticks = max(end_ticks - start_ticks, FRAME_TICKS)
    return TitleChunk(text=text, start_ticks=start_ticks, duration_ticks=duration_ticks)


def chunk_end_ticks(chunk: TitleChunk) -> int:
    return chunk.start_ticks + chunk.duration_ticks


def flatten_same_start_runs(
    chunks: list[TitleChunk],
    clearance_ticks: int,
) -> list[TitleChunk]:
    if len(chunks) < 2:
        return chunks

    flattened: list[TitleChunk] = []
    index = 0

    while index < len(chunks):
        run_start_ticks = chunks[index].start_ticks
        run: list[TitleChunk] = [chunks[index]]
        index += 1

        while index < len(chunks) and chunks[index].start_ticks == run_start_ticks:
            run.append(chunks[index])
            index += 1

        if len(run) == 1:
            flattened.extend(run)
            continue

        next_start_ticks = chunks[index].start_ticks if index < len(chunks) else None
        run_end_ticks = max(chunk_end_ticks(chunk) for chunk in run)
        if next_start_ticks is not None:
            run_end_ticks = max(next_start_ticks, run_start_ticks + (len(run) * FRAME_TICKS))

        gap_ticks = 0
        if len(run) > 1:
            max_gap_budget = max(
                0,
                run_end_ticks - run_start_ticks - (len(run) * FRAME_TICKS),
            )
            gap_ticks = min(clearance_ticks, max_gap_budget // (len(run) - 1))

        cursor_ticks = run_start_ticks
        for run_index, chunk in enumerate(run):
            remaining_chunks = len(run) - run_index - 1
            latest_end_ticks = run_end_ticks - (
                (remaining_chunks * FRAME_TICKS) + (remaining_chunks * gap_ticks)
            )
            desired_end_ticks = cursor_ticks + chunk.duration_ticks
            resolved_end_ticks = min(
                max(cursor_ticks + FRAME_TICKS, desired_end_ticks),
                latest_end_ticks,
            )
            flattened.append(
                replace(
                    chunk,
                    start_ticks=cursor_ticks,
                    duration_ticks=max(resolved_end_ticks - cursor_ticks, FRAME_TICKS),
                )
            )
            cursor_ticks = resolved_end_ticks + gap_ticks

    return flattened


def shift_overlapping_chunk_starts(chunks: list[TitleChunk]) -> list[TitleChunk]:
    if len(chunks) < 2:
        return chunks

    shifted: list[TitleChunk] = []
    previous_end_ticks = 0

    for chunk in chunks:
        resolved_start_ticks = max(chunk.start_ticks, previous_end_ticks)
        resolved_end_ticks = max(chunk_end_ticks(chunk), resolved_start_ticks + FRAME_TICKS)
        shifted.append(
            replace(
                chunk,
                start_ticks=resolved_start_ticks,
                duration_ticks=max(resolved_end_ticks - resolved_start_ticks, FRAME_TICKS),
            )
        )
        previous_end_ticks = resolved_end_ticks

    return shifted


def trim_chunk_ends_to_clearance(
    chunks: list[TitleChunk],
    clearance_ticks: int,
) -> list[TitleChunk]:
    if len(chunks) < 2:
        return chunks

    normalized: list[TitleChunk] = []

    for index, chunk in enumerate(chunks):
        next_start_ticks = None
        for next_chunk in chunks[index + 1 :]:
            if next_chunk.start_ticks > chunk.start_ticks:
                next_start_ticks = next_chunk.start_ticks
                break

        adjusted_end_ticks = chunk_end_ticks(chunk)
        if next_start_ticks is not None:
            available_clearance_ticks = max(
                0,
                next_start_ticks - (chunk.start_ticks + FRAME_TICKS),
            )
            applied_clearance_ticks = min(clearance_ticks, available_clearance_ticks)
            adjusted_end_ticks = max(
                chunk.start_ticks + FRAME_TICKS,
                next_start_ticks - applied_clearance_ticks,
            )

        normalized.append(
            replace(
                chunk,
                duration_ticks=max(adjusted_end_ticks - chunk.start_ticks, FRAME_TICKS),
            )
        )

    return normalized


def normalize_caption_chunks(
    chunks: list[TitleChunk],
    clearance_ticks: int,
) -> list[TitleChunk]:
    if len(chunks) < 2:
        return chunks

    ordered = [chunk for _, chunk in sorted(enumerate(chunks), key=lambda item: (item[1].start_ticks, item[0]))]
    ordered = flatten_same_start_runs(ordered, clearance_ticks)
    ordered = trim_chunk_ends_to_clearance(ordered, clearance_ticks)
    ordered = shift_overlapping_chunk_starts(ordered)
    return trim_chunk_ends_to_clearance(ordered, clearance_ticks)


def score_words(words: list[TimedWord]) -> dict[str, float]:
    normalized_tokens = [normalize_token(word.text) for word in words]
    counts = Counter(
        token
        for token in normalized_tokens
        if token and token not in STOP_WORDS and token not in FILLER_WORDS
    )
    scores: dict[str, float] = {}

    for index, token in enumerate(normalized_tokens):
        if not token or token in STOP_WORDS or token in FILLER_WORDS:
            continue

        score = 1.0
        if token in HOOK_WORDS:
            score += 1.2
        if any(char.isdigit() for char in token):
            score += 1.1
        if len(token) >= 7:
            score += 0.35

        frequency = counts.get(token, 1)
        if frequency == 1:
            score += 0.45
        elif frequency <= 3:
            score += 0.2

        if index < 18:
            score += 0.25

        scores[token] = max(scores.get(token, 0.0), score)

    return scores


def split_words_for_lines(words: list[str], pivot_index: int, style_name: str) -> list[str]:
    if len(words) <= 1:
        return words

    if style_name in {"stacked", "hero", "side"}:
        split_at = 1 if len(words) == 2 else max(1, min(len(words) - 1, (len(words) + 1) // 2))
        return [" ".join(words[:split_at]), " ".join(words[split_at:])]

    if style_name in {"focus", "hook", "burst"}:
        if len(words) == 2:
            return [words[0], words[1]]
        if len(words) == 3 and pivot_index == 1:
            return [words[0], words[1], words[2]]
        if pivot_index <= 0:
            return [words[0], " ".join(words[1:])]
        if pivot_index >= len(words) - 1:
            return [" ".join(words[:-1]), words[-1]]

        before = " ".join(words[:pivot_index])
        pivot = words[pivot_index]
        after = " ".join(words[pivot_index + 1 :])
        lines = []
        if before:
            lines.append(before)
        lines.append(pivot)
        if after:
            lines.append(after)
        return lines

    return [" ".join(words)]


def build_title_chunk(
    text: str,
    words: list[TimedWord],
    style_name: str,
    font_scale: float,
    font_color: str,
    position: str,
    alignment: str = "center",
    start_ticks: int | None = None,
    end_ticks: int | None = None,
) -> TitleChunk:
    resolved_start = words[0].start_ticks if start_ticks is None else start_ticks
    resolved_end = max(words[-1].end_ticks, resolved_start + FRAME_TICKS) if end_ticks is None else max(end_ticks, resolved_start + FRAME_TICKS)
    duration_ticks = max(resolved_end - resolved_start, FRAME_TICKS)
    normalized_text = (
        clean_display_multiline(text.splitlines())
        if "\n" in text
        else clean_display_text(text)
    )
    return TitleChunk(
        text=normalized_text,
        start_ticks=resolved_start,
        duration_ticks=duration_ticks,
        style_name=style_name,
        font_color=font_color,
        font_scale=font_scale,
        position=position,
        alignment=alignment,
    )


def choose_template(
    words: list[TimedWord],
    keyword_scores: dict[str, float],
    caption_mode: str,
    variation_rate: float,
    hook_window_ticks: int,
    style_seed: int,
) -> tuple[str, int]:
    normalized = [normalize_token(word.text) for word in words]
    scored = [
        (index, keyword_scores.get(token, 0.0), token)
        for index, token in enumerate(normalized)
        if token
    ]

    pivot_index = 0
    pivot_score = 0.0
    if scored:
        pivot_index, pivot_score, _ = max(scored, key=lambda item: item[1])

    if caption_mode == "classic":
        return "classic", pivot_index

    key = f"{words[0].start_ticks}:{' '.join(word.text for word in words)}"
    roll = stable_ratio(style_seed, key)
    early_hook = words[0].start_ticks <= hook_window_ticks
    has_hook_word = any(
        token in HOOK_WORDS or any(char.isdigit() for char in token)
        for token in normalized
        if token
    )
    short_phrase = len([token for token in normalized if token]) <= 3
    strong_pivot = pivot_score >= 1.9
    very_strong_pivot = pivot_score >= 2.4 or has_hook_word

    if early_hook and very_strong_pivot and short_phrase and roll < 0.82:
        return "burst", pivot_index

    if early_hook and (strong_pivot or has_hook_word) and roll < 0.88:
        return "hero", pivot_index

    if short_phrase and strong_pivot and roll < min(0.72, variation_rate + 0.18):
        return "orbit", pivot_index

    if strong_pivot and roll < min(0.76, variation_rate + 0.14):
        return "side", pivot_index

    if roll < variation_rate * 0.48:
        return "hero", pivot_index

    return "classic", pivot_index


def base_face_anchors(face: FaceObservation | None) -> dict[str, tuple[float, float]]:
    if face is None:
        return {
            "top_center": (0.50, 0.18),
            "left_mid": (0.24, 0.42),
            "right_mid": (0.76, 0.42),
            "left_low": (0.26, 0.64),
            "right_low": (0.74, 0.64),
            "center_mid": (0.50, 0.38),
        }

    top_y = clamp(face.center_y - (face.height * 1.45), 0.14, 0.28)
    mid_y = clamp(face.center_y, 0.30, 0.56)
    low_y = clamp(face.center_y + (face.height * 1.05), 0.58, 0.72)
    left_x = clamp(face.center_x - max(0.24, face.width * 1.35), 0.18, 0.34)
    right_x = clamp(face.center_x + max(0.24, face.width * 1.35), 0.66, 0.82)
    center_x = clamp(face.center_x, 0.42, 0.58)

    return {
        "top_center": (0.50, top_y),
        "left_mid": (left_x, mid_y),
        "right_mid": (right_x, mid_y),
        "left_low": (left_x, low_y),
        "right_low": (right_x, low_y),
        "center_mid": (center_x, clamp(face.center_y - (face.height * 0.25), 0.26, 0.48)),
    }


def choose_side_anchor(face: FaceObservation | None, anchors: dict[str, tuple[float, float]], style_seed: int, words: list[TimedWord]) -> tuple[float, float]:
    if face is not None:
        if face.center_x <= 0.50:
            return anchors["right_mid"]
        return anchors["left_mid"]

    roll = stable_ratio(style_seed + 11, f"{words[0].start_ticks}:{words[0].text}")
    return anchors["left_mid"] if roll < 0.5 else anchors["right_mid"]


def build_progressive_stack_chunks(
    words: list[TimedWord],
    style_name: str,
    position: str,
    font_scale: float,
    font_color: str,
    alignment: str = "center",
) -> list[TitleChunk]:
    chunks: list[TitleChunk] = []
    visible_words: list[str] = []

    for index, word in enumerate(words):
        token = clean_display_word(word.text).upper()
        if not token:
            continue
        visible_words.append(token)
        start_ticks = word.start_ticks
        if index + 1 < len(words):
            end_ticks = max(words[index + 1].start_ticks, start_ticks + FRAME_TICKS)
        else:
            end_ticks = max(words[-1].end_ticks, start_ticks + FRAME_TICKS)

        chunks.append(
            build_title_chunk(
                text="\n".join(visible_words),
                words=words,
                style_name=style_name,
                font_scale=font_scale,
                font_color=font_color,
                position=position,
                alignment=alignment,
                start_ticks=start_ticks,
                end_ticks=end_ticks,
            )
        )

    return chunks


def build_styled_chunks(
    words: list[TimedWord],
    style_name: str,
    pivot_index: int,
    layout_context: LayoutContext | None,
    media_time_ticks: int,
    style_seed: int,
) -> list[TitleChunk]:
    raw_words = [clean_display_word(word.text) for word in words if clean_display_word(word.text)]
    if not raw_words:
        return []

    face = nearest_face(layout_context, media_time_ticks)
    layout_width = layout_context.width if layout_context is not None else SCREEN_WIDTH
    layout_height = layout_context.height if layout_context is not None else SCREEN_HEIGHT
    anchors = base_face_anchors(face)

    if style_name == "classic":
        text = clean_display_text(" ".join(word.upper() for word in raw_words))
        return [
            build_title_chunk(
                text=text,
                words=words,
                style_name="classic",
                font_scale=1.0,
                font_color=FONT_COLOR,
                position=POSITION,
            )
        ]

    if style_name == "hero":
        return build_progressive_stack_chunks(
            words=words,
            style_name="hero",
            position=normalized_to_position(*anchors["top_center"], width=layout_width, height=layout_height),
            font_scale=1.20,
            font_color=HOOK_FONT_COLOR,
        )

    if style_name == "side":
        side_anchor = choose_side_anchor(face, anchors, style_seed, words)
        return build_progressive_stack_chunks(
            words=words,
            style_name="side",
            position=normalized_to_position(*side_anchor, width=layout_width, height=layout_height),
            font_scale=1.08,
            font_color=HOOK_FONT_COLOR,
        )

    if style_name == "orbit":
        orbit_words = raw_words[:3]
        pivot_word = raw_words[min(pivot_index, len(raw_words) - 1)]
        side_anchor = choose_side_anchor(face, anchors, style_seed, words)
        if side_anchor == anchors["left_mid"]:
            orbit_points = [anchors["top_center"], anchors["left_mid"], anchors["left_low"]]
        else:
            orbit_points = [anchors["top_center"], anchors["right_mid"], anchors["right_low"]]

        chunks: list[TitleChunk] = []
        for index, word_text in enumerate(orbit_words):
            scale = 1.28 if word_text == pivot_word else (0.98 if index == 0 else 1.06)
            color = FOCUS_FONT_COLOR if word_text == pivot_word else HOOK_FONT_COLOR
            source_word = words[min(index, len(words) - 1)]
            chunks.append(
                build_title_chunk(
                    text=word_text.lower(),
                    words=[source_word],
                    style_name="orbit",
                    font_scale=scale,
                    font_color=color,
                    position=normalized_to_position(
                        *orbit_points[min(index, len(orbit_points) - 1)],
                        width=layout_width,
                        height=layout_height,
                    ),
                    start_ticks=source_word.start_ticks,
                    end_ticks=source_word.end_ticks,
                )
            )
        return chunks

    if style_name == "burst":
        pivot_word = raw_words[min(pivot_index, len(raw_words) - 1)].lower()
        support_words = [word.lower() for idx, word in enumerate(raw_words) if idx != pivot_index]
        support_text = clean_display_multiline(split_words_for_lines(support_words or [pivot_word], min(pivot_index, max(len(support_words) - 1, 0)), "burst"))
        side_anchor = choose_side_anchor(face, anchors, style_seed + 23, words)
        primary_anchor = anchors["center_mid"] if face is not None else anchors["top_center"]
        chunks = []
        if support_text:
            chunks.append(
                build_title_chunk(
                    text=support_text,
                    words=words,
                    style_name="burst",
                    font_scale=0.88,
                    font_color=HOOK_FONT_COLOR,
                    position=normalized_to_position(*side_anchor, width=layout_width, height=layout_height),
                    start_ticks=words[0].start_ticks,
                    end_ticks=max(words[min(1, len(words) - 1)].end_ticks, words[0].start_ticks + FRAME_TICKS),
                )
            )
        chunks.append(
            build_title_chunk(
                text=pivot_word,
                words=[words[min(pivot_index, len(words) - 1)]],
                style_name="burst",
                font_scale=1.42,
                font_color=FOCUS_FONT_COLOR,
                position=normalized_to_position(*primary_anchor, width=layout_width, height=layout_height),
                start_ticks=words[min(pivot_index, len(words) - 1)].start_ticks,
                end_ticks=max(words[-1].end_ticks, words[min(pivot_index, len(words) - 1)].start_ticks + FRAME_TICKS),
            )
        )
        return chunks

    return [
        build_title_chunk(
            text=clean_display_text(" ".join(raw_words)),
            words=words,
            style_name="classic",
            font_scale=1.0,
            font_color=FONT_COLOR,
            position=POSITION,
        )
    ]


def chunk_words(
    words: list[TimedWord],
    clip: ClipUse,
    max_words: int,
    max_gap_ticks: int,
    caption_mode: str,
    variation_rate: float,
    hook_window_ticks: int,
    style_seed: int,
    keyword_scores: dict[str, float],
    layout_context: LayoutContext | None,
) -> list[TitleChunk]:
    groups = group_words(words, max_words, max_gap_ticks)
    chunks: list[TitleChunk] = []

    for group in groups:
        style_name, pivot_index = choose_template(
            words=group,
            keyword_scores=keyword_scores,
            caption_mode=caption_mode,
            variation_rate=variation_rate,
            hook_window_ticks=hook_window_ticks,
            style_seed=style_seed,
        )
        media_time_ticks = clip.source_in_ticks + (group[0].start_ticks - clip.sequence_offset_ticks)
        chunks.extend(
            build_styled_chunks(
                group,
                style_name,
                pivot_index,
                layout_context,
                media_time_ticks,
                style_seed,
            )
        )

    return chunks


def map_words_into_clip(clip: ClipUse, asset_words: list[TimedWord]) -> list[TimedWord]:
    clip_start = clip.source_in_ticks
    clip_end = clip.source_in_ticks + clip.duration_ticks
    mapped_words: list[TimedWord] = []

    for word in asset_words:
        if word.end_ticks <= clip_start:
            continue
        if word.start_ticks >= clip_end:
            break

        local_start = max(word.start_ticks, clip_start) - clip_start
        local_end = min(word.end_ticks, clip_end) - clip_start

        mapped_start = clip.sequence_offset_ticks + local_start
        mapped_end = clip.sequence_offset_ticks + max(local_end, local_start + FRAME_TICKS)
        text = clean_display_word(word.text)
        if not text:
            continue

        mapped_words.append(TimedWord(text=text, start_ticks=mapped_start, end_ticks=mapped_end))

    return mapped_words


def add_title_to_gap(
    gap: ET.Element,
    chunk: TitleChunk,
    effect_id: str,
    ts_counter: int,
    caption_lead_ticks: int,
    frame_bounds: FrameBounds,
) -> int:
    title_start = snap_ticks_up(max(0, chunk.start_ticks - caption_lead_ticks))
    title_end = max(
        snap_ticks_down(chunk.start_ticks + chunk.duration_ticks),
        title_start + FRAME_TICKS,
    )
    title_offset = title_start + GAP_START_TICKS
    title_duration = max(title_end - title_start, FRAME_TICKS)
    ts_id = f"ts{ts_counter}"

    title = ET.SubElement(
        gap,
        "title",
        {
            "ref": effect_id,
            "lane": "1",
            "offset": fmt_ticks(title_offset),
            "name": f"Pop: {chunk.text[:60]}",
            "start": fmt_ticks(GAP_START_TICKS),
            "duration": fmt_ticks(title_duration),
        },
    )

    alignment_value = {
        "left": "0 (Left)",
        "center": "1 (Center)",
        "right": "2 (Right)",
    }.get(chunk.alignment, "1 (Center)")

    ET.SubElement(
        title,
        "param",
        {
            "name": "Alignment",
            "key": "9999/10199/10201/2/354/1002961760/401",
            "value": alignment_value,
        },
    )
    ET.SubElement(
        title,
        "param",
        {
            "name": "Out Sequencing",
            "key": "9999/10199/10201/4/10233/201/202",
            "value": "0 (To)",
        },
    )
    ET.SubElement(
        title,
        "param",
        {
            "name": "Color",
            "key": "9999/10199/10201/5/10203/14/16",
            "value": "1.06026 0.617676 -0.243766",
        },
    )
    ET.SubElement(
        title,
        "param",
        {
            "name": "Wrap Mode",
            "key": "9999/10199/10201/5/10203/21/25/5",
            "value": "1 (Repeat)",
        },
    )
    ET.SubElement(
        title,
        "param",
        {
            "name": "Font",
            "key": "9999/10199/10201/5/10203/83",
            "value": "99 4",
        },
    )

    text_tag = ET.SubElement(title, "text")
    ts_ref = ET.SubElement(text_tag, "text-style", {"ref": ts_id})
    ts_ref.text = chunk.text

    ts_def = ET.SubElement(title, "text-style-def", {"id": ts_id})
    ET.SubElement(
        ts_def,
        "text-style",
        {
            "font": FONT,
            "fontSize": str(dynamic_font_size(chunk.text, chunk.font_scale)),
            "fontColor": chunk.font_color,
            "bold": "1",
            "alignment": chunk.alignment,
        },
    )
    ET.SubElement(
        title,
        "adjust-transform",
        {"position": clamp_position_string(chunk.position, frame_bounds)},
    )
    return ts_counter + 1


def build_titles_gap(
    root: ET.Element,
    sequence: ET.Element,
    mapped_words_by_clip: list[tuple[ClipUse, list[TimedWord]]],
    sequence_duration_ticks: int,
    effect_id: str,
    gap_name: str,
    max_words_per_pop: int,
    max_gap_seconds: float,
    caption_mode: str,
    variation_rate: float,
    hook_window_seconds: float,
    style_seed: int,
    caption_lead_seconds: float,
    caption_clearance_seconds: float,
) -> tuple[ET.Element, int, Counter[str]]:
    gap = ET.Element(
        "gap",
        {
            "name": gap_name,
            "offset": "0s",
            "start": fmt_ticks(GAP_START_TICKS),
            "duration": fmt_ticks(sequence_duration_ticks),
        },
    )

    max_gap_ticks = seconds_to_ticks(max_gap_seconds)
    hook_window_ticks = seconds_to_ticks(hook_window_seconds)
    caption_lead_ticks = seconds_to_ticks(caption_lead_seconds)
    caption_clearance_ticks = seconds_to_ticks(caption_clearance_seconds)
    frame_bounds = get_sequence_frame_bounds(root, sequence)
    titles_created = 0
    ts_counter = 1000
    style_counts: Counter[str] = Counter()
    all_mapped_words: list[TimedWord] = []
    generated_chunks: list[TitleChunk] = []

    for _, mapped_words in mapped_words_by_clip:
        all_mapped_words.extend(mapped_words)

    keyword_scores = score_words(all_mapped_words)
    layout_contexts: dict[Path, LayoutContext] = {}
    if caption_mode != "classic":
        for media_path in sorted({clip.media_path for clip, _ in mapped_words_by_clip}, key=str):
            layout_contexts[media_path] = detect_layout_context(media_path)

    for clip, mapped_words in mapped_words_by_clip:
        chunks = chunk_words(
            words=mapped_words,
            clip=clip,
            max_words=max_words_per_pop,
            max_gap_ticks=max_gap_ticks,
            caption_mode=caption_mode,
            variation_rate=variation_rate,
            hook_window_ticks=hook_window_ticks,
            style_seed=style_seed,
            keyword_scores=keyword_scores,
            layout_context=layout_contexts.get(clip.media_path),
        )
        generated_chunks.extend(chunk for chunk in chunks if chunk.text)

    for chunk in normalize_caption_chunks(generated_chunks, caption_clearance_ticks):
        ts_counter = add_title_to_gap(
            gap,
            chunk,
            effect_id,
            ts_counter,
            caption_lead_ticks,
            frame_bounds,
        )
        titles_created += 1
        style_counts[chunk.style_name] += 1

    return gap, titles_created, style_counts


def build_output_path(input_xml: Path, explicit_output: str | None) -> Path:
    if explicit_output:
        return Path(explicit_output).expanduser().resolve()
    return input_xml.with_name(f"{input_xml.stem}_captions.fcpxml")


def choose_cache_dir(input_xml: Path, output_xml: Path, explicit_cache_dir: str | None) -> Path:
    if explicit_cache_dir:
        return Path(explicit_cache_dir).expanduser().resolve()

    candidates = [
        input_xml.parent / ".caption_cache",
        output_xml.parent / ".caption_cache",
        Path.cwd() / ".caption_cache",
    ]

    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate.resolve()
        except OSError:
            continue

    return (output_xml.parent / ".caption_cache").resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate FCPXML title captions by transcribing the media referenced in an FCPXML file.",
    )
    parser.add_argument("input_xml", help="Path to the source FCPXML file.")
    parser.add_argument(
        "-o",
        "--output",
        help="Path for the generated FCPXML. Defaults to <input>_captions.fcpxml.",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "faster-whisper", "whisper"],
        default="auto",
        help="Whisper backend to use.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model name or path. Defaults to large-v3.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use. Defaults to auto, which resolves to cuda or cpu.",
    )
    parser.add_argument(
        "--compute-type",
        default="auto",
        help="faster-whisper compute type. Defaults to auto.",
    )
    parser.add_argument(
        "--language",
        help="Optional language hint, for example en.",
    )
    parser.add_argument(
        "--max-words-per-pop",
        type=int,
        default=DEFAULT_WORDS_PER_POP,
        help="Maximum words per title card.",
    )
    parser.add_argument(
        "--max-gap-seconds",
        type=float,
        default=DEFAULT_MAX_GAP_SECONDS,
        help="Split titles when the pause between words exceeds this many seconds.",
    )
    parser.add_argument(
        "--effect-id",
        default=DEFAULT_EFFECT_ID,
        help="Effect resource id for the title template.",
    )
    parser.add_argument(
        "--gap-name",
        default=DEFAULT_GAP_NAME,
        help="Gap name to use for generated titles.",
    )
    parser.add_argument(
        "--cache-dir",
        help="Optional transcript cache directory. Defaults to .caption_cache beside the input XML.",
    )
    parser.add_argument(
        "--caption-mode",
        choices=["classic", "instagram-variable"],
        default=DEFAULT_CAPTION_MODE,
        help=(
            "Caption style engine. 'classic' preserves the current 3-word pop style. "
            "'instagram-variable' keeps the same base system but mixes in stacked and emphasis layouts."
        ),
    )
    parser.add_argument(
        "--variation-rate",
        type=float,
        default=DEFAULT_VARIATION_RATE,
        help="How often the variable mode should use non-classic templates. Defaults to 0.28.",
    )
    parser.add_argument(
        "--caption-seed",
        type=int,
        default=DEFAULT_CAPTION_SEED,
        help="Deterministic seed for style selection.",
    )
    parser.add_argument(
        "--hook-window-seconds",
        type=float,
        default=DEFAULT_HOOK_WINDOW_SECONDS,
        help="Treat captions in the first N seconds as hook candidates. Defaults to 3.0.",
    )
    parser.add_argument(
        "--caption-lead-seconds",
        type=float,
        help=(
            "Show captions slightly before the spoken audio. "
            "Defaults to 0.0 in classic mode and 0.02 in instagram-variable mode."
        ),
    )
    parser.add_argument(
        "--caption-clearance-seconds",
        type=float,
        default=DEFAULT_CAPTION_CLEARANCE_SECONDS,
        help=(
            "Trim each generated title so it ends this long before the next title starts "
            "when possible. Helps prevent stacked overlaps in the timeline."
        ),
    )
    parser.add_argument(
        "--keep-existing-captions",
        action="store_true",
        help="Leave existing <caption> nodes in place instead of removing them.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_xml = Path(args.input_xml).expanduser().resolve()
    if not input_xml.exists():
        print(f"Input XML not found: {input_xml}", file=sys.stderr)
        return 1

    output_xml = build_output_path(input_xml, args.output)
    cache_dir = choose_cache_dir(input_xml, output_xml, args.cache_dir)

    tree = ET.parse(input_xml)
    root = tree.getroot()

    backend = choose_backend(args.backend)
    device = detect_device() if args.device == "auto" else args.device
    compute_type = None if args.compute_type == "auto" else args.compute_type

    print(f"Input XML: {input_xml}")
    print(f"Output XML: {output_xml}")
    print(f"Backend: {backend}")
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Caption Mode: {args.caption_mode}")

    assets = build_asset_resources(root, input_xml)
    if not assets:
        print("No media assets with file paths were found in the FCPXML.", file=sys.stderr)
        return 1

    sequence, spine, clips = collect_clip_uses(root, assets)
    if not clips:
        print("No video clips referencing media assets were found in the sequence.", file=sys.stderr)
        return 1

    missing_paths = sorted(
        {clip.media_path for clip in clips if not clip.media_path.exists()},
        key=str,
    )
    if missing_paths:
        print("Missing media files referenced by the FCPXML:", file=sys.stderr)
        for path in missing_paths:
            print(f"  {path}", file=sys.stderr)
        return 1

    print(f"Found {len(clips)} video clip placements across {len({clip.media_path for clip in clips})} asset(s).")
    for clip in clips[:10]:
        print(
            "  "
            f"{clip.media_path.name} | seq={ticks_to_seconds(clip.sequence_offset_ticks):.2f}s "
            f"| src_in={ticks_to_seconds(clip.source_in_ticks):.2f}s "
            f"| dur={ticks_to_seconds(clip.duration_ticks):.2f}s"
        )
    if len(clips) > 10:
        print(f"  ... {len(clips) - 10} more clip placements")

    transcriber = build_transcriber(
        backend=backend,
        model_name=args.model,
        device=device,
        compute_type=compute_type,
        language=args.language,
    )

    mapped_words_by_clip: list[tuple[ClipUse, list[TimedWord]]] = []
    total_used_seconds = 0.0
    for clip in clips:
        clip_words = transcribe_clip_use(
            clip=clip,
            transcriber=transcriber,
            backend=backend,
            model_name=args.model,
            language=args.language,
            cache_dir=cache_dir,
        )
        if not clip_words:
            continue
        mapped_words_by_clip.append((clip, clip_words))
        total_used_seconds += ticks_to_seconds(clip.duration_ticks)

    print(
        f"Transcribed {len(mapped_words_by_clip)} used clip range(s) "
        f"covering {total_used_seconds:.2f}s of timeline media."
    )

    sequence_duration_ticks = get_sequence_duration_ticks(sequence, clips)
    caption_lead_seconds = args.caption_lead_seconds
    if caption_lead_seconds is None:
        if args.caption_mode == "instagram-variable":
            caption_lead_seconds = DEFAULT_VARIABLE_CAPTION_LEAD_SECONDS
        else:
            caption_lead_seconds = DEFAULT_CLASSIC_CAPTION_LEAD_SECONDS

    gap, titles_created, style_counts = build_titles_gap(
        root=root,
        sequence=sequence,
        mapped_words_by_clip=mapped_words_by_clip,
        sequence_duration_ticks=sequence_duration_ticks,
        effect_id=args.effect_id,
        gap_name=args.gap_name,
        max_words_per_pop=max(1, args.max_words_per_pop),
        max_gap_seconds=max(0.05, args.max_gap_seconds),
        caption_mode=args.caption_mode,
        variation_rate=min(max(args.variation_rate, 0.0), 1.0),
        hook_window_seconds=max(0.0, args.hook_window_seconds),
        style_seed=args.caption_seed,
        caption_lead_seconds=max(0.0, caption_lead_seconds),
        caption_clearance_seconds=max(0.0, args.caption_clearance_seconds),
    )

    removed_gap_count = 0
    removed_caption_count = 0
    if titles_created > 0:
        ensure_effect_resource(root, args.effect_id)
        removed_gap_count = remove_existing_generated_gap(spine, args.gap_name)
        if not args.keep_existing_captions:
            removed_caption_count = remove_existing_captions(root)
        spine.insert(0, gap)

    tree.write(output_xml, encoding="UTF-8", xml_declaration=True)

    print(f"Removed {removed_gap_count} existing generated gap(s).")
    print(f"Removed {removed_caption_count} existing caption node(s).")
    print(f"Created {titles_created} title(s).")
    if style_counts:
        print("Style counts:")
        for style_name, count in sorted(style_counts.items()):
            print(f"  {style_name}: {count}")
    if titles_created == 0:
        print("No spoken words were turned into titles, so the output timeline was left unchanged.")
    print(f"Wrote {output_xml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
