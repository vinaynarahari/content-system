from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote, urlparse

VIDEO_EXTENSIONS = {".mov", ".mp4", ".m4v", ".avi", ".mkv"}
FCPXML_EXTENSIONS = {".fcpxml"}
FCPXML_PACKAGE_EXTENSION = ".fcpxmld"
DEFAULT_LUT_DIR = "LUTs"
DEFAULT_OUTPUT_DIR = "Graded"
DEFAULT_SOCIAL_QUALITY = "18"
LUT_ALIASES_FILE = "lut_aliases.json"


@dataclass(frozen=True)
class VideoMetadata:
    path: Path
    width: int
    height: int
    duration_seconds: float
    pix_fmt: str
    color_space: str
    color_transfer: str
    color_primaries: str

    @property
    def source_profile(self) -> str:
        if (
            self.color_space == "bt2020nc"
            and self.color_transfer == "arib-std-b67"
            and self.color_primaries == "bt2020"
        ):
            return "iphone-hlg"

        if (
            self.color_space == "bt709"
            and self.color_transfer == "bt709"
            and self.color_primaries == "bt709"
        ):
            return "rec709"

        return "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize iPhone footage into a LUT-ready working image and optionally apply a .cube LUT."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help=(
            "One or more video files, directories of video files, "
            "FCPXML files, or .fcpxmld project packages."
        ),
    )
    parser.add_argument(
        "--lut",
        help="A LUT alias, filename, stem, or absolute path to a .cube file.",
    )
    parser.add_argument(
        "--look",
        choices=["custom1", "custom2", "custom3", "custom4"],
        help=(
            "Apply a built-in programmatic grade instead of a LUT. "
            "custom1 is tuned for talking-head social videos. "
            "custom2 is a brighter, more vibrant variant with cleaner skin tones. "
            "custom3 is a warmer, sunnier California-style variant. "
            "custom4 is a flatter, rebuild-from-scratch talking-head grade."
        ),
    )
    parser.add_argument(
        "--lut-dir",
        default=DEFAULT_LUT_DIR,
        help="Directory that stores user-supplied LUTs. Defaults to ./LUTs.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for graded output files. Defaults to ./Graded.",
    )
    parser.add_argument(
        "--working-space",
        choices=["rec709", "flat709"],
        default="rec709",
        help=(
            "Working image before LUT application. "
            "Use rec709 for LUTs designed for normal SDR input. "
            "Use flat709 for a softer, lower-contrast base."
        ),
    )
    parser.add_argument(
        "--preset",
        choices=["social", "intermediate"],
        default="social",
        help=(
            "Output intent. "
            "social creates a compressed delivery file for Instagram/Reels/TikTok. "
            "intermediate creates a larger edit-friendly file."
        ),
    )
    parser.add_argument(
        "--codec",
        choices=["prores", "h264", "hevc"],
        help="Optional codec override. Defaults depend on --preset.",
    )
    parser.add_argument(
        "--quality",
        default=DEFAULT_SOCIAL_QUALITY,
        help=(
            "CRF value for h264/hevc output. Lower is higher quality. Ignored for ProRes. "
            f"Defaults to {DEFAULT_SOCIAL_QUALITY}."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the ffmpeg commands without running them.",
    )
    return parser.parse_args()


def iter_input_files(paths: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Input path not found: {path}")

        if path.is_file():
            if path.suffix.lower() in VIDEO_EXTENSIONS:
                files.append(path)
            elif path.suffix.lower() in FCPXML_EXTENSIONS:
                files.extend(resolve_fcpxml_media(path))
            continue

        if path.suffix.lower() == FCPXML_PACKAGE_EXTENSION:
            files.extend(resolve_fcpxmld_media(path))
            continue

        for child in sorted(path.iterdir()):
            if child.is_file() and child.suffix.lower() in VIDEO_EXTENSIONS:
                files.append(child.resolve())

    unique_files: list[Path] = []
    seen: set[Path] = set()
    for path in files:
        if path not in seen:
            unique_files.append(path)
            seen.add(path)
    return unique_files


def resolve_fcpxmld_media(package_path: Path) -> list[Path]:
    info_path = package_path / "Info.fcpxml"
    if not info_path.exists():
        raise FileNotFoundError(f"FCPXML package is missing Info.fcpxml: {package_path}")
    return resolve_fcpxml_media(info_path)


def resolve_fcpxml_media(xml_path: Path) -> list[Path]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    resources = root.find("resources")
    if resources is None:
        return []

    assets: dict[str, Path] = {}
    for asset in resources.findall("asset"):
        asset_id = asset.get("id")
        if not asset_id or asset.get("hasVideo") == "0":
            continue

        src = extract_asset_src(asset)
        if not src:
            continue

        assets[asset_id] = resolve_media_path(src, xml_path)

    used_asset_ids = {
        element.get("ref")
        for element in root.iter()
        if element.get("ref") in assets
    }

    if used_asset_ids:
        paths = [assets[asset_id] for asset_id in sorted(used_asset_ids)]
    else:
        paths = [assets[asset_id] for asset_id in sorted(assets)]

    return [path for path in paths if path.suffix.lower() in VIDEO_EXTENSIONS]


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


def probe_video(path: Path) -> VideoMetadata:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,pix_fmt,color_space,color_transfer,color_primaries:format=duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    payload = json.loads(result.stdout)
    streams = payload.get("streams") or []
    if not streams:
        raise ValueError(f"No video stream found in {path}")

    stream = streams[0]
    format_payload = payload.get("format") or {}
    return VideoMetadata(
        path=path,
        width=int(stream.get("width") or 0),
        height=int(stream.get("height") or 0),
        duration_seconds=float(format_payload.get("duration") or 0.0),
        pix_fmt=stream.get("pix_fmt") or "unknown",
        color_space=stream.get("color_space") or "unknown",
        color_transfer=stream.get("color_transfer") or "unknown",
        color_primaries=stream.get("color_primaries") or "unknown",
    )


def escape_filter_path(path: Path) -> str:
    escaped = str(path.resolve())
    escaped = escaped.replace("\\", "\\\\")
    escaped = escaped.replace(":", "\\:")
    escaped = escaped.replace("'", r"\'")
    return escaped


def normalize_lut_key(value: str) -> str:
    normalized = value.strip().lower()
    normalized = re.sub(r"\.cube$", "", normalized)
    normalized = re.sub(r"[^a-z0-9]+", "-", normalized)
    normalized = re.sub(r"-+", "-", normalized).strip("-")
    return normalized


def load_lut_aliases(config_path: Path) -> dict[str, str]:
    if not config_path.exists():
        return {}

    payload = json.loads(config_path.read_text())
    aliases = payload.get("aliases", {})
    return {
        normalize_lut_key(alias): relative_path
        for alias, relative_path in aliases.items()
        if isinstance(alias, str) and isinstance(relative_path, str)
    }


def resolve_lut(lut_arg: str | None, lut_dir: Path) -> Path | None:
    if not lut_arg:
        return None

    candidate = Path(lut_arg).expanduser()
    if candidate.exists():
        return candidate.resolve()

    direct = (lut_dir / lut_arg).resolve()
    if direct.exists():
        return direct

    aliases = load_lut_aliases(Path(__file__).with_name(LUT_ALIASES_FILE))
    alias_match = aliases.get(normalize_lut_key(lut_arg))
    if alias_match:
        aliased = Path(alias_match).expanduser()
        if not aliased.is_absolute():
            aliased = (lut_dir / aliased).resolve()
        else:
            aliased = aliased.resolve()
        if aliased.exists():
            return aliased

    if not candidate.suffix:
        target_key = normalize_lut_key(lut_arg)
        recursive_matches = sorted(lut_dir.rglob("*.cube"))
        for path in recursive_matches:
            if normalize_lut_key(path.stem) == target_key:
                return path.resolve()

    raise FileNotFoundError(f"Could not resolve LUT '{lut_arg}' in {lut_dir}")


def build_normalization_filters(metadata: VideoMetadata, working_space: str) -> list[str]:
    filters: list[str] = []

    if metadata.source_profile == "iphone-hlg":
        filters.append(
            "scale="
            "in_color_matrix=bt2020:"
            "out_color_matrix=bt709:"
            "in_primaries=bt2020:"
            "out_primaries=bt709:"
            "in_transfer=arib-std-b67:"
            "out_transfer=bt709"
        )

    if working_space == "flat709":
        filters.append("eq=contrast=0.92:saturation=0.92:brightness=0.01")

    return filters


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def median_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


@dataclass(frozen=True)
class GradeAnalysis:
    sample_count: int
    face_sample_count: int
    global_luma_mean: float
    global_luma_std: float
    global_saturation_mean: float
    shadow_clip: float
    highlight_clip: float
    face_luma_mean: float | None
    face_luma_std: float | None
    face_saturation_mean: float | None
    face_red_mean: float | None
    face_green_mean: float | None
    face_blue_mean: float | None

    @property
    def face_ratio(self) -> float:
        if self.sample_count <= 0:
            return 0.0
        return self.face_sample_count / self.sample_count


@dataclass(frozen=True)
class CustomLookSettings:
    name: str
    analysis: GradeAnalysis
    curves_master: str
    brightness: float
    contrast: float
    saturation: float
    warm_shift: float
    tint_shift: float
    vibrance_amount: float
    sharpen_amount: float


def select_primary_face(
    faces: list[tuple[int, int, int, int]],
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int] | None:
    if not faces:
        return None

    center_x = frame_width / 2.0
    center_y = frame_height * 0.42
    best_face: tuple[int, int, int, int] | None = None
    best_score = -1.0

    for x, y, w, h in faces:
        area = float(w * h)
        face_center_x = x + (w / 2.0)
        face_center_y = y + (h / 2.0)
        distance = ((face_center_x - center_x) / frame_width) ** 2 + (
            (face_center_y - center_y) / frame_height
        ) ** 2
        score = area * (1.35 - distance)
        if score > best_score:
            best_score = score
            best_face = (x, y, w, h)

    return best_face


def crop_face_center(
    frame,
    face_box: tuple[int, int, int, int],
):
    x, y, w, h = face_box
    inset_x = max(0, x + int(w * 0.22))
    inset_y = max(0, y + int(h * 0.24))
    inset_w = max(1, int(w * 0.56))
    inset_h = max(1, int(h * 0.48))
    end_x = min(frame.shape[1], inset_x + inset_w)
    end_y = min(frame.shape[0], inset_y + inset_h)
    return frame[inset_y:end_y, inset_x:end_x]


def analyze_video_for_custom_look(metadata: VideoMetadata, max_samples: int = 12) -> GradeAnalysis:
    try:
        import cv2
        import numpy as np
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Built-in custom looks require a working OpenCV + NumPy install for frame analysis."
        ) from exc

    cap = cv2.VideoCapture(str(metadata.path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for custom look analysis: {metadata.path}")

    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(str(cascade_path))
    if face_cascade.empty():
        raise RuntimeError(f"Could not load face cascade: {cascade_path}")

    duration = metadata.duration_seconds
    if duration <= 0:
        frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if frame_count > 0 and fps > 0:
            duration = frame_count / fps
        else:
            duration = 1.0

    sample_count = min(max_samples, max(3, int(duration // 6) + 4))
    if duration <= 1.5 or sample_count <= 1:
        sample_times = [max(duration * 0.5, 0.0)]
    else:
        edge_padding = min(0.75, duration * 0.08)
        start = edge_padding
        end = max(start + 0.01, duration - edge_padding)
        sample_times = [
            start + ((end - start) * index / (sample_count - 1))
            for index in range(sample_count)
        ]

    global_luma_means: list[float] = []
    global_luma_stds: list[float] = []
    global_saturation_means: list[float] = []
    shadow_clips: list[float] = []
    highlight_clips: list[float] = []
    face_luma_means: list[float] = []
    face_luma_stds: list[float] = []
    face_saturation_means: list[float] = []
    face_red_means: list[float] = []
    face_green_means: list[float] = []
    face_blue_means: list[float] = []
    processed_samples = 0

    try:
        for sample_time in sample_times:
            cap.set(cv2.CAP_PROP_POS_MSEC, sample_time * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            height, width = frame.shape[:2]
            max_dimension = max(height, width)
            if max_dimension > 720:
                scale = 720.0 / max_dimension
                frame = cv2.resize(
                    frame,
                    (max(1, int(width * scale)), max(1, int(height * scale))),
                    interpolation=cv2.INTER_AREA,
                )

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            luma = gray.astype("float32") / 255.0
            saturation = hsv[:, :, 1].astype("float32") / 255.0

            global_luma_means.append(float(luma.mean()))
            global_luma_stds.append(float(luma.std()))
            global_saturation_means.append(float(saturation.mean()))
            shadow_clips.append(float((luma < 0.04).mean()))
            highlight_clips.append(float((luma > 0.96).mean()))
            processed_samples += 1

            face_candidates = face_cascade.detectMultiScale(
                cv2.equalizeHist(gray),
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(max(48, frame.shape[1] // 7), max(48, frame.shape[1] // 7)),
            )
            face_box = select_primary_face(
                [tuple(map(int, candidate)) for candidate in face_candidates],
                frame.shape[1],
                frame.shape[0],
            )
            if face_box is None:
                continue

            face_roi = crop_face_center(frame, face_box)
            if face_roi.size == 0:
                continue

            face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY).astype("float32") / 255.0
            face_hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
            face_sat = face_hsv[:, :, 1].astype("float32") / 255.0
            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB).astype("float32") / 255.0

            face_luma_means.append(float(face_gray.mean()))
            face_luma_stds.append(float(face_gray.std()))
            face_saturation_means.append(float(face_sat.mean()))
            face_red_means.append(float(face_rgb[:, :, 0].mean()))
            face_green_means.append(float(face_rgb[:, :, 1].mean()))
            face_blue_means.append(float(face_rgb[:, :, 2].mean()))
    finally:
        cap.release()

    if processed_samples == 0:
        raise RuntimeError(f"Custom look analysis could not sample any frames from {metadata.path}")

    return GradeAnalysis(
        sample_count=processed_samples,
        face_sample_count=len(face_luma_means),
        global_luma_mean=float(statistics.median(global_luma_means)),
        global_luma_std=float(statistics.median(global_luma_stds)),
        global_saturation_mean=float(statistics.median(global_saturation_means)),
        shadow_clip=float(statistics.median(shadow_clips)),
        highlight_clip=float(statistics.median(highlight_clips)),
        face_luma_mean=median_or_none(face_luma_means),
        face_luma_std=median_or_none(face_luma_stds),
        face_saturation_mean=median_or_none(face_saturation_means),
        face_red_mean=median_or_none(face_red_means),
        face_green_mean=median_or_none(face_green_means),
        face_blue_mean=median_or_none(face_blue_means),
    )


def derive_custom1_settings(analysis: GradeAnalysis) -> CustomLookSettings:
    reference_luma = analysis.face_luma_mean or analysis.global_luma_mean
    reference_saturation = analysis.face_saturation_mean or analysis.global_saturation_mean
    face_weight = clamp(analysis.face_ratio / 0.5, 0.0, 1.0)

    target_face_luma = 0.60
    exposure_error = target_face_luma - reference_luma

    brightness = clamp((exposure_error * 0.18) - (analysis.highlight_clip * 0.03), -0.04, 0.05)
    contrast = clamp(
        1.00 + ((0.18 - analysis.global_luma_std) * 0.28) - (analysis.highlight_clip * 0.12),
        0.99,
        1.08,
    )
    saturation = clamp(
        1.04 + ((0.24 - reference_saturation) * 0.22) - (analysis.highlight_clip * 0.04),
        1.00,
        1.10,
    )

    shadow_point = clamp(
        0.20 + (exposure_error * 0.07) - (analysis.shadow_clip * 0.08),
        0.17,
        0.23,
    )
    mid_point = clamp(0.50 + (exposure_error * 0.28), 0.46, 0.58)
    high_point = clamp(
        0.82 + (exposure_error * 0.04) - (max(0.0, analysis.highlight_clip - 0.02) * 0.55),
        0.78,
        0.88,
    )
    curves_master = (
        f"0/0 0.22/{shadow_point:.3f} 0.50/{mid_point:.3f} "
        f"0.82/{high_point:.3f} 1/1"
    )

    warm_shift = 0.0
    tint_shift = 0.0
    if (
        face_weight > 0.0
        and analysis.face_red_mean is not None
        and analysis.face_green_mean is not None
        and analysis.face_blue_mean is not None
    ):
        warmth = analysis.face_red_mean - analysis.face_blue_mean
        green_bias = analysis.face_green_mean - (
            (analysis.face_red_mean + analysis.face_blue_mean) / 2.0
        )
        warm_shift = clamp(0.014 + ((0.08 - warmth) * 0.08), 0.008, 0.03) * face_weight
        tint_shift = clamp(((-green_bias) * 0.12) - 0.01, -0.025, 0.0) * face_weight

    sharpen_amount = clamp(
        0.22 + max(0.0, 0.17 - analysis.global_luma_std) * 0.75,
        0.14,
        0.32,
    )
    if reference_luma < 0.46 or analysis.shadow_clip > 0.08:
        sharpen_amount = max(0.12, sharpen_amount - 0.05)

    return CustomLookSettings(
        name="custom1",
        analysis=analysis,
        curves_master=curves_master,
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        warm_shift=warm_shift,
        tint_shift=tint_shift,
        vibrance_amount=0.0,
        sharpen_amount=sharpen_amount,
    )


def build_custom1_filters(settings: CustomLookSettings) -> list[str]:
    warm_highlight = settings.warm_shift * 0.55
    tint_highlight = settings.tint_shift * 0.45
    cool_midtone = settings.warm_shift * -0.82
    cool_highlight = warm_highlight * -0.82

    return [
        "colorcorrect=analyze=median:saturation=1",
        (
            "colorbalance="
            f"rm={settings.warm_shift:.4f}:"
            f"gm={settings.tint_shift:.4f}:"
            f"bm={cool_midtone:.4f}:"
            f"rh={warm_highlight:.4f}:"
            f"gh={tint_highlight:.4f}:"
            f"bh={cool_highlight:.4f}:"
            "pl=1"
        ),
        f"curves=master='{settings.curves_master}':interp=pchip",
        (
            "eq="
            f"brightness={settings.brightness:.4f}:"
            f"contrast={settings.contrast:.4f}:"
            f"saturation={settings.saturation:.4f}"
        ),
        f"unsharp=lx=5:ly=5:la={settings.sharpen_amount:.4f}",
    ]


def derive_custom2_settings(analysis: GradeAnalysis) -> CustomLookSettings:
    reference_luma = analysis.face_luma_mean or analysis.global_luma_mean
    reference_saturation = analysis.face_saturation_mean or analysis.global_saturation_mean
    face_weight = clamp(analysis.face_ratio / 0.45, 0.0, 1.0)

    target_face_luma = 0.63
    exposure_error = target_face_luma - reference_luma

    brightness = clamp((exposure_error * 0.22) - (analysis.highlight_clip * 0.03), -0.015, 0.06)
    contrast = clamp(
        1.03 + ((0.17 - analysis.global_luma_std) * 0.32) - (analysis.highlight_clip * 0.10),
        1.01,
        1.12,
    )
    saturation = clamp(
        1.08 + ((0.25 - reference_saturation) * 0.28) - (analysis.highlight_clip * 0.03),
        1.04,
        1.16,
    )

    shadow_point = clamp(
        0.21 + (exposure_error * 0.06) - (analysis.shadow_clip * 0.06),
        0.18,
        0.25,
    )
    mid_point = clamp(0.52 + (exposure_error * 0.30), 0.49, 0.60)
    high_point = clamp(
        0.84 + (exposure_error * 0.05) - (max(0.0, analysis.highlight_clip - 0.03) * 0.45),
        0.80,
        0.90,
    )
    curves_master = (
        f"0/0 0.22/{shadow_point:.3f} 0.50/{mid_point:.3f} "
        f"0.82/{high_point:.3f} 1/1"
    )

    warm_shift = clamp(
        0.010
        + (max(0.0, 0.28 - analysis.global_saturation_mean) * 0.05)
        - (analysis.highlight_clip * 0.02),
        0.008,
        0.022,
    )
    tint_shift = clamp(-0.006 - (analysis.highlight_clip * 0.02), -0.012, -0.003)
    if (
        analysis.face_red_mean is not None
        and analysis.face_green_mean is not None
        and analysis.face_blue_mean is not None
    ):
        warmth = analysis.face_red_mean - analysis.face_blue_mean
        green_bias = analysis.face_green_mean - (
            (analysis.face_red_mean + analysis.face_blue_mean) / 2.0
        )
        warm_shift += clamp(0.012 + ((0.10 - warmth) * 0.11), 0.0, 0.022) * max(
            0.45, face_weight
        )
        tint_shift += clamp(((-green_bias) * 0.10) - 0.002, -0.010, 0.004) * max(
            0.45, face_weight
        )

    warm_shift = clamp(warm_shift, 0.010, 0.040)
    tint_shift = clamp(tint_shift, -0.018, 0.004)
    vibrance_amount = clamp(
        0.10
        + (max(0.0, 0.27 - analysis.global_saturation_mean) * 0.34)
        + (max(0.0, 0.24 - reference_saturation) * 0.20)
        - (analysis.highlight_clip * 0.10),
        0.08,
        0.24,
    )

    sharpen_amount = clamp(
        0.24 + max(0.0, 0.18 - analysis.global_luma_std) * 0.85,
        0.16,
        0.34,
    )
    if reference_luma < 0.46 or analysis.shadow_clip > 0.08:
        sharpen_amount = max(0.15, sharpen_amount - 0.05)

    return CustomLookSettings(
        name="custom2",
        analysis=analysis,
        curves_master=curves_master,
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        warm_shift=warm_shift,
        tint_shift=tint_shift,
        vibrance_amount=vibrance_amount,
        sharpen_amount=sharpen_amount,
    )


def build_custom2_filters(settings: CustomLookSettings) -> list[str]:
    warm_highlight = settings.warm_shift * 0.72
    tint_highlight = settings.tint_shift * 0.25
    cool_midtone = settings.warm_shift * -0.34
    cool_highlight = warm_highlight * -0.18

    return [
        "colorcorrect=analyze=median:saturation=1",
        (
            "colorbalance="
            f"rm={settings.warm_shift:.4f}:"
            f"gm={settings.tint_shift:.4f}:"
            f"bm={cool_midtone:.4f}:"
            f"rh={warm_highlight:.4f}:"
            f"gh={tint_highlight:.4f}:"
            f"bh={cool_highlight:.4f}:"
            "pl=1"
        ),
        f"curves=master='{settings.curves_master}':interp=pchip",
        (
            "eq="
            f"brightness={settings.brightness:.4f}:"
            f"contrast={settings.contrast:.4f}:"
            f"saturation={settings.saturation:.4f}"
        ),
        f"vibrance=intensity={settings.vibrance_amount:.4f}",
        f"unsharp=lx=5:ly=5:la={settings.sharpen_amount:.4f}",
    ]


def derive_custom3_settings(analysis: GradeAnalysis) -> CustomLookSettings:
    reference_luma = analysis.face_luma_mean or analysis.global_luma_mean
    reference_saturation = analysis.face_saturation_mean or analysis.global_saturation_mean
    face_weight = clamp(analysis.face_ratio / 0.42, 0.0, 1.0)

    target_face_luma = 0.66
    exposure_error = target_face_luma - reference_luma

    brightness = clamp((exposure_error * 0.24) - (analysis.highlight_clip * 0.02), 0.0, 0.07)
    contrast = clamp(
        1.04 + ((0.17 - analysis.global_luma_std) * 0.26) - (analysis.highlight_clip * 0.08),
        1.02,
        1.10,
    )
    saturation = clamp(
        1.10 + ((0.26 - reference_saturation) * 0.30) - (analysis.highlight_clip * 0.03),
        1.06,
        1.18,
    )

    shadow_point = clamp(
        0.22 + (exposure_error * 0.05) - (analysis.shadow_clip * 0.05),
        0.19,
        0.26,
    )
    mid_point = clamp(0.53 + (exposure_error * 0.34), 0.51, 0.62)
    high_point = clamp(
        0.86 + (exposure_error * 0.04) - (max(0.0, analysis.highlight_clip - 0.04) * 0.35),
        0.83,
        0.91,
    )
    curves_master = (
        f"0/0 0.22/{shadow_point:.3f} 0.50/{mid_point:.3f} "
        f"0.82/{high_point:.3f} 1/1"
    )

    warm_shift = clamp(
        0.016
        + (max(0.0, 0.30 - analysis.global_saturation_mean) * 0.05)
        + (max(0.0, 0.52 - reference_luma) * 0.02)
        - (analysis.highlight_clip * 0.02),
        0.014,
        0.028,
    )
    tint_shift = clamp(-0.003 - (analysis.highlight_clip * 0.015), -0.008, 0.001)
    if (
        analysis.face_red_mean is not None
        and analysis.face_green_mean is not None
        and analysis.face_blue_mean is not None
    ):
        warmth = analysis.face_red_mean - analysis.face_blue_mean
        green_bias = analysis.face_green_mean - (
            (analysis.face_red_mean + analysis.face_blue_mean) / 2.0
        )
        warm_shift += clamp(0.016 + ((0.12 - warmth) * 0.13), 0.0, 0.028) * max(
            0.55, face_weight
        )
        tint_shift += clamp(((-green_bias) * 0.09), -0.008, 0.006) * max(0.5, face_weight)

    warm_shift = clamp(warm_shift, 0.016, 0.050)
    tint_shift = clamp(tint_shift, -0.012, 0.006)
    vibrance_amount = clamp(
        0.14
        + (max(0.0, 0.28 - analysis.global_saturation_mean) * 0.34)
        + (max(0.0, 0.25 - reference_saturation) * 0.24)
        - (analysis.highlight_clip * 0.08),
        0.12,
        0.28,
    )

    sharpen_amount = clamp(
        0.22 + max(0.0, 0.18 - analysis.global_luma_std) * 0.70,
        0.15,
        0.30,
    )
    if reference_luma < 0.45 or analysis.shadow_clip > 0.08:
        sharpen_amount = max(0.14, sharpen_amount - 0.04)

    return CustomLookSettings(
        name="custom3",
        analysis=analysis,
        curves_master=curves_master,
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        warm_shift=warm_shift,
        tint_shift=tint_shift,
        vibrance_amount=vibrance_amount,
        sharpen_amount=sharpen_amount,
    )


def build_custom3_filters(settings: CustomLookSettings) -> list[str]:
    warm_midtone = settings.warm_shift
    warm_highlight = settings.warm_shift * 0.92
    cool_midtone = settings.warm_shift * -0.18
    cool_highlight = settings.warm_shift * -0.06

    return [
        "colorcorrect=analyze=median:saturation=1",
        (
            "colorbalance="
            f"rm={warm_midtone:.4f}:"
            f"gm={settings.tint_shift:.4f}:"
            f"bm={cool_midtone:.4f}:"
            f"rh={warm_highlight:.4f}:"
            f"gh={0.0000:.4f}:"
            f"bh={cool_highlight:.4f}:"
            "pl=1"
        ),
        f"curves=master='{settings.curves_master}':interp=pchip",
        (
            "eq="
            f"brightness={settings.brightness:.4f}:"
            f"contrast={settings.contrast:.4f}:"
            f"saturation={settings.saturation:.4f}"
        ),
        f"vibrance=intensity={settings.vibrance_amount:.4f}:rbal=1.10:gbal=1.00:bbal=0.92",
        f"unsharp=lx=5:ly=5:la={settings.sharpen_amount:.4f}",
    ]


def derive_custom4_settings(analysis: GradeAnalysis) -> CustomLookSettings:
    reference_luma = analysis.face_luma_mean or analysis.global_luma_mean
    reference_saturation = analysis.face_saturation_mean or analysis.global_saturation_mean
    face_weight = clamp(analysis.face_ratio / 0.40, 0.0, 1.0)

    target_face_luma = 0.60
    exposure_error = target_face_luma - reference_luma

    brightness = clamp((exposure_error * 0.16) + 0.006 - (analysis.highlight_clip * 0.02), 0.0, 0.04)
    contrast = clamp(
        1.03 + ((0.18 - analysis.global_luma_std) * 0.18) - (analysis.highlight_clip * 0.08),
        1.00,
        1.07,
    )
    saturation = clamp(
        1.05 + ((0.24 - reference_saturation) * 0.14) - (analysis.highlight_clip * 0.03),
        1.02,
        1.09,
    )

    shadow_point = clamp(
        0.20 + (exposure_error * 0.04) - (analysis.shadow_clip * 0.04),
        0.18,
        0.23,
    )
    mid_point = clamp(0.52 + (exposure_error * 0.18), 0.50, 0.57)
    high_point = clamp(
        0.84 + (exposure_error * 0.03) - (max(0.0, analysis.highlight_clip - 0.03) * 0.30),
        0.82,
        0.88,
    )
    curves_master = (
        f"0/0 0.22/{shadow_point:.3f} 0.50/{mid_point:.3f} "
        f"0.82/{high_point:.3f} 1/1"
    )

    warm_shift = 0.006
    tint_shift = 0.0
    if (
        face_weight > 0.0
        and analysis.face_red_mean is not None
        and analysis.face_green_mean is not None
        and analysis.face_blue_mean is not None
    ):
        warmth = analysis.face_red_mean - analysis.face_blue_mean
        green_bias = analysis.face_green_mean - (
            (analysis.face_red_mean + analysis.face_blue_mean) / 2.0
        )
        warm_shift = 0.006 + (clamp(0.11 - warmth, -0.02, 0.12) * 0.08 * max(0.55, face_weight))
        tint_shift = clamp((-green_bias) * 0.06, -0.004, 0.006) * max(0.45, face_weight)

    warm_shift = clamp(warm_shift, 0.004, 0.018)
    tint_shift = clamp(tint_shift, -0.004, 0.006)
    vibrance_amount = clamp(
        0.06
        + (max(0.0, 0.25 - analysis.global_saturation_mean) * 0.16)
        + (max(0.0, 0.23 - reference_saturation) * 0.10)
        - (analysis.highlight_clip * 0.04),
        0.05,
        0.12,
    )

    sharpen_amount = clamp(
        0.18 + max(0.0, 0.17 - analysis.global_luma_std) * 0.55,
        0.12,
        0.24,
    )
    if reference_luma < 0.45 or analysis.shadow_clip > 0.08:
        sharpen_amount = max(0.12, sharpen_amount - 0.03)

    return CustomLookSettings(
        name="custom4",
        analysis=analysis,
        curves_master=curves_master,
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        warm_shift=warm_shift,
        tint_shift=tint_shift,
        vibrance_amount=vibrance_amount,
        sharpen_amount=sharpen_amount,
    )


def build_custom4_filters(settings: CustomLookSettings, working_space: str) -> list[str]:
    filters: list[str] = []
    if working_space != "flat709":
        # Build from a softer, log-like base before applying the creative grade.
        filters.append("eq=contrast=0.96:saturation=0.96:brightness=0.006")

    red_mid_curve = clamp(0.53 + (settings.warm_shift * 1.6), 0.53, 0.57)
    blue_mid_curve = clamp(0.53 - (settings.warm_shift * 1.2), 0.50, 0.53)
    blue_high_curve = clamp(0.99 - (settings.warm_shift * 0.5), 0.98, 0.995)

    filters.extend(
        [
            (
                "colorbalance="
                f"rs={settings.warm_shift * 0.12:.4f}:"
                f"gs={settings.tint_shift * 0.12:.4f}:"
                f"bs={settings.warm_shift * -0.08:.4f}:"
                f"rm={settings.warm_shift:.4f}:"
                f"gm={settings.tint_shift:.4f}:"
                f"bm={settings.warm_shift * -0.10:.4f}:"
                f"rh={settings.warm_shift * 0.55:.4f}:"
                f"gh={settings.tint_shift * 0.20:.4f}:"
                f"bh={settings.warm_shift * -0.05:.4f}:"
                "pl=1"
            ),
            (
                "curves="
                f"master='{settings.curves_master}':"
                f"red='0/0 0.55/{red_mid_curve:.3f} 1/1':"
                f"blue='0/0 0.55/{blue_mid_curve:.3f} 1/{blue_high_curve:.3f}':"
                "interp=pchip"
            ),
            (
                "eq="
                f"brightness={settings.brightness:.4f}:"
                f"contrast={settings.contrast:.4f}:"
                f"saturation={settings.saturation:.4f}"
            ),
            f"vibrance=intensity={settings.vibrance_amount:.4f}",
            f"unsharp=lx=5:ly=5:la={settings.sharpen_amount:.4f}",
        ]
    )
    return filters


def derive_custom_look_settings(look_name: str, analysis: GradeAnalysis) -> CustomLookSettings:
    if look_name == "custom1":
        return derive_custom1_settings(analysis)
    if look_name == "custom2":
        return derive_custom2_settings(analysis)
    if look_name == "custom3":
        return derive_custom3_settings(analysis)
    if look_name == "custom4":
        return derive_custom4_settings(analysis)
    raise ValueError(f"Unsupported custom look: {look_name}")


def build_custom_look_filters(settings: CustomLookSettings, working_space: str) -> list[str]:
    if settings.name == "custom1":
        return build_custom1_filters(settings)
    if settings.name == "custom2":
        return build_custom2_filters(settings)
    if settings.name == "custom3":
        return build_custom3_filters(settings)
    if settings.name == "custom4":
        return build_custom4_filters(settings, working_space)
    raise ValueError(f"Unsupported custom look: {settings.name}")


def build_output_path(
    input_path: Path,
    output_dir: Path,
    working_space: str,
    lut_path: Path | None,
    look_name: str | None,
    codec: str,
) -> Path:
    if codec == "prores":
        suffix = ".mov"
    else:
        suffix = ".mp4"

    parts = [input_path.stem, working_space]
    if look_name is not None:
        parts.append(look_name)
    elif lut_path is not None:
        parts.append(lut_path.stem)
    else:
        parts.append("normalized")
    name = "__".join(parts) + suffix
    return output_dir / name


def resolve_codec(preset: str, codec_override: str | None) -> str:
    if codec_override:
        return codec_override

    if preset == "intermediate":
        return "prores"

    return "h264"


def build_ffmpeg_command(
    input_path: Path,
    output_path: Path,
    metadata: VideoMetadata,
    preset: str,
    working_space: str,
    lut_path: Path | None,
    custom_look: CustomLookSettings | None,
    codec: str,
    quality: str,
    overwrite: bool,
) -> list[str]:
    filters = build_normalization_filters(metadata, working_space)
    if custom_look is not None:
        filters.extend(build_custom_look_filters(custom_look, working_space))
    elif lut_path is not None:
        filters.append(f"lut3d=file='{escape_filter_path(lut_path)}':interp=tetrahedral")

    if codec == "prores":
        filters.append("format=yuv422p10le")
    elif metadata.pix_fmt.endswith("10le"):
        filters.append("format=yuv420p10le")
    else:
        filters.append("format=yuv420p")

    cmd = ["ffmpeg", "-hide_banner"]
    cmd.append("-y" if overwrite else "-n")
    cmd.extend(["-i", str(input_path)])

    if filters:
        cmd.extend(["-vf", ",".join(filters)])

    if codec == "prores":
        cmd.extend(
            [
                "-c:v",
                "prores_ks",
                "-profile:v",
                "3",
                "-pix_fmt",
                "yuv422p10le",
                "-vendor",
                "apl0",
            ]
        )
    elif codec == "h264":
        cmd.extend(
            [
                "-c:v",
                "libx264",
                "-crf",
                quality,
                "-pix_fmt",
                "yuv420p",
            ]
        )
    else:
        cmd.extend(
            [
                "-c:v",
                "libx265",
                "-crf",
                quality,
                "-pix_fmt",
                "yuv420p10le",
            ]
        )

    cmd.extend(["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"])

    if preset == "social":
        cmd.extend(["-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart"])
    else:
        cmd.extend(["-c:a", "copy"])

    cmd.append(str(output_path))
    return cmd


def main() -> int:
    args = parse_args()
    lut_dir = Path(args.lut_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    lut_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.lut and args.look:
        print("Use either --lut or --look, not both.", file=sys.stderr)
        return 1
    lut_path = resolve_lut(args.lut, lut_dir)
    codec = resolve_codec(args.preset, args.codec)

    input_files = iter_input_files(args.inputs)
    if not input_files:
        print("No supported video files found in the provided inputs.", file=sys.stderr)
        return 1

    for input_path in input_files:
        metadata = probe_video(input_path)
        custom_look = None
        if args.look is not None:
            custom_look = derive_custom_look_settings(
                args.look,
                analyze_video_for_custom_look(metadata),
            )
        output_path = build_output_path(
            input_path=input_path,
            output_dir=output_dir,
            working_space=args.working_space,
            lut_path=lut_path,
            look_name=custom_look.name if custom_look else None,
            codec=codec,
        )
        cmd = build_ffmpeg_command(
            input_path=input_path,
            output_path=output_path,
            metadata=metadata,
            preset=args.preset,
            working_space=args.working_space,
            lut_path=lut_path,
            custom_look=custom_look,
            codec=codec,
            quality=args.quality,
            overwrite=args.overwrite,
        )

        print(f"Input: {input_path}", flush=True)
        print(f"Detected profile: {metadata.source_profile}", flush=True)
        print(f"Preset: {args.preset}", flush=True)
        print(f"Codec: {codec}", flush=True)
        print(f"Working space: {args.working_space}", flush=True)
        print(f"Look: {custom_look.name if custom_look else 'none'}", flush=True)
        print(f"LUT: {lut_path if lut_path else 'none'}", flush=True)
        if custom_look is not None:
            analysis = custom_look.analysis
            face_luma = (
                f"{analysis.face_luma_mean:.3f}" if analysis.face_luma_mean is not None else "n/a"
            )
            print(
                (
                    "Look analysis: "
                    f"samples={analysis.sample_count}, "
                    f"face_samples={analysis.face_sample_count}, "
                    f"global_luma={analysis.global_luma_mean:.3f}, "
                    f"face_luma={face_luma}, "
                    f"global_sat={analysis.global_saturation_mean:.3f}"
                ),
                flush=True,
            )
            print(
                (
                    "Look params: "
                    f"brightness={custom_look.brightness:.4f}, "
                    f"contrast={custom_look.contrast:.4f}, "
                    f"saturation={custom_look.saturation:.4f}, "
                    f"warm={custom_look.warm_shift:.4f}, "
                    f"tint={custom_look.tint_shift:.4f}, "
                    f"vibrance={custom_look.vibrance_amount:.4f}, "
                    f"sharpen={custom_look.sharpen_amount:.4f}"
                ),
                flush=True,
            )
        print(f"Output: {output_path}", flush=True)

        if args.dry_run:
            print("Command:", flush=True)
            print(" ".join(cmd), flush=True)
            print(flush=True)
            continue

        subprocess.run(cmd, check=True)
        print(flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
