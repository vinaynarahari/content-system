from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import sqlite3
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import requests
from bs4 import BeautifulSoup
from yt_dlp import YoutubeDL

from make_matrx_reveal import OUTPUT_MOV as BRAND_REVEAL_MOV
from make_matrx_reveal import OUTPUT_MP4 as BRAND_REVEAL_MP4
from make_matrx_reveal import main as make_matrx_reveal_main
from video_script import (
    DEFAULT_BRIEF_PATH,
    DEFAULT_CREATORS,
    DEFAULT_CREATOR_PROFILES_PATH,
    DEFAULT_SPEECH_RATE_WPM,
    BriefClaims,
    CreatorProfile,
    SourceAnalysis,
    TranscriptSegment,
    analyze_source_transcript,
    clean_caption_text,
    estimate_spoken_seconds,
    extract_brief_claims,
    extract_transcript_text,
    fetch_transcript_from_url,
    generate_short_form_script,
    is_url,
    load_creator_profiles,
    load_transcript_from_file,
    slugify,
    split_sentences,
    word_count,
)


ROOT = Path(__file__).resolve().parent
STUDIO_DIR = ROOT / ".studio"
DB_PATH = STUDIO_DIR / "studio.db"
ASSETS_DIR = STUDIO_DIR / "assets"
PROJECTS_DIR = STUDIO_DIR / "projects"
REPORTS_DIR = ROOT / "analysis" / "studio"
TRAINING_STATE_PATH = REPORTS_DIR / "training_state.json"
TRAINING_REPORT_PATH = REPORTS_DIR / "training_report.md"
CONFIG_PATH = ROOT / "studio_config.json"

DEFAULT_CONFIG: dict[str, Any] = {
    "platform_priority": "instagram",
    "first_party_handles": {"instagram": [], "tiktok": []},
    "yt_dlp": {"cookies_from_browser": "", "cookie_file": ""},
    "browser_session": {"storage_state_path": "", "headless": True},
    "defaults": {
        "brief_path": str(DEFAULT_BRIEF_PATH.relative_to(ROOT)),
        "owner_type": "first_party_matrix",
        "caption_mode": "instagram-variable",
    },
}

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
    "for",
    "from",
    "has",
    "have",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "them",
    "they",
    "this",
    "to",
    "was",
    "we",
    "with",
    "you",
    "your",
}

SCRIPT_RISK_KEYWORDS = {
    "too_much_setup": ("because", "when", "recently", "once", "last week"),
    "weak_payoff": ("infrastructure", "system", "capability", "outcomes"),
}

HASHTAG_BANK = {
    "ai": ["#ai", "#aitools", "#artificialintelligence", "#aiautomation"],
    "coding": ["#coding", "#developer", "#softwareengineer", "#buildinpublic"],
    "agentic": ["#agenticai", "#aicoding", "#developertools", "#engineering"],
    "startup": ["#startup", "#founder", "#saas", "#b2bsaas"],
    "memory": ["#contextengineering", "#knowledgesystems", "#workflowautomation"],
    "matrx": ["#matrx", "#mtrx", "#engineeringops"],
}

MUSIC_BANK = {
    "contrarian-problem-solution": [
        {
            "query": "minimal tension tech beat",
            "reason": "Supports a blunt hook without overpowering the voiceover.",
            "energy": "medium",
        },
        {
            "query": "clean cinematic entrepreneur underscore",
            "reason": "Keeps the clip feeling polished and serious.",
            "energy": "low",
        },
    ],
    "story-turn-reveal": [
        {
            "query": "ambient story reveal beat",
            "reason": "Gives the middle reveal room to breathe.",
            "energy": "low",
        },
        {
            "query": "soft pulsing tech background",
            "reason": "Fits founder storytelling with product context.",
            "energy": "low",
        },
    ],
    "pain-reveal-payoff": [
        {
            "query": "subtle driving startup underscore",
            "reason": "Works for explanatory clips with forward motion.",
            "energy": "medium",
        },
        {
            "query": "clean motivational tech pulse",
            "reason": "Fits polished Reels voiceover with business stakes.",
            "energy": "medium",
        },
    ],
    "listicle-reframe": [
        {
            "query": "fast precise listicle beat",
            "reason": "Helps keep structured multi-point clips moving.",
            "energy": "high",
        }
    ],
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config() -> dict[str, Any]:
    config = DEFAULT_CONFIG
    if CONFIG_PATH.exists():
        config = deep_merge(DEFAULT_CONFIG, json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    return config


def ensure_dirs() -> None:
    STUDIO_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def get_connection(path: Path = DB_PATH) -> sqlite3.Connection:
    ensure_dirs()
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS content_items (
            content_id TEXT PRIMARY KEY,
            platform TEXT NOT NULL,
            platform_post_id TEXT,
            owner_type TEXT NOT NULL,
            url TEXT UNIQUE,
            title TEXT,
            uploader TEXT,
            published_at TEXT,
            duration_seconds REAL,
            raw_caption TEXT,
            transcript_text TEXT,
            hook_text TEXT,
            structure_type TEXT,
            topic TEXT,
            angle TEXT,
            views INTEGER,
            likes INTEGER,
            comments INTEGER,
            shares INTEGER,
            saves INTEGER,
            watch_time_total REAL,
            average_watch_time REAL,
            completion_rate REAL,
            accounts_reached INTEGER,
            follows INTEGER,
            profile_visits INTEGER,
            retention_json TEXT,
            public_metrics_json TEXT,
            derived_json TEXT,
            notes TEXT,
            ingestion_completeness REAL,
            transcript_path TEXT,
            metadata_path TEXT,
            video_path TEXT,
            screenshot_dir TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS screenshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content_id TEXT NOT NULL,
            path TEXT NOT NULL,
            ocr_text TEXT,
            metrics_json TEXT,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS project_runs (
            project_id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            input_ref TEXT,
            project_dir TEXT NOT NULL,
            manifest_path TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """
    )
    return conn


def normalize_url(source: str) -> str:
    if not is_url(source):
        return source
    parsed = urlparse(source)
    path = re.sub(r"/+$", "", parsed.path or "/")
    normalized = parsed._replace(query="", fragment="", path=path)
    return normalized.geturl()


def detect_platform(source: str) -> str:
    if not is_url(source):
        return "local"
    host = urlparse(source).netloc.lower()
    if "instagram.com" in host:
        return "instagram"
    if "tiktok.com" in host:
        return "tiktok"
    return "web"


def extract_platform_post_id(source: str) -> str | None:
    if not is_url(source):
        return None
    parsed = urlparse(source)
    path_parts = [part for part in parsed.path.split("/") if part]
    if "instagram.com" in parsed.netloc.lower():
        for prefix in ("reel", "p", "tv"):
            if prefix in path_parts:
                index = path_parts.index(prefix)
                if index + 1 < len(path_parts):
                    return path_parts[index + 1]
    if "tiktok.com" in parsed.netloc.lower():
        if "video" in path_parts:
            index = path_parts.index("video")
            if index + 1 < len(path_parts):
                return path_parts[index + 1]
        query_id = parse_qs(parsed.query).get("share_item_id")
        if query_id:
            return query_id[0]
    return None


def compute_content_id(platform: str, source: str, post_id: str | None) -> str:
    if post_id:
        return f"{platform}_{slugify(post_id)}"
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:12]
    return f"{platform}_{digest}"


def parse_compact_number(raw: str) -> int | None:
    token = raw.strip().replace(",", "").replace(" ", "")
    match = re.match(r"(?i)^(\d+(?:\.\d+)?)([kmb])?$", token)
    if not match:
        return None
    value = float(match.group(1))
    suffix = (match.group(2) or "").lower()
    multiplier = {"": 1, "k": 1_000, "m": 1_000_000, "b": 1_000_000_000}[suffix]
    return int(round(value * multiplier))


def parse_seconds_value(raw: str) -> float | None:
    token = raw.strip().lower()
    if not token:
        return None
    timestamp_match = re.match(r"^(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)$", token)
    if timestamp_match:
        hours = int(timestamp_match.group(1) or 0)
        minutes = int(timestamp_match.group(2))
        seconds = float(timestamp_match.group(3))
        return (hours * 3600) + (minutes * 60) + seconds
    unit_matches = re.findall(r"(\d+(?:\.\d+)?)\s*([hms])", token)
    if unit_matches:
        total = 0.0
        for value, unit in unit_matches:
            number = float(value)
            if unit == "h":
                total += number * 3600
            elif unit == "m":
                total += number * 60
            else:
                total += number
        return total
    seconds_match = re.match(r"^(\d+(?:\.\d+)?)\s*(?:sec|secs|second|seconds|s)$", token)
    if seconds_match:
        return float(seconds_match.group(1))
    return None


def parse_percent_value(raw: str) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*%", raw)
    if not match:
        return None
    return round(float(match.group(1)) / 100.0, 4)


def ydl_options(config: dict[str, Any], download: bool, outtmpl: str | None = None) -> dict[str, Any]:
    options: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }
    if not download:
        options["skip_download"] = True
    if outtmpl is not None:
        options["outtmpl"] = outtmpl
    ytdlp_config = config.get("yt_dlp", {})
    cookies_from_browser = ytdlp_config.get("cookies_from_browser") or ""
    if cookies_from_browser:
        options["cookiesfrombrowser"] = (cookies_from_browser,)
    cookie_file = ytdlp_config.get("cookie_file") or ""
    if cookie_file:
        options["cookiefile"] = str(Path(cookie_file).expanduser())
    return options


def extract_public_metadata(source: str, config: dict[str, Any], asset_dir: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    try:
        with YoutubeDL(ydl_options(config, download=False)) as ydl:
            info = ydl.extract_info(source, download=False)
        metadata = {
            "title": info.get("title") or "",
            "uploader": info.get("uploader") or info.get("channel") or "",
            "published_at": datetime.fromtimestamp(info["timestamp"], tz=timezone.utc).isoformat()
            if info.get("timestamp")
            else (info.get("upload_date") or ""),
            "duration_seconds": info.get("duration"),
            "raw_caption": info.get("description") or "",
            "views": info.get("view_count"),
            "likes": info.get("like_count"),
            "comments": info.get("comment_count"),
            "shares": info.get("repost_count") or info.get("share_count"),
            "thumbnail": info.get("thumbnail") or "",
            "extractor": info.get("extractor_key") or info.get("extractor") or "",
            "full_info": info,
        }
        if metadata["thumbnail"]:
            download_thumbnail(metadata["thumbnail"], asset_dir / "thumbnail.jpg")
    except Exception as exc:
        metadata["extract_error"] = str(exc)

    if metadata.get("title"):
        return metadata

    try:
        response = requests.get(
            source,
            timeout=20,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        og_title = soup.find("meta", attrs={"property": "og:title"})
        og_desc = soup.find("meta", attrs={"property": "og:description"})
        og_image = soup.find("meta", attrs={"property": "og:image"})
        if og_title:
            metadata["title"] = og_title.get("content", "")
        if og_desc:
            metadata["raw_caption"] = og_desc.get("content", "")
        if og_image:
            metadata["thumbnail"] = og_image.get("content", "")
            download_thumbnail(metadata["thumbnail"], asset_dir / "thumbnail.jpg")
    except Exception as exc:
        metadata["html_extract_error"] = str(exc)
    return metadata


def download_thumbnail(url: str, destination: Path) -> None:
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        destination.write_bytes(response.content)
    except Exception:
        return


def transcript_segments_from_text(text: str, speech_rate_wpm: float) -> list[TranscriptSegment]:
    sentences = split_sentences(clean_caption_text(text))
    if not sentences:
        cleaned = clean_caption_text(text)
        if not cleaned:
            return []
        seconds = max(2.0, estimate_spoken_seconds(cleaned, speech_rate_wpm))
        return [TranscriptSegment(start_seconds=0.0, end_seconds=seconds, text=cleaned)]
    segments: list[TranscriptSegment] = []
    cursor = 0.0
    for sentence in sentences:
        duration = max(0.8, estimate_spoken_seconds(sentence, speech_rate_wpm))
        segments.append(
            TranscriptSegment(
                start_seconds=round(cursor, 2),
                end_seconds=round(cursor + duration, 2),
                text=sentence,
            )
        )
        cursor += duration
    return segments


def extract_transcript_and_metadata(
    source: str,
    config: dict[str, Any],
    asset_dir: Path,
    speech_rate_wpm: float,
) -> tuple[str, list[TranscriptSegment], str]:
    if is_url(source):
        metadata, segments = fetch_transcript_from_url(
            source,
            language="en",
            model_name="large-v3",
            cache_dir=ROOT / ".script_cache",
        )
        transcript_text = extract_transcript_text(segments)
        transcript_path = asset_dir / "transcript.txt"
        transcript_path.write_text(transcript_text, encoding="utf-8")
        metadata_path = asset_dir / "transcript_metadata.json"
        metadata_path.write_text(json.dumps(asdict(metadata), indent=2), encoding="utf-8")
        return transcript_text, segments, metadata.transcript_source

    source_path = Path(source).expanduser().resolve()
    segments = load_transcript_from_file(source_path, speech_rate_wpm)
    transcript_text = extract_transcript_text(segments)
    transcript_path = asset_dir / "transcript.txt"
    transcript_path.write_text(transcript_text, encoding="utf-8")
    return transcript_text, segments, f"file:{source_path.suffix.lower() or '.txt'}"


def ocr_image_text(path: Path) -> str:
    result = subprocess.run(
        ["tesseract", str(path), "stdout", "--psm", "6"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def extract_metrics_from_text(text: str) -> dict[str, Any]:
    cleaned = text.replace("\r", "\n")
    lines = [re.sub(r"\s+", " ", line).strip() for line in cleaned.splitlines() if line.strip()]
    metrics: dict[str, Any] = {}
    retention_points: dict[str, float] = {}

    def assign_number(label: str, field: str) -> None:
        for line in lines:
            lower = line.lower()
            if label not in lower:
                continue
            for token in re.findall(r"\d+(?:\.\d+)?\s*[kmbKMB]?", line):
                value = parse_compact_number(token)
                if value is not None:
                    metrics[field] = value
                    return

    for label, field in (
        ("views", "views"),
        ("plays", "views"),
        ("likes", "likes"),
        ("comments", "comments"),
        ("shares", "shares"),
        ("saves", "saves"),
        ("accounts reached", "accounts_reached"),
        ("reached", "accounts_reached"),
        ("profile visits", "profile_visits"),
        ("profile activity", "profile_visits"),
        ("follows", "follows"),
    ):
        assign_number(label, field)

    for line in lines:
        lower = line.lower()
        if "average watch time" in lower:
            value = parse_seconds_value(line) or parse_seconds_value(line.split("average watch time", 1)[-1].strip())
            if value is not None:
                metrics["average_watch_time"] = value
        elif "watch time" in lower and "average" not in lower:
            value = parse_seconds_value(line)
            if value is not None:
                metrics["watch_time_total"] = value
        elif "completion rate" in lower:
            value = parse_percent_value(line)
            if value is not None:
                metrics["completion_rate"] = value
        elif "3 sec" in lower or "3-second" in lower:
            value = parse_percent_value(line)
            if value is not None:
                retention_points["3s_hold_rate"] = value
        elif "5 sec" in lower or "5-second" in lower:
            value = parse_percent_value(line)
            if value is not None:
                retention_points["5s_hold_rate"] = value
        elif "25%" in lower and "retention" in lower:
            value = parse_percent_value(line)
            if value is not None:
                retention_points["25pct"] = value
        elif "50%" in lower and "retention" in lower:
            value = parse_percent_value(line)
            if value is not None:
                retention_points["50pct"] = value
        elif "75%" in lower and "retention" in lower:
            value = parse_percent_value(line)
            if value is not None:
                retention_points["75pct"] = value

    if retention_points:
        metrics["retention_points"] = retention_points
    return metrics


def collect_screenshot_metrics(paths: list[Path], asset_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    screenshot_dir = asset_dir / "screenshots"
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    merged_metrics: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    for index, source_path in enumerate(paths, start=1):
        destination = screenshot_dir / f"{index:02d}_{source_path.name}"
        if source_path.resolve() != destination.resolve():
            shutil.copy2(source_path, destination)
        ocr_text = ""
        metrics: dict[str, Any] = {}
        error: str | None = None
        try:
            ocr_text = ocr_image_text(destination)
            metrics = extract_metrics_from_text(ocr_text)
            for key, value in metrics.items():
                if value is not None:
                    merged_metrics[key] = value
        except Exception as exc:
            error = str(exc)
        records.append(
            {
                "path": str(destination),
                "ocr_text": ocr_text,
                "metrics": metrics,
                "error": error,
            }
        )
    return records, merged_metrics


def maybe_capture_session_artifacts(
    url: str,
    asset_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    browser_config = config.get("browser_session", {})
    storage_state_path = browser_config.get("storage_state_path") or ""
    if not storage_state_path:
        return {"status": "skipped", "reason": "no_storage_state"}

    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"status": "failed", "error": f"playwright_unavailable: {exc}"}

    artifacts_dir = asset_dir / "session"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    screenshot_path = artifacts_dir / "page.png"
    html_path = artifacts_dir / "page.html"
    text_path = artifacts_dir / "page.txt"

    try:  # pragma: no cover - network + browser dependent
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=bool(browser_config.get("headless", True)))
            context = browser.new_context(storage_state=str(Path(storage_state_path).expanduser()))
            page = context.new_page()
            page.goto(url, wait_until="networkidle", timeout=45_000)
            page.screenshot(path=str(screenshot_path), full_page=True)
            html = page.content()
            html_path.write_text(html, encoding="utf-8")
            visible_text = page.locator("body").inner_text()
            text_path.write_text(visible_text, encoding="utf-8")
            metrics = extract_metrics_from_text(visible_text)
            context.close()
            browser.close()
            return {
                "status": "captured",
                "screenshot_path": str(screenshot_path),
                "html_path": str(html_path),
                "text_path": str(text_path),
                "metrics": metrics,
            }
    except Exception as exc:
        return {"status": "failed", "error": str(exc)}


def infer_owner_type(
    config: dict[str, Any],
    platform: str,
    uploader: str,
    title: str,
    caption: str,
    transcript_text: str,
    explicit_owner_type: str | None,
) -> str:
    if explicit_owner_type:
        return explicit_owner_type
    handles = {
        handle.lower().strip("@")
        for handle in config.get("first_party_handles", {}).get(platform, [])
        if handle
    }
    normalized_uploader = uploader.lower().strip("@")
    if normalized_uploader in handles:
        combined = " ".join([title, caption, transcript_text]).lower()
        if any(keyword in combined for keyword in ("matrx", "matrix")):
            return "first_party_matrix"
        return "first_party_voice"
    return "external_reference"


def compute_ingestion_completeness(payload: dict[str, Any]) -> float:
    weighted_fields = [
        ("title", 1.0),
        ("duration_seconds", 1.0),
        ("raw_caption", 1.0),
        ("transcript_text", 2.0),
        ("views", 1.0),
        ("likes", 0.5),
        ("comments", 0.5),
        ("shares", 1.0),
        ("saves", 1.0),
        ("average_watch_time", 1.5),
        ("completion_rate", 1.5),
        ("accounts_reached", 0.75),
        ("follows", 0.75),
    ]
    total_weight = sum(weight for _, weight in weighted_fields)
    score = 0.0
    for field, weight in weighted_fields:
        value = payload.get(field)
        if value not in (None, "", {}, []):
            score += weight
    return round(score / total_weight, 3)


def content_topic_guess(title: str, caption: str, transcript_text: str) -> str:
    combined = " ".join([title, caption, transcript_text]).lower()
    if any(keyword in combined for keyword in ("matrx", "matrix", "assistant", "ai coding", "developer tool")):
        return "mtrx-product"
    if any(keyword in combined for keyword in ("startup", "founder", "business", "saas")):
        return "founder-business"
    if any(keyword in combined for keyword in ("coding", "engineer", "developer", "software")):
        return "engineering"
    return "general"


def content_angle_guess(analysis: SourceAnalysis) -> str:
    if analysis.structure_template == "contrarian-problem-solution":
        return "contrarian-reframe"
    if analysis.structure_template == "story-turn-reveal":
        return "story-reveal"
    if analysis.structure_template == "listicle-reframe":
        return "structured-breakdown"
    return "pain-to-payoff"


def serialize_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True)


def upsert_content_item(conn: sqlite3.Connection, payload: dict[str, Any]) -> None:
    fields = [
        "content_id",
        "platform",
        "platform_post_id",
        "owner_type",
        "url",
        "title",
        "uploader",
        "published_at",
        "duration_seconds",
        "raw_caption",
        "transcript_text",
        "hook_text",
        "structure_type",
        "topic",
        "angle",
        "views",
        "likes",
        "comments",
        "shares",
        "saves",
        "watch_time_total",
        "average_watch_time",
        "completion_rate",
        "accounts_reached",
        "follows",
        "profile_visits",
        "retention_json",
        "public_metrics_json",
        "derived_json",
        "notes",
        "ingestion_completeness",
        "transcript_path",
        "metadata_path",
        "video_path",
        "screenshot_dir",
        "created_at",
        "updated_at",
    ]
    values = [payload.get(field) for field in fields]
    placeholders = ", ".join("?" for _ in fields)
    assignments = ", ".join(f"{field}=excluded.{field}" for field in fields[1:])
    conn.execute(
        f"""
        INSERT INTO content_items ({", ".join(fields)})
        VALUES ({placeholders})
        ON CONFLICT(content_id) DO UPDATE SET
            {assignments}
        """,
        values,
    )
    conn.commit()


def replace_screenshot_rows(conn: sqlite3.Connection, content_id: str, screenshots: list[dict[str, Any]]) -> None:
    conn.execute("DELETE FROM screenshots WHERE content_id = ?", (content_id,))
    for record in screenshots:
        conn.execute(
            """
            INSERT INTO screenshots (content_id, path, ocr_text, metrics_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                content_id,
                record["path"],
                record.get("ocr_text", ""),
                serialize_json(record.get("metrics", {})),
                utc_now_iso(),
            ),
        )
    conn.commit()


def row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    payload = dict(row)
    for field in ("retention_json", "public_metrics_json", "derived_json"):
        value = payload.get(field)
        if value:
            payload[field] = json.loads(value)
        else:
            payload[field] = {}
    return payload


def get_content_item(conn: sqlite3.Connection, identifier: str) -> dict[str, Any] | None:
    normalized = normalize_url(identifier)
    row = conn.execute(
        """
        SELECT * FROM content_items
        WHERE content_id = ? OR url = ? OR platform_post_id = ?
        LIMIT 1
        """,
        (identifier, normalized, identifier),
    ).fetchone()
    return row_to_dict(row) if row else None


def list_content_items(conn: sqlite3.Connection, owner_type: str | None = None) -> list[dict[str, Any]]:
    if owner_type:
        rows = conn.execute(
            "SELECT * FROM content_items WHERE owner_type = ? ORDER BY updated_at DESC",
            (owner_type,),
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM content_items ORDER BY updated_at DESC").fetchall()
    return [row_to_dict(row) for row in rows]


def extract_keywords(text: str, limit: int = 12) -> list[str]:
    words = re.findall(r"[A-Za-z0-9']+", text.lower())
    counts = Counter(word for word in words if word not in STOP_WORDS and len(word) > 2)
    return [word for word, _ in counts.most_common(limit)]


def analyze_transcript_for_item(
    transcript_text: str,
    speech_rate_wpm: float,
) -> tuple[SourceAnalysis, dict[str, Any]]:
    segments = transcript_segments_from_text(transcript_text, speech_rate_wpm)
    analysis = analyze_source_transcript(segments, speech_rate_wpm)
    derived = {
        "source_word_count": analysis.source_word_count,
        "sentence_count": analysis.sentence_count,
        "cadence_hint_words_per_line": analysis.cadence_hint_words_per_line,
        "detected_elements": [asdict(element) for element in analysis.detected_elements],
    }
    return analysis, derived


def render_content_review(item: dict[str, Any], screenshots: list[dict[str, Any]]) -> str:
    lines = [f"# {item['content_id']} Review", ""]
    lines.append("## Identity")
    lines.append(f"- Platform: `{item['platform']}`")
    lines.append(f"- Owner type: `{item['owner_type']}`")
    lines.append(f"- URL: `{item['url']}`")
    if item.get("title"):
        lines.append(f"- Title: {item['title']}")
    if item.get("uploader"):
        lines.append(f"- Uploader: `{item['uploader']}`")
    lines.append(f"- Ingestion completeness: `{item.get('ingestion_completeness', 0):.2f}`")
    lines.append("")
    lines.append("## Performance")
    for label, field in (
        ("Views", "views"),
        ("Likes", "likes"),
        ("Comments", "comments"),
        ("Shares", "shares"),
        ("Saves", "saves"),
        ("Average watch time", "average_watch_time"),
        ("Completion rate", "completion_rate"),
        ("Accounts reached", "accounts_reached"),
        ("Follows", "follows"),
    ):
        value = item.get(field)
        if value not in (None, ""):
            lines.append(f"- {label}: `{value}`")
    if item.get("retention_json"):
        lines.append(f"- Retention: `{json.dumps(item['retention_json'], sort_keys=True)}`")
    lines.append("")
    lines.append("## Script Read")
    if item.get("hook_text"):
        lines.append(f"- Hook: {item['hook_text']}")
    if item.get("structure_type"):
        lines.append(f"- Structure: `{item['structure_type']}`")
    if item.get("topic"):
        lines.append(f"- Topic: `{item['topic']}`")
    if item.get("angle"):
        lines.append(f"- Angle: `{item['angle']}`")
    lines.append("")
    lines.append("## Screenshots")
    if not screenshots:
        lines.append("- none")
    for record in screenshots:
        lines.append(f"- `{record['path']}`")
    lines.append("")
    return "\n".join(lines)


def fetch_screenshots_for_item(conn: sqlite3.Connection, content_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT path, ocr_text, metrics_json FROM screenshots WHERE content_id = ? ORDER BY id",
        (content_id,),
    ).fetchall()
    results = []
    for row in rows:
        results.append(
            {
                "path": row["path"],
                "ocr_text": row["ocr_text"],
                "metrics": json.loads(row["metrics_json"]) if row["metrics_json"] else {},
            }
        )
    return results


def write_review_report(content_id: str, review_markdown: str) -> Path:
    review_dir = REPORTS_DIR / "reviews"
    review_dir.mkdir(parents=True, exist_ok=True)
    path = review_dir / f"{content_id}.md"
    path.write_text(review_markdown, encoding="utf-8")
    return path


def safe_ratio(numerator: float | int | None, denominator: float | int | None) -> float | None:
    if numerator in (None, "") or denominator in (None, "", 0):
        return None
    try:
        denominator_value = float(denominator)
        if denominator_value <= 0:
            return None
        return float(numerator) / denominator_value
    except (TypeError, ValueError):
        return None


def percentile_rank(values: list[float], target: float) -> float:
    if not values:
        return 0.5
    less = sum(1 for value in values if value < target)
    equal = sum(1 for value in values if value == target)
    return round((less + (0.5 * equal)) / len(values), 4)


def compute_performance_features(item: dict[str, Any]) -> dict[str, float]:
    views = item.get("views") or 0
    duration = item.get("duration_seconds")
    features: dict[str, float] = {}
    watch_ratio = safe_ratio(item.get("average_watch_time"), duration)
    if watch_ratio is not None:
        features["watch_ratio"] = min(1.5, watch_ratio)
    completion_rate = item.get("completion_rate")
    if completion_rate not in (None, ""):
        features["completion_rate"] = float(completion_rate)
    shares_per_1k = safe_ratio(item.get("shares"), views)
    if shares_per_1k is not None:
        features["shares_per_1k"] = shares_per_1k * 1000
    saves_per_1k = safe_ratio(item.get("saves"), views)
    if saves_per_1k is not None:
        features["saves_per_1k"] = saves_per_1k * 1000
    follows_per_1k = safe_ratio(item.get("follows"), views)
    if follows_per_1k is not None:
        features["follows_per_1k"] = follows_per_1k * 1000
    profile_per_1k = safe_ratio(item.get("profile_visits"), views)
    if profile_per_1k is not None:
        features["profile_visits_per_1k"] = profile_per_1k * 1000
    likes_rate = safe_ratio(item.get("likes"), views)
    if likes_rate is not None:
        features["likes_rate"] = likes_rate
    comments_rate = safe_ratio(item.get("comments"), views)
    if comments_rate is not None:
        features["comments_rate"] = comments_rate
    return features


def compute_reference_engagement(item: dict[str, Any]) -> float:
    views = item.get("views") or 0
    if not views:
        return 0.0
    score = 0.0
    for field, weight in (("likes", 1.0), ("comments", 2.0), ("shares", 3.0), ("saves", 3.0)):
        value = item.get(field) or 0
        score += weight * (value / views)
    return round(score, 4)


def compute_scores(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    weighted_metrics = {
        "watch_ratio": 0.28,
        "completion_rate": 0.22,
        "shares_per_1k": 0.18,
        "saves_per_1k": 0.14,
        "follows_per_1k": 0.10,
        "profile_visits_per_1k": 0.05,
        "likes_rate": 0.02,
        "comments_rate": 0.01,
    }
    feature_values: dict[str, list[float]] = defaultdict(list)
    per_item_features: dict[str, dict[str, float]] = {}
    for item in items:
        features = compute_performance_features(item)
        per_item_features[item["content_id"]] = features
        for key, value in features.items():
            feature_values[key].append(value)

    scores: dict[str, dict[str, Any]] = {}
    for item in items:
        features = per_item_features[item["content_id"]]
        weighted_sum = 0.0
        total_weight = 0.0
        for key, weight in weighted_metrics.items():
            if key not in features:
                continue
            weighted_sum += percentile_rank(feature_values[key], features[key]) * weight
            total_weight += weight
        if total_weight > 0:
            performance_score = round(weighted_sum / total_weight, 4)
        else:
            performance_score = round(compute_reference_engagement(item), 4)
        scores[item["content_id"]] = {
            "performance_score": performance_score,
            "features": features,
            "reference_engagement_score": compute_reference_engagement(item),
        }
    return scores


def build_training_state(items: list[dict[str, Any]], speech_rate_wpm: float) -> dict[str, Any]:
    if not items:
        return {
            "generated_at": utc_now_iso(),
            "counts": {},
            "target_duration_seconds": 30.0,
            "preferred_structures": [],
            "preferred_elements": [],
            "voice_profile": {},
            "top_examples": [],
            "recommendations": [],
        }

    analyzed_items = []
    for item in items:
        transcript_text = item.get("transcript_text") or item.get("raw_caption") or ""
        analysis = None
        if transcript_text:
            analysis, analysis_payload = analyze_transcript_for_item(transcript_text, speech_rate_wpm)
        else:
            analysis_payload = {}
        analyzed_items.append((item, analysis, analysis_payload))

    score_map = compute_scores(items)
    first_party_matrix = [item for item in items if item["owner_type"] == "first_party_matrix"]
    first_party_voice = [item for item in items if item["owner_type"] == "first_party_voice"]
    external_reference = [item for item in items if item["owner_type"] == "external_reference"]

    scored_matrix = sorted(
        first_party_matrix,
        key=lambda entry: score_map[entry["content_id"]]["performance_score"],
        reverse=True,
    )
    duration_candidates = [
        item["duration_seconds"]
        for item in (scored_matrix[:5] or first_party_matrix or items)
        if item.get("duration_seconds")
    ]
    target_duration = round(statistics.median(duration_candidates), 2) if duration_candidates else 30.0

    structure_scores: dict[str, list[float]] = defaultdict(list)
    element_scores: dict[str, list[float]] = defaultdict(list)
    sentence_lengths: list[float] = []
    for item, analysis, _ in analyzed_items:
        if analysis is None:
            continue
        item_score = score_map[item["content_id"]]["performance_score"]
        if item["owner_type"] == "first_party_matrix":
            structure_scores[analysis.structure_template].append(item_score)
            for element in analysis.detected_elements:
                element_scores[element.name].append(item_score)
        if item["owner_type"] == "first_party_voice":
            for sentence in split_sentences(item.get("transcript_text") or ""):
                sentence_lengths.append(word_count(sentence))

    preferred_structures = [
        {
            "name": name,
            "average_score": round(sum(values) / len(values), 4),
            "count": len(values),
        }
        for name, values in structure_scores.items()
    ]
    preferred_structures.sort(key=lambda entry: (entry["average_score"], entry["count"]), reverse=True)

    preferred_elements = [
        {
            "name": name,
            "average_score": round(sum(values) / len(values), 4),
            "count": len(values),
        }
        for name, values in element_scores.items()
    ]
    preferred_elements.sort(key=lambda entry: (entry["average_score"], entry["count"]), reverse=True)

    top_examples = []
    for item in sorted(items, key=lambda entry: score_map[entry["content_id"]]["performance_score"], reverse=True)[:8]:
        top_examples.append(
            {
                "content_id": item["content_id"],
                "title": item.get("title") or item.get("url") or item["content_id"],
                "owner_type": item["owner_type"],
                "performance_score": score_map[item["content_id"]]["performance_score"],
                "structure_type": item.get("structure_type"),
                "hook_text": item.get("hook_text"),
            }
        )

    voice_profile = {
        "median_sentence_words": round(statistics.median(sentence_lengths), 2) if sentence_lengths else 9.0,
        "avg_sentence_words": round(statistics.mean(sentence_lengths), 2) if sentence_lengths else 9.0,
        "item_count": len(first_party_voice),
    }
    recommendations: list[str] = []
    if preferred_structures:
        recommendations.append(
            f"Lean into `{preferred_structures[0]['name']}` structure first because it currently scores best in the Matrix set."
        )
    if preferred_elements:
        recommendations.append(
            f"High-performing hooks most often use `{preferred_elements[0]['name']}` as the opening pattern."
        )
    if target_duration:
        recommendations.append(f"Default target duration is `{target_duration:.1f}s` based on the strongest available examples.")
    if first_party_voice:
        recommendations.append(
            f"Keep spoken cadence near `{voice_profile['median_sentence_words']}` words per sentence to stay close to your natural delivery."
        )
    if external_reference:
        recommendations.append("Use external winners for structure transfer, not product language.")

    return {
        "generated_at": utc_now_iso(),
        "counts": {
            "all_items": len(items),
            "first_party_matrix": len(first_party_matrix),
            "first_party_voice": len(first_party_voice),
            "external_reference": len(external_reference),
        },
        "target_duration_seconds": target_duration,
        "preferred_structures": preferred_structures[:4],
        "preferred_elements": preferred_elements[:5],
        "voice_profile": voice_profile,
        "top_examples": top_examples,
        "recommendations": recommendations,
        "score_map": score_map,
    }


def render_training_report(state: dict[str, Any]) -> str:
    lines = ["# Studio Training Report", ""]
    lines.append("## Dataset")
    for key, value in state.get("counts", {}).items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    lines.append("## Learned Targets")
    lines.append(f"- Target duration: `{state.get('target_duration_seconds', 30.0)}s`")
    voice = state.get("voice_profile", {})
    if voice:
        lines.append(f"- Median sentence words: `{voice.get('median_sentence_words', 9)}`")
        lines.append(f"- Voice items: `{voice.get('item_count', 0)}`")
    lines.append("")
    lines.append("## Preferred Structures")
    for entry in state.get("preferred_structures", []):
        lines.append(
            f"- `{entry['name']}`: average score `{entry['average_score']}` across `{entry['count']}` item(s)"
        )
    lines.append("")
    lines.append("## Preferred Hook Elements")
    for entry in state.get("preferred_elements", []):
        lines.append(
            f"- `{entry['name']}`: average score `{entry['average_score']}` across `{entry['count']}` item(s)"
        )
    lines.append("")
    lines.append("## Top Examples")
    for example in state.get("top_examples", []):
        lines.append(
            f"- `{example['content_id']}` `{example['performance_score']}` `{example.get('owner_type', '')}` {example.get('title', '')}"
        )
    lines.append("")
    lines.append("## Recommendations")
    for entry in state.get("recommendations", []):
        lines.append(f"- {entry}")
    lines.append("")
    return "\n".join(lines)


def save_training_state(state: dict[str, Any]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    TRAINING_STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")
    TRAINING_REPORT_PATH.write_text(render_training_report(state), encoding="utf-8")


def load_training_state() -> dict[str, Any] | None:
    if not TRAINING_STATE_PATH.exists():
        return None
    return json.loads(TRAINING_STATE_PATH.read_text(encoding="utf-8"))


def resolve_brief_text(source: str | None, config: dict[str, Any]) -> str:
    if not source:
        default_brief = config.get("defaults", {}).get("brief_path") or str(DEFAULT_BRIEF_PATH.relative_to(ROOT))
        brief_path = ROOT / default_brief
        return brief_path.read_text(encoding="utf-8")
    candidate_path = Path(source).expanduser()
    if candidate_path.exists():
        return candidate_path.read_text(encoding="utf-8")
    return source


def select_reference_items(
    items: list[dict[str, Any]],
    training_state: dict[str, Any],
    brief_text: str,
    limit: int = 3,
) -> list[dict[str, Any]]:
    brief_keywords = set(extract_keywords(brief_text, limit=16))
    scored: list[tuple[float, dict[str, Any]]] = []
    score_map = training_state.get("score_map", {})
    for item in items:
        transcript = item.get("transcript_text") or ""
        haystack = " ".join(
            [item.get("title") or "", item.get("raw_caption") or "", transcript]
        ).lower()
        overlap = sum(1 for keyword in brief_keywords if keyword in haystack)
        performance_score = score_map.get(item["content_id"], {}).get("performance_score", 0.5)
        owner_bonus = 0.25 if item["owner_type"] == "external_reference" else 0.15
        scored.append((overlap + performance_score + owner_bonus, item))
    scored.sort(key=lambda entry: entry[0], reverse=True)
    return [item for _, item in scored[:limit]]


def pick_creator_profiles(
    creator_profiles: dict[str, CreatorProfile],
    slugs: list[str],
) -> list[CreatorProfile]:
    profiles = []
    for slug in slugs:
        profile = creator_profiles.get(slug)
        if profile is None or profile.research_status != "researched":
            continue
        profiles.append(profile)
    if not profiles:
        for slug in DEFAULT_CREATORS:
            profile = creator_profiles[slug]
            if profile.research_status == "researched":
                profiles.append(profile)
    return profiles


def structure_variants(training_state: dict[str, Any]) -> list[str]:
    preferred = [entry["name"] for entry in training_state.get("preferred_structures", [])]
    variants = preferred + [
        "contrarian-problem-solution",
        "pain-reveal-payoff",
        "story-turn-reveal",
    ]
    deduped = []
    for item in variants:
        if item not in deduped:
            deduped.append(item)
    return deduped[:4]


def profile_variants() -> list[list[str]]:
    return [
        ["alex-hormozi", "leila-hormozi", "codie-sanchez"],
        ["alex-hormozi", "short-form-founder", "leila-hormozi"],
        ["codie-sanchez", "gary-vaynerchuk", "alex-hormozi"],
        ["story-operator", "simon-squibb", "leila-hormozi"],
    ]


def candidate_score(
    generated: dict[str, Any],
    structure_template: str,
    training_state: dict[str, Any],
    brief_text: str,
) -> tuple[float, list[str]]:
    reasons: list[str] = []
    score = 0.0
    structure_scores = {
        entry["name"]: entry["average_score"] for entry in training_state.get("preferred_structures", [])
    }
    structure_score = structure_scores.get(structure_template, 0.55)
    score += structure_score * 0.35
    reasons.append(f"structure:{structure_template}={structure_score:.2f}")

    target_duration = float(training_state.get("target_duration_seconds") or 30.0)
    duration = float(generated["estimated_duration_seconds"])
    duration_score = max(0.0, 1.0 - (abs(duration - target_duration) / max(8.0, target_duration)))
    score += duration_score * 0.20
    reasons.append(f"duration={duration_score:.2f}")

    brief_keywords = extract_keywords(brief_text, limit=16)
    script_lower = generated["script_text"].lower()
    coverage = sum(1 for keyword in brief_keywords if keyword in script_lower)
    coverage_score = min(1.0, coverage / max(6, len(brief_keywords)))
    score += coverage_score * 0.20
    reasons.append(f"coverage={coverage_score:.2f}")

    mean_line_words = statistics.mean(word_count(line) for line in generated["script_lines"])
    clarity_score = max(0.0, 1.0 - (abs(mean_line_words - 8.0) / 8.0))
    score += clarity_score * 0.15
    reasons.append(f"clarity={clarity_score:.2f}")

    hook_line = generated["script_lines"][0].lower() if generated["script_lines"] else ""
    hook_bonus = 0.0
    if any(token in hook_line for token in ("problem", "hidden", "isn't", "missing", "smart")):
        hook_bonus = 0.1
    score += hook_bonus
    reasons.append(f"hook={hook_bonus:.2f}")
    return round(score, 4), reasons


def build_risk_flags(generated: dict[str, Any], brief_text: str) -> list[str]:
    flags: list[str] = []
    first_line = generated["script_lines"][0] if generated["script_lines"] else ""
    if word_count(first_line) > 12:
        flags.append("hook_may_be_too_long")
    if "mtrx" not in generated["script_text"].lower() and "matrix" not in generated["script_text"].lower():
        flags.append("brand_name_missing")
    if generated["estimated_duration_seconds"] > 40:
        flags.append("long_for_reels")
    if len(extract_keywords(brief_text, limit=12)) > 0:
        coverage = sum(
            1 for keyword in extract_keywords(brief_text, limit=12) if keyword in generated["script_text"].lower()
        )
        if coverage < 3:
            flags.append("weak_brief_coverage")
    return flags


def select_brand_reveal_beat(beats: list[dict[str, Any]]) -> int:
    for beat in beats:
        if "mtrx" in beat["text"].lower() or "matrix" in beat["text"].lower():
            return beat["index"]
    return 2 if len(beats) >= 2 else 1


def classify_visual_for_beat(text: str) -> str:
    lower = text.lower()
    if any(keyword in lower for keyword in ("metrics", "dashboard", "governed", "measure", "track")):
        return "proof_screen"
    if any(keyword in lower for keyword in ("model", "learns", "training")):
        return "training"
    if any(keyword in lower for keyword in ("mtrx", "matrix", "missing layer")):
        return "brand_reveal"
    return "builder_realworld"


def build_broll_plan(beats: list[dict[str, Any]]) -> dict[str, Any]:
    plan = []
    brand_beat = select_brand_reveal_beat(beats)
    for beat in beats:
        visual = classify_visual_for_beat(beat["text"])
        if beat["index"] == brand_beat:
            visual = "brand_reveal"
        if visual == "proof_screen":
            query = "analytics dashboard software screen"
        elif visual == "training":
            query = "ai model training data flow"
        elif visual == "brand_reveal":
            query = "matrx logo reveal"
        else:
            query = "developer building software typing closeup"
        plan.append(
            {
                "beat_index": beat["index"],
                "start_seconds": beat["start_seconds"],
                "end_seconds": beat["end_seconds"],
                "text": beat["text"],
                "visual": visual,
                "search_query": query,
            }
        )
    return {
        "brand_reveal_insert_after_beat": brand_beat,
        "beats": plan,
        "recommended_broll_style": "builder",
    }


def build_post_caption(script_lines: list[str]) -> str:
    if not script_lines:
        return "MTRX helps AI coding assistants remember how your team actually works."
    opener = script_lines[0].strip()
    middle = " ".join(script_lines[1:3]).strip()
    close = "Follow for systems that make AI coding feel less forgetful and more usable."
    parts = [opener]
    if middle:
        parts.append(middle)
    parts.append(close)
    return "\n\n".join(parts)


def build_hashtag_recommendations(script_text: str) -> list[dict[str, str]]:
    lower = script_text.lower()
    keys = ["ai", "coding", "agentic", "startup"]
    if "memory" in lower or "context" in lower:
        keys.append("memory")
    if "mtrx" in lower or "matrix" in lower:
        keys.append("matrx")
    seen: set[str] = set()
    suggestions: list[dict[str, str]] = []
    for key in keys:
        for tag in HASHTAG_BANK.get(key, []):
            if tag in seen:
                continue
            seen.add(tag)
            suggestions.append({"tag": tag, "reason": f"Matches the `{key}` angle of the script."})
    return suggestions[:12]


def build_music_recommendations(structure_template: str) -> list[dict[str, str]]:
    return MUSIC_BANK.get(structure_template, MUSIC_BANK["pain-reveal-payoff"])


def build_command_list(
    project_dir: Path,
    broll_plan: dict[str, Any],
    fcpxml: str | None,
    caption_mode: str,
) -> list[dict[str, str]]:
    commands: list[dict[str, str]] = []
    commands.append(
        {
            "label": "Generate MATRX brand reveal if missing",
            "command": "python3 make_matrx_reveal.py",
        }
    )
    commands.append(
        {
            "label": "Generate supporting B-roll montage",
            "command": f"./broll {max(6, round(sum(beat['end_seconds'] - beat['start_seconds'] for beat in broll_plan['beats']) * 0.45, 1))} --style {broll_plan['recommended_broll_style']}",
        }
    )
    if fcpxml:
        commands.append(
            {
                "label": "Generate captions for Final Cut project",
                "command": f"./caption {shlex.quote(fcpxml)} --caption-mode {caption_mode}",
            }
        )
    commands.append(
        {
            "label": "Review the generated project package",
            "command": f"open {shlex.quote(str(project_dir))}",
        }
    )
    return commands


def ensure_brand_reveal(render_if_missing: bool) -> dict[str, Any]:
    if BRAND_REVEAL_MOV.exists() and BRAND_REVEAL_MP4.exists():
        return {"status": "ready", "mov": str(BRAND_REVEAL_MOV), "mp4": str(BRAND_REVEAL_MP4)}
    if not render_if_missing:
        return {"status": "missing", "mov": str(BRAND_REVEAL_MOV), "mp4": str(BRAND_REVEAL_MP4)}
    make_matrx_reveal_main()
    return {"status": "rendered", "mov": str(BRAND_REVEAL_MOV), "mp4": str(BRAND_REVEAL_MP4)}


def maybe_run_caption(fcpxml: str, caption_mode: str) -> dict[str, Any]:
    command = ["./caption", fcpxml, "--caption-mode", caption_mode]
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return {"status": "completed", "stdout": completed.stdout}
    except subprocess.CalledProcessError as exc:
        return {"status": "failed", "error": exc.stderr or exc.stdout}


def maybe_run_broll(seconds: float, style: str) -> dict[str, Any]:
    command = ["./broll", str(seconds), "--style", style]
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return {"status": "completed", "stdout": completed.stdout}
    except subprocess.CalledProcessError as exc:
        return {"status": "failed", "error": exc.stderr or exc.stdout}


def write_project_manifest(project_id: str, kind: str, input_ref: str, project_dir: Path, manifest: dict[str, Any], conn: sqlite3.Connection) -> Path:
    project_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = project_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    conn.execute(
        """
        INSERT INTO project_runs (project_id, kind, input_ref, project_dir, manifest_path, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(project_id) DO UPDATE SET
            kind=excluded.kind,
            input_ref=excluded.input_ref,
            project_dir=excluded.project_dir,
            manifest_path=excluded.manifest_path,
            created_at=excluded.created_at
        """,
        (project_id, kind, input_ref, str(project_dir), str(manifest_path), utc_now_iso()),
    )
    conn.commit()
    return manifest_path


def resolve_project(project_ref: str, conn: sqlite3.Connection) -> tuple[str, Path, dict[str, Any]]:
    candidate = Path(project_ref).expanduser()
    if candidate.is_dir():
        manifest_path = candidate / "manifest.json"
        if not manifest_path.exists():
            raise SystemExit(f"Missing manifest.json in project directory: {candidate}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return manifest["project_id"], candidate, manifest
    row = conn.execute(
        "SELECT project_id, project_dir, manifest_path FROM project_runs WHERE project_id = ? LIMIT 1",
        (project_ref,),
    ).fetchone()
    if row is None:
        raise SystemExit(f"Unknown project: {project_ref}")
    manifest_path = Path(row["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return row["project_id"], Path(row["project_dir"]), manifest


def save_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def project_id_for_input(source: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{slugify(source)[:32]}_{timestamp}"


def command_ingest(args: argparse.Namespace) -> int:
    config = load_config()
    conn = get_connection()
    source = normalize_url(args.source)
    platform = detect_platform(source)
    post_id = extract_platform_post_id(source)
    content_id = compute_content_id(platform, source, post_id)
    asset_dir = ASSETS_DIR / content_id
    asset_dir.mkdir(parents=True, exist_ok=True)

    metadata = extract_public_metadata(source, config, asset_dir) if is_url(source) else {}
    transcript_text = ""
    segments: list[TranscriptSegment] = []
    transcript_source = ""
    transcript_error: str | None = None
    transcript_path = asset_dir / "transcript.txt"
    metadata_path = asset_dir / "metadata.json"
    if not args.skip_transcript:
        try:
            transcript_text, segments, transcript_source = extract_transcript_and_metadata(
                source,
                config,
                asset_dir,
                args.speech_rate_wpm,
            )
        except Exception as exc:
            transcript_error = str(exc)
    if transcript_text:
        transcript_path.write_text(transcript_text, encoding="utf-8")

    if metadata:
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    screenshot_paths = [Path(path).expanduser().resolve() for path in (args.screenshots or [])]
    screenshot_records, screenshot_metrics = collect_screenshot_metrics(screenshot_paths, asset_dir) if screenshot_paths else ([], {})
    session_capture = maybe_capture_session_artifacts(source, asset_dir, config) if args.capture_session and is_url(source) else {"status": "skipped"}

    title = metadata.get("title") or (Path(source).stem if not is_url(source) else content_id)
    uploader = metadata.get("uploader") or ""
    raw_caption = metadata.get("raw_caption") or ""
    if not transcript_text and raw_caption:
        transcript_text = raw_caption
        segments = transcript_segments_from_text(transcript_text, args.speech_rate_wpm)
        transcript_source = "caption_fallback"

    analysis, analysis_payload = analyze_transcript_for_item(transcript_text, args.speech_rate_wpm) if transcript_text else (None, {})

    merged_metrics = {}
    public_metrics = {
        key: value
        for key, value in {
            "views": metadata.get("views"),
            "likes": metadata.get("likes"),
            "comments": metadata.get("comments"),
            "shares": metadata.get("shares"),
        }.items()
        if value not in (None, "")
    }
    merged_metrics.update(public_metrics)
    merged_metrics.update(screenshot_metrics)
    merged_metrics.update(session_capture.get("metrics", {}))

    owner_type = infer_owner_type(
        config=config,
        platform=platform,
        uploader=uploader,
        title=title,
        caption=raw_caption,
        transcript_text=transcript_text,
        explicit_owner_type=args.owner_type,
    )
    published_at = metadata.get("published_at") or utc_now_iso()
    payload = {
        "content_id": content_id,
        "platform": platform,
        "platform_post_id": post_id,
        "owner_type": owner_type,
        "url": source,
        "title": title,
        "uploader": uploader,
        "published_at": published_at,
        "duration_seconds": metadata.get("duration_seconds"),
        "raw_caption": raw_caption,
        "transcript_text": transcript_text,
        "hook_text": analysis.hook_text if analysis else "",
        "structure_type": analysis.structure_template if analysis else "",
        "topic": content_topic_guess(title, raw_caption, transcript_text),
        "angle": content_angle_guess(analysis) if analysis else "",
        "views": merged_metrics.get("views"),
        "likes": merged_metrics.get("likes"),
        "comments": merged_metrics.get("comments"),
        "shares": merged_metrics.get("shares"),
        "saves": merged_metrics.get("saves"),
        "watch_time_total": merged_metrics.get("watch_time_total"),
        "average_watch_time": merged_metrics.get("average_watch_time"),
        "completion_rate": merged_metrics.get("completion_rate"),
        "accounts_reached": merged_metrics.get("accounts_reached"),
        "follows": merged_metrics.get("follows"),
        "profile_visits": merged_metrics.get("profile_visits"),
        "retention_json": serialize_json(merged_metrics.get("retention_points", {})),
        "public_metrics_json": serialize_json(
            {
                "metadata": metadata,
                "screenshot_metrics": screenshot_metrics,
                "session_capture": session_capture,
                "transcript_source": transcript_source,
                "transcript_error": transcript_error,
            }
        ),
        "derived_json": serialize_json(
            {
                "analysis": analysis_payload,
                "transcript_source": transcript_source,
                "transcript_error": transcript_error,
            }
        ),
        "notes": args.notes or "",
        "ingestion_completeness": 0.0,
        "transcript_path": str(transcript_path) if transcript_path.exists() else "",
        "metadata_path": str(metadata_path) if metadata_path.exists() else "",
        "video_path": "",
        "screenshot_dir": str((asset_dir / "screenshots")) if screenshot_paths else "",
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
    }
    payload["ingestion_completeness"] = compute_ingestion_completeness(payload)

    upsert_content_item(conn, payload)
    replace_screenshot_rows(conn, content_id, screenshot_records)

    item = get_content_item(conn, content_id)
    assert item is not None
    review_path = write_review_report(content_id, render_content_review(item, screenshot_records))
    print(f"Ingested: {content_id}")
    print(f"Review: {review_path}")
    print(f"Completeness: {payload['ingestion_completeness']:.2f}")
    return 0


def command_train(args: argparse.Namespace) -> int:
    conn = get_connection()
    items = list_content_items(conn, owner_type=args.owner_type)
    state = build_training_state(items, args.speech_rate_wpm)
    save_training_state(state)
    print(f"Training state: {TRAINING_STATE_PATH}")
    print(f"Training report: {TRAINING_REPORT_PATH}")
    print(f"Items analyzed: {state['counts'].get('all_items', 0)}")
    return 0


def generate_candidate_scripts(
    brief_text: str,
    reference_items: list[dict[str, Any]],
    training_state: dict[str, Any],
    creator_profiles: dict[str, CreatorProfile],
    speech_rate_wpm: float,
) -> list[dict[str, Any]]:
    claims = extract_brief_claims(brief_text)
    candidates: list[dict[str, Any]] = []
    if not reference_items:
        reference_items = [
            {
                "content_id": "brief_reference",
                "title": "Brief Reference",
                "transcript_text": brief_text,
                "owner_type": "first_party_matrix",
            }
        ]
    for reference in reference_items:
        source_analysis, _ = analyze_transcript_for_item(
            reference.get("transcript_text") or reference.get("raw_caption") or brief_text,
            speech_rate_wpm,
        )
        for structure_template in structure_variants(training_state):
            adapted_analysis = replace(source_analysis, structure_template=structure_template)
            for slugs in profile_variants():
                profiles = pick_creator_profiles(creator_profiles, slugs)
                if not profiles:
                    continue
                generated = generate_short_form_script(
                    brief_claims=claims,
                    source_analysis=adapted_analysis,
                    active_profiles=profiles,
                    min_seconds=20.0,
                    max_seconds=45.0,
                    speech_rate_wpm=speech_rate_wpm,
                )
                score, reasons = candidate_score(
                    generated=generated,
                    structure_template=structure_template,
                    training_state=training_state,
                    brief_text=brief_text,
                )
                candidates.append(
                    {
                        "reference_content_id": reference["content_id"],
                        "reference_title": reference.get("title") or reference["content_id"],
                        "structure_template": structure_template,
                        "creator_slugs": [profile.slug for profile in profiles],
                        "creator_names": [profile.display_name for profile in profiles],
                        "claims": asdict(claims),
                        "source_analysis": {
                            "hook_text": adapted_analysis.hook_text,
                            "structure_template": adapted_analysis.structure_template,
                            "detected_elements": [asdict(element) for element in adapted_analysis.detected_elements],
                        },
                        "generated": generated,
                        "score": score,
                        "score_reasons": reasons,
                        "risk_flags": build_risk_flags(generated, brief_text),
                    }
                )
    candidates.sort(key=lambda entry: entry["score"], reverse=True)
    deduped: list[dict[str, Any]] = []
    seen_scripts: set[str] = set()
    for candidate in candidates:
        script_key = candidate["generated"]["script_text"].lower()
        if script_key in seen_scripts:
            continue
        seen_scripts.add(script_key)
        deduped.append(candidate)
    return deduped[:5]


def render_project_markdown(project_manifest: dict[str, Any]) -> str:
    recommended = project_manifest["recommended_candidate"]
    lines = [f"# {project_manifest['project_id']}", ""]
    lines.append("## Recommended Script")
    lines.append("```text")
    lines.extend(recommended["generated"]["script_lines"])
    lines.append("```")
    lines.append("")
    lines.append(f"Estimated duration: `{recommended['generated']['estimated_duration_seconds']}s`")
    lines.append(f"Structure: `{recommended['structure_template']}`")
    lines.append(f"Creators: `{', '.join(recommended['creator_slugs'])}`")
    lines.append("")
    lines.append("## Risk Flags")
    if recommended["risk_flags"]:
        for flag in recommended["risk_flags"]:
            lines.append(f"- {flag}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Alternate Hooks")
    for hook in recommended["generated"]["hook_options"]:
        lines.append(f"- {hook}")
    lines.append("")
    lines.append("## Commands")
    for command in project_manifest["commands"]:
        lines.append(f"- `{command['label']}`: `{command['command']}`")
    lines.append("")
    return "\n".join(lines)


def build_project_package(
    conn: sqlite3.Connection,
    input_ref: str,
    brief_text: str,
    training_state: dict[str, Any],
    fcpxml: str | None,
    caption_mode: str,
    render_brand_reveal: bool,
    build_broll: bool,
    run_caption: bool,
    speech_rate_wpm: float,
) -> tuple[str, Path, dict[str, Any]]:
    creator_profiles = load_creator_profiles(DEFAULT_CREATOR_PROFILES_PATH)
    items = list_content_items(conn)
    reference_items = select_reference_items(items, training_state, brief_text)
    candidates = generate_candidate_scripts(
        brief_text=brief_text,
        reference_items=reference_items,
        training_state=training_state,
        creator_profiles=creator_profiles,
        speech_rate_wpm=speech_rate_wpm,
    )
    if not candidates:
        raise SystemExit("Could not generate any candidate scripts.")
    recommended = candidates[0]
    broll_plan = build_broll_plan(recommended["generated"]["beats"])
    hashtags = build_hashtag_recommendations(recommended["generated"]["script_text"])
    music = build_music_recommendations(recommended["structure_template"])
    brand_reveal = ensure_brand_reveal(render_if_missing=render_brand_reveal)
    caption_result = maybe_run_caption(fcpxml, caption_mode) if (fcpxml and run_caption) else {"status": "skipped"}
    broll_result = (
        maybe_run_broll(
            seconds=max(6.0, round(recommended["generated"]["estimated_duration_seconds"] * 0.4, 1)),
            style=broll_plan["recommended_broll_style"],
        )
        if build_broll
        else {"status": "skipped"}
    )
    project_id = project_id_for_input(input_ref)
    project_dir = PROJECTS_DIR / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    commands = build_command_list(
        project_dir=project_dir,
        broll_plan=broll_plan,
        fcpxml=fcpxml,
        caption_mode=caption_mode,
    )
    project_manifest = {
        "project_id": project_id,
        "input_ref": input_ref,
        "created_at": utc_now_iso(),
        "training_state_path": str(TRAINING_STATE_PATH) if TRAINING_STATE_PATH.exists() else "",
        "reference_items": [
            {
                "content_id": item["content_id"],
                "title": item.get("title") or item["content_id"],
                "owner_type": item["owner_type"],
            }
            for item in reference_items
        ],
        "recommended_candidate": recommended,
        "alternate_candidates": candidates[1:4],
        "broll_plan": broll_plan,
        "post_caption": build_post_caption(recommended["generated"]["script_lines"]),
        "hashtags": hashtags,
        "music_recommendations": music,
        "brand_reveal": brand_reveal,
        "caption_result": caption_result,
        "broll_result": broll_result,
        "commands": commands,
        "fcpxml": fcpxml or "",
        "caption_mode": caption_mode,
    }
    save_markdown(project_dir / "project.md", render_project_markdown(project_manifest))
    save_markdown(
        project_dir / "script.txt",
        "\n".join(recommended["generated"]["script_lines"]),
    )
    (project_dir / "beats.json").write_text(
        json.dumps(recommended["generated"]["beats"], indent=2),
        encoding="utf-8",
    )
    (project_dir / "broll_plan.json").write_text(json.dumps(broll_plan, indent=2), encoding="utf-8")
    (project_dir / "posting_package.json").write_text(
        json.dumps(
            {
                "post_caption": project_manifest["post_caption"],
                "hashtags": hashtags,
                "music_recommendations": music,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    commands_path = project_dir / "commands.sh"
    commands_path.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n\n"
        + "\n".join(command["command"] for command in commands),
        encoding="utf-8",
    )
    commands_path.chmod(0o755)
    manifest_path = write_project_manifest(project_id, "project", input_ref, project_dir, project_manifest, conn)
    return project_id, project_dir, project_manifest | {"manifest_path": str(manifest_path)}


def command_ideate(args: argparse.Namespace) -> int:
    conn = get_connection()
    training_state = load_training_state() or build_training_state(list_content_items(conn), args.speech_rate_wpm)
    if not TRAINING_STATE_PATH.exists():
        save_training_state(training_state)
    brief_text = resolve_brief_text(args.source, load_config())
    project_id, project_dir, manifest = build_project_package(
        conn=conn,
        input_ref=args.source,
        brief_text=brief_text,
        training_state=training_state,
        fcpxml=None,
        caption_mode="instagram-variable",
        render_brand_reveal=args.render_brand_reveal,
        build_broll=False,
        run_caption=False,
        speech_rate_wpm=args.speech_rate_wpm,
    )
    print(f"Project: {project_id}")
    print(f"Directory: {project_dir}")
    print(f"Recommended script: {project_dir / 'script.txt'}")
    print(f"Manifest: {manifest['manifest_path']}")
    return 0


def command_produce(args: argparse.Namespace) -> int:
    config = load_config()
    conn = get_connection()
    training_state = load_training_state() or build_training_state(list_content_items(conn), args.speech_rate_wpm)
    if not TRAINING_STATE_PATH.exists():
        save_training_state(training_state)
    brief_text = resolve_brief_text(args.source, config)
    project_id, project_dir, manifest = build_project_package(
        conn=conn,
        input_ref=args.source,
        brief_text=brief_text,
        training_state=training_state,
        fcpxml=args.fcpxml,
        caption_mode=args.caption_mode or config.get("defaults", {}).get("caption_mode", "instagram-variable"),
        render_brand_reveal=args.render_brand_reveal,
        build_broll=args.build_broll,
        run_caption=args.run_caption,
        speech_rate_wpm=args.speech_rate_wpm,
    )
    print(f"Project: {project_id}")
    print(f"Directory: {project_dir}")
    print(f"Posting package: {project_dir / 'posting_package.json'}")
    print(f"Manifest: {manifest['manifest_path']}")
    return 0


def command_package(args: argparse.Namespace) -> int:
    conn = get_connection()
    _, project_dir, manifest = resolve_project(args.project, conn)
    posting_package = {
        "post_caption": manifest["post_caption"],
        "hashtags": manifest["hashtags"],
        "music_recommendations": manifest["music_recommendations"],
        "recommended_script": manifest["recommended_candidate"]["generated"]["script_lines"],
    }
    json_path = project_dir / "posting_package.json"
    md_path = project_dir / "posting_package.md"
    json_path.write_text(json.dumps(posting_package, indent=2), encoding="utf-8")
    markdown_lines = ["# Posting Package", "", "## Caption", manifest["post_caption"], "", "## Hashtags"]
    for entry in manifest["hashtags"]:
        markdown_lines.append(f"- {entry['tag']}: {entry['reason']}")
    markdown_lines.append("")
    markdown_lines.append("## Music Recommendations")
    for entry in manifest["music_recommendations"]:
        markdown_lines.append(f"- `{entry['query']}`: {entry['reason']}")
    md_path.write_text("\n".join(markdown_lines), encoding="utf-8")
    print(f"Posting package: {json_path}")
    print(f"Human-readable package: {md_path}")
    return 0


def command_review(args: argparse.Namespace) -> int:
    conn = get_connection()
    item = get_content_item(conn, args.identifier)
    if item is None:
        raise SystemExit(f"Content not found: {args.identifier}")
    screenshots = fetch_screenshots_for_item(conn, item["content_id"])
    review_path = write_review_report(item["content_id"], render_content_review(item, screenshots))
    print(f"Review: {review_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Studio orchestration for MATRX short-form content operations.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Ingest a Reel/TikTok link or local transcript.")
    ingest.add_argument("source", help="Instagram Reel link, TikTok link, or local transcript file.")
    ingest.add_argument(
        "--screenshots",
        nargs="*",
        help="Optional insights screenshots used for OCR metric extraction.",
    )
    ingest.add_argument(
        "--owner-type",
        choices=["first_party_matrix", "first_party_voice", "external_reference"],
        help="Override owner type inference.",
    )
    ingest.add_argument("--notes", help="Optional notes stored with the content item.")
    ingest.add_argument(
        "--skip-transcript",
        action="store_true",
        help="Skip transcript extraction and rely on caption text only.",
    )
    ingest.add_argument(
        "--capture-session",
        action="store_true",
        help="Try Playwright page capture using the configured logged-in storage state.",
    )
    ingest.add_argument(
        "--speech-rate-wpm",
        type=float,
        default=DEFAULT_SPEECH_RATE_WPM,
        help="Speech rate used for transcript analysis fallback.",
    )

    train = subparsers.add_parser("train", help="Analyze the local dataset and build a training state.")
    train.add_argument(
        "--owner-type",
        choices=["first_party_matrix", "first_party_voice", "external_reference"],
        help="Restrict training to one owner type.",
    )
    train.add_argument(
        "--speech-rate-wpm",
        type=float,
        default=DEFAULT_SPEECH_RATE_WPM,
        help="Speech rate used for transcript analysis.",
    )

    ideate = subparsers.add_parser("ideate", help="Generate a script and project package from a brief.")
    ideate.add_argument("source", nargs="?", help="Brief file path or inline brief text.")
    ideate.add_argument(
        "--render-brand-reveal",
        action="store_true",
        help="Render the MATRX brand reveal if it is missing.",
    )
    ideate.add_argument(
        "--speech-rate-wpm",
        type=float,
        default=DEFAULT_SPEECH_RATE_WPM,
        help="Speech rate used for candidate generation.",
    )

    produce = subparsers.add_parser("produce", help="Generate a fuller content package and optional edit helpers.")
    produce.add_argument("source", nargs="?", help="Brief file path or inline brief text.")
    produce.add_argument(
        "--fcpxml",
        help="Optional Final Cut XML or bundle Info.fcpxml path used for caption generation.",
    )
    produce.add_argument(
        "--caption-mode",
        choices=["classic", "instagram-variable"],
        help="Caption mode passed through to the caption tool.",
    )
    produce.add_argument(
        "--run-caption",
        action="store_true",
        help="Run the caption tool immediately when --fcpxml is provided.",
    )
    produce.add_argument(
        "--build-broll",
        action="store_true",
        help="Run the B-roll generator immediately for the project package.",
    )
    produce.add_argument(
        "--render-brand-reveal",
        action="store_true",
        help="Render the MATRX brand reveal if it is missing.",
    )
    produce.add_argument(
        "--speech-rate-wpm",
        type=float,
        default=DEFAULT_SPEECH_RATE_WPM,
        help="Speech rate used for candidate generation.",
    )

    package = subparsers.add_parser("package", help="Refresh or regenerate the posting package for a project.")
    package.add_argument("project", help="Project ID or project directory.")

    review = subparsers.add_parser("review", help="Write a review summary for an ingested content item.")
    review.add_argument("identifier", help="Content ID, platform post ID, or normalized URL.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "ingest":
        return command_ingest(args)
    if args.command == "train":
        return command_train(args)
    if args.command == "ideate":
        return command_ideate(args)
    if args.command == "produce":
        return command_produce(args)
    if args.command == "package":
        return command_package(args)
    if args.command == "review":
        return command_review(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
