from __future__ import annotations

import argparse
import html
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from yt_dlp import YoutubeDL

try:
    from faster_whisper import WhisperModel
except ImportError:  # pragma: no cover - optional runtime dependency
    WhisperModel = None


ROOT = Path(__file__).resolve().parent
DEFAULT_BRIEF_PATH = ROOT / "briefs" / "matrx.md"
DEFAULT_CREATOR_PROFILES_PATH = ROOT / "script_creator_profiles.json"
DEFAULT_OUTPUT_DIR = ROOT / "analysis" / "scripts"
DEFAULT_TARGET_SECONDS = 30.0
DEFAULT_MIN_SECONDS = 20.0
DEFAULT_MAX_SECONDS = 45.0
DEFAULT_SPEECH_RATE_WPM = 185.0
DEFAULT_TRANSCRIBE_MODEL = "large-v3"
DEFAULT_CREATORS = ("alex-hormozi", "leila-hormozi", "codie-sanchez")

HOOK_PATTERNS = {
    "direct_address": re.compile(r"\b(you|your|if you)\b", re.IGNORECASE),
    "contrarian": re.compile(r"\b(wrong|stop|never|nobody|isn't|is not|don't|doesn't)\b", re.IGNORECASE),
    "curiosity": re.compile(r"\b(why|how|what if|here's|the reason|this is why)\b", re.IGNORECASE),
    "pain": re.compile(r"\b(problem|mistake|fail|broke|friction|hidden tax|cost|chaos|rework|forget)\b", re.IGNORECASE),
    "proof": re.compile(r"\b(i|we)\b.{0,24}\b(built|made|learned|tested|scaled|saw|found)\b", re.IGNORECASE),
    "specificity": re.compile(r"\b\d+(?:\.\d+)?\b|%|\$"),
    "reframe": re.compile(r"\b(but|instead|actually|the shift|the problem isn't)\b", re.IGNORECASE),
    "list_frame": re.compile(r"\b(one|two|three|first|second|third)\b", re.IGNORECASE),
}

PAIN_WORDS = {
    "amnesiac",
    "chaos",
    "cost",
    "debugging",
    "forget",
    "forgot",
    "forgetful",
    "friction",
    "guessy",
    "hidden",
    "mistake",
    "problem",
    "redebugging",
    "re-explaining",
    "re-explain",
    "re-litigating",
    "re-stating",
    "restate",
    "rework",
    "risk",
    "slow",
    "tax",
}
SOLUTION_WORDS = {
    "assistant",
    "assistants",
    "context",
    "governed",
    "history",
    "infrastructure",
    "layer",
    "learn",
    "memory",
    "mtrx",
    "organization",
    "outcomes",
    "project",
    "rules",
    "team",
}
PAYOFF_WORDS = {
    "compound",
    "confidence",
    "consistent",
    "faster",
    "governed",
    "measurable",
    "memory",
    "quality",
    "scale",
    "safer",
    "speed",
}
FILLER_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "but",
    "for",
    "from",
    "how",
    "if",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "so",
    "that",
    "the",
    "their",
    "them",
    "they",
    "this",
    "to",
    "we",
    "what",
    "when",
    "with",
    "you",
    "your",
}


@dataclass(frozen=True)
class TranscriptSegment:
    start_seconds: float
    end_seconds: float
    text: str


@dataclass(frozen=True)
class SourceMetadata:
    source: str
    title: str
    uploader: str
    duration_seconds: float | None
    transcript_source: str


@dataclass(frozen=True)
class CreatorProfile:
    slug: str
    display_name: str
    research_status: str
    platform_fit: str
    voice_traits: list[str]
    hook_patterns: list[str]
    script_rules: list[str]
    avoid: list[str]


@dataclass(frozen=True)
class ViralElement:
    name: str
    score: float
    evidence: list[str]


@dataclass(frozen=True)
class SourceAnalysis:
    structure_template: str
    hook_text: str
    detected_elements: list[ViralElement]
    source_word_count: int
    estimated_source_seconds: float
    sentence_count: int
    cadence_hint_words_per_line: int


@dataclass(frozen=True)
class BriefClaims:
    problem: str
    solution: str
    mechanism: str
    payoff: str
    positioning: str


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "script"


def is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def clean_caption_text(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\{\\an\d+\}", " ", text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    raw_parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [part.strip() for part in raw_parts if part.strip()]


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", text))


def estimate_spoken_seconds(text: str, speech_rate_wpm: float = DEFAULT_SPEECH_RATE_WPM) -> float:
    return (word_count(text) / max(120.0, speech_rate_wpm)) * 60.0


def parse_timestamp(value: str) -> float:
    value = value.strip().replace(",", ".")
    parts = value.split(":")
    if len(parts) == 2:
        hours = 0
        minutes = int(parts[0])
        seconds = float(parts[1])
    elif len(parts) == 3:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
    else:
        raise ValueError(f"Unsupported timestamp: {value}")
    return (hours * 3600) + (minutes * 60) + seconds


def merge_segments(segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
    merged: list[TranscriptSegment] = []
    for segment in segments:
        text = clean_caption_text(segment.text)
        if not text:
            continue
        if merged and merged[-1].text == text and segment.start_seconds <= merged[-1].end_seconds + 0.25:
            merged[-1] = TranscriptSegment(
                start_seconds=merged[-1].start_seconds,
                end_seconds=max(merged[-1].end_seconds, segment.end_seconds),
                text=merged[-1].text,
            )
            continue
        merged.append(
            TranscriptSegment(
                start_seconds=segment.start_seconds,
                end_seconds=max(segment.end_seconds, segment.start_seconds + 0.1),
                text=text,
            )
        )
    return merged


def parse_vtt(content: str) -> list[TranscriptSegment]:
    blocks = re.split(r"\n\s*\n", content.replace("\r\n", "\n"))
    segments: list[TranscriptSegment] = []
    for block in blocks:
        lines = [line.strip("\ufeff") for line in block.splitlines() if line.strip()]
        if not lines or lines[0].upper().startswith(("WEBVTT", "NOTE", "STYLE")):
            continue
        cue_line_index = 0
        if "-->" not in lines[cue_line_index] and len(lines) > 1:
            cue_line_index = 1
        if cue_line_index >= len(lines) or "-->" not in lines[cue_line_index]:
            continue
        start_text, end_text = [part.strip().split(" ")[0] for part in lines[cue_line_index].split("-->")[:2]]
        payload = clean_caption_text(" ".join(lines[cue_line_index + 1 :]))
        if not payload:
            continue
        segments.append(
            TranscriptSegment(
                start_seconds=parse_timestamp(start_text),
                end_seconds=parse_timestamp(end_text),
                text=payload,
            )
        )
    return merge_segments(segments)


def parse_srt(content: str) -> list[TranscriptSegment]:
    blocks = re.split(r"\n\s*\n", content.replace("\r\n", "\n"))
    segments: list[TranscriptSegment] = []
    for block in blocks:
        lines = [line for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue
        cue_line = lines[1] if "-->" in lines[1] else lines[0]
        if "-->" not in cue_line:
            continue
        start_text, end_text = [part.strip().split(" ")[0] for part in cue_line.split("-->")[:2]]
        payload_lines = lines[2:] if cue_line == lines[1] else lines[1:]
        payload = clean_caption_text(" ".join(payload_lines))
        if not payload:
            continue
        segments.append(
            TranscriptSegment(
                start_seconds=parse_timestamp(start_text),
                end_seconds=parse_timestamp(end_text),
                text=payload,
            )
        )
    return merge_segments(segments)


def parse_json3(content: str) -> list[TranscriptSegment]:
    data = json.loads(content)
    segments: list[TranscriptSegment] = []
    for event in data.get("events", []):
        pieces = event.get("segs") or []
        if not pieces:
            continue
        text = clean_caption_text("".join(piece.get("utf8", "") for piece in pieces))
        if not text:
            continue
        start_seconds = float(event.get("tStartMs", 0)) / 1000.0
        duration_seconds = float(event.get("dDurationMs", 0)) / 1000.0
        if duration_seconds <= 0:
            duration_seconds = max(0.35, word_count(text) / 3.8)
        segments.append(
            TranscriptSegment(
                start_seconds=start_seconds,
                end_seconds=start_seconds + duration_seconds,
                text=text,
            )
        )
    return merge_segments(segments)


def parse_plain_text(content: str, speech_rate_wpm: float) -> list[TranscriptSegment]:
    text = clean_caption_text(content)
    if not text:
        return []
    estimated_seconds = max(2.0, estimate_spoken_seconds(text, speech_rate_wpm))
    return [TranscriptSegment(start_seconds=0.0, end_seconds=estimated_seconds, text=text)]


def load_transcript_from_file(path: Path, speech_rate_wpm: float) -> list[TranscriptSegment]:
    content = read_text(path)
    suffix = path.suffix.lower()
    if suffix == ".vtt":
        return parse_vtt(content)
    if suffix == ".srt":
        return parse_srt(content)
    if suffix == ".json":
        return parse_json3(content)
    return parse_plain_text(content, speech_rate_wpm)


def select_caption_track(info: dict[str, Any], language: str) -> tuple[str, dict[str, Any]] | None:
    language_candidates = [language, language.replace("_", "-"), "en", "en-US"]
    buckets = [
        ("manual_subtitles", info.get("subtitles") or {}),
        ("automatic_captions", info.get("automatic_captions") or {}),
    ]
    rank = {"json3": 0, "srv3": 1, "vtt": 2, "ttml": 3, "srt": 4}
    for transcript_source, tracks_by_language in buckets:
        for candidate in language_candidates:
            tracks = tracks_by_language.get(candidate) or []
            if not tracks:
                continue
            track = min(tracks, key=lambda item: rank.get(item.get("ext", "zzz"), 99))
            return transcript_source, track
    return None


def download_transcript_from_track(track: dict[str, Any]) -> list[TranscriptSegment]:
    response = requests.get(track["url"], timeout=30)
    response.raise_for_status()
    ext = track.get("ext", "").lower()
    if ext == "json3":
        return parse_json3(response.text)
    if ext in {"vtt", "webvtt"}:
        return parse_vtt(response.text)
    if ext == "srt":
        return parse_srt(response.text)
    return parse_plain_text(response.text, DEFAULT_SPEECH_RATE_WPM)


def find_downloaded_audio(cache_dir: Path, stem: str) -> Path | None:
    for path in sorted(cache_dir.glob(f"{stem}.*")):
        if path.suffix in {".part", ".ytdl", ".json"}:
            continue
        return path
    return None


def transcribe_audio(path: Path, model_name: str, language: str) -> list[TranscriptSegment]:
    if WhisperModel is None:
        raise RuntimeError("faster_whisper is not installed in this environment.")
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(str(path), language=language, vad_filter=True)
    transcript_segments: list[TranscriptSegment] = []
    for segment in segments:
        text = clean_caption_text(segment.text)
        if not text:
            continue
        transcript_segments.append(
            TranscriptSegment(
                start_seconds=float(segment.start),
                end_seconds=float(segment.end),
                text=text,
            )
        )
    return merge_segments(transcript_segments)


def fetch_transcript_from_url(
    url: str,
    language: str,
    model_name: str,
    cache_dir: Path,
) -> tuple[SourceMetadata, list[TranscriptSegment]]:
    with YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True, "noplaylist": True}) as ydl:
        info = ydl.extract_info(url, download=False)

    title = info.get("title") or "Untitled Source"
    metadata = SourceMetadata(
        source=url,
        title=title,
        uploader=info.get("uploader") or info.get("channel") or "Unknown",
        duration_seconds=info.get("duration"),
        transcript_source="unavailable",
    )

    selected_track = select_caption_track(info, language)
    if selected_track is not None:
        transcript_source, track = selected_track
        segments = download_transcript_from_track(track)
        return (
            SourceMetadata(
                source=url,
                title=metadata.title,
                uploader=metadata.uploader,
                duration_seconds=metadata.duration_seconds,
                transcript_source=transcript_source,
            ),
            segments,
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = slugify(title)[:48] or "source"
    outtmpl = str(cache_dir / f"{stem}.%(ext)s")
    with YoutubeDL(
        {
            "format": "bestaudio/best",
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "outtmpl": outtmpl,
        }
    ) as ydl:
        ydl.download([url])
    audio_path = find_downloaded_audio(cache_dir, stem)
    if audio_path is None:
        raise RuntimeError("Failed to download audio for fallback transcription.")
    segments = transcribe_audio(audio_path, model_name, language)
    return (
        SourceMetadata(
            source=url,
            title=metadata.title,
            uploader=metadata.uploader,
            duration_seconds=metadata.duration_seconds,
            transcript_source=f"whisper:{model_name}",
        ),
        segments,
    )


def load_creator_profiles(path: Path) -> dict[str, CreatorProfile]:
    raw = json.loads(read_text(path))
    profiles: dict[str, CreatorProfile] = {}
    for slug, payload in raw.items():
        profiles[slug] = CreatorProfile(slug=slug, **payload)
    return profiles


def choose_sentence(sentences: list[str], keywords: set[str], fallback: str) -> str:
    best_sentence = ""
    best_score = -1
    for sentence in sentences:
        lowered = sentence.lower()
        keyword_hits = sum(1 for keyword in keywords if keyword in lowered)
        score = keyword_hits * 12
        score += max(0, 18 - abs(word_count(sentence) - 18))
        if score > best_score:
            best_score = score
            best_sentence = sentence
    return best_sentence or fallback


def extract_brief_claims(brief_text: str) -> BriefClaims:
    sentences = split_sentences(brief_text)
    problem = choose_sentence(
        sentences,
        {"amnesiac", "forget", "re-explain", "re-debug", "re-litigating", "hidden tax", "friction"},
        "AI assistants lose context, so teams keep paying the same rework tax.",
    )
    solution = choose_sentence(
        sentences,
        {"mtrx is", "layer", "assistants", "remember", "rules", "learn"},
        "MTRX is the layer between your team and the assistants you already use.",
    )
    mechanism = choose_sentence(
        sentences,
        {"context", "memory", "surface", "outcomes", "worked", "rules", "project"},
        "It carries forward context, keeps the assistant inside your rules, and learns from what worked.",
    )
    payoff = choose_sentence(
        sentences,
        {"scale", "quality", "compounds", "confidence", "faster", "measurable", "managed"},
        "So AI stops acting like a clever chat and starts behaving like company infrastructure.",
    )
    positioning = choose_sentence(
        sentences,
        {"managed company capability", "governed", "measurable", "grounded"},
        "MTRX turns AI coding assistants into a managed company capability.",
    )
    return BriefClaims(
        problem=clean_caption_text(problem),
        solution=clean_caption_text(solution),
        mechanism=clean_caption_text(mechanism),
        payoff=clean_caption_text(payoff),
        positioning=clean_caption_text(positioning),
    )


def extract_transcript_text(segments: list[TranscriptSegment]) -> str:
    return " ".join(segment.text for segment in segments).strip()


def detect_viral_elements(sentences: list[str]) -> list[ViralElement]:
    opening = " ".join(sentences[:2]).strip()
    elements: list[ViralElement] = []
    for name, pattern in HOOK_PATTERNS.items():
        hits = [sentence for sentence in sentences[:4] if pattern.search(sentence)]
        if hits:
            score = min(1.0, 0.35 + (0.2 * len(hits)))
            if pattern.search(opening):
                score += 0.15
            elements.append(
                ViralElement(
                    name=name,
                    score=round(min(score, 1.0), 2),
                    evidence=hits[:2],
                )
            )
    elements.sort(key=lambda item: item.score, reverse=True)
    return elements


def choose_structure_template(sentences: list[str], elements: list[ViralElement]) -> str:
    opening = " ".join(sentences[:2]).lower()
    if any(element.name == "list_frame" and element.score >= 0.5 for element in elements):
        return "listicle-reframe"
    if any(element.name == "contrarian" and element.score >= 0.5 for element in elements):
        return "contrarian-problem-solution"
    if "then" in opening or "realized" in opening or "turned out" in opening:
        return "story-turn-reveal"
    return "pain-reveal-payoff"


def analyze_source_transcript(segments: list[TranscriptSegment], speech_rate_wpm: float) -> SourceAnalysis:
    transcript = extract_transcript_text(segments)
    sentences = split_sentences(transcript)
    opening_sentences = sentences[:2] if sentences else [transcript]
    hook_text = clean_caption_text(" ".join(opening_sentences))[:220]
    elements = detect_viral_elements(sentences or [transcript])
    cadence_samples = [word_count(segment.text) for segment in segments if segment.text]
    cadence_hint = max(5, min(11, round(sum(cadence_samples[:8]) / max(1, min(8, len(cadence_samples)))))) if cadence_samples else 8
    return SourceAnalysis(
        structure_template=choose_structure_template(sentences, elements),
        hook_text=hook_text,
        detected_elements=elements,
        source_word_count=word_count(transcript),
        estimated_source_seconds=max(
            segments[-1].end_seconds if segments else 0.0,
            estimate_spoken_seconds(transcript, speech_rate_wpm),
        ),
        sentence_count=len(sentences),
        cadence_hint_words_per_line=cadence_hint,
    )


def join_profile_lists(profiles: list[CreatorProfile], field: str) -> list[str]:
    output: list[str] = []
    for profile in profiles:
        output.extend(getattr(profile, field))
    return output


def build_hook_options(
    brief_claims: BriefClaims,
    active_profiles: list[CreatorProfile],
    template: str,
) -> list[str]:
    base_hooks = [
        "AI coding assistants are smart. They're just alone.",
        "AI gives you intelligence. MTRX gives it memory.",
        "The hidden cost of AI coding is the forgetting.",
        "Your team doesn't need another AI chat. It needs a missing layer.",
        "The problem with AI coding assistants isn't intelligence. It's memory.",
        "Everyone wants smarter models. Most teams are missing the system around them.",
    ]
    if template == "contrarian-problem-solution":
        base_hooks.insert(0, "The model isn't the problem. The missing layer is.")
    if template == "listicle-reframe":
        base_hooks.insert(0, "Three things break AI coding at scale. This is the biggest one.")
    profile_hooks = join_profile_lists(active_profiles, "hook_patterns")
    candidates = [clean_caption_text(hook) for hook in base_hooks + profile_hooks]
    deduped: list[str] = []
    seen: set[str] = set()
    for hook in candidates:
        key = hook.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(hook)
    return deduped[:6]


def choose_main_hook(hook_options: list[str], source_analysis: SourceAnalysis) -> str:
    prefer_contrarian = any(element.name == "contrarian" for element in source_analysis.detected_elements[:2])
    prefer_direct = any(element.name == "direct_address" for element in source_analysis.detected_elements[:2])
    for hook in hook_options:
        lower = hook.lower()
        if prefer_contrarian and ("isn't" in lower or "problem" in lower):
            return hook
        if prefer_direct and "your team" in lower:
            return hook
    return hook_options[0]


def line_bank_from_brief(brief_claims: BriefClaims) -> dict[str, str]:
    return {
        "problem": "Every team keeps reloading context and re-solving the same failures.",
        "solution": "MTRX sits between your team and the assistants you already use.",
        "mechanism": "It remembers what matters, keeps them inside your rules, and learns from what actually worked.",
        "payoff": "So AI stops acting like a chat and starts behaving like infrastructure.",
        "positioning": "That turns AI coding into a governed company capability.",
    }


def fit_script_lines_to_duration(
    lines: list[str],
    optional_lines: list[str],
    min_seconds: float,
    max_seconds: float,
    speech_rate_wpm: float,
) -> list[str]:
    working = [line for line in lines if line]
    optional_queue = [line for line in optional_lines if line]
    while estimate_spoken_seconds(" ".join(working), speech_rate_wpm) < min_seconds and optional_queue:
        working.insert(max(2, len(working) - 2), optional_queue.pop(0))
    while estimate_spoken_seconds(" ".join(working), speech_rate_wpm) > max_seconds and len(working) > 5:
        removable_index = max(2, len(working) - 3)
        working.pop(removable_index)
    return working


def break_long_lines(lines: list[str], target_words_per_line: int) -> list[str]:
    output: list[str] = []
    for line in lines:
        words = line.split()
        if len(words) <= target_words_per_line:
            output.append(line.strip())
            continue
        cursor = 0
        while cursor < len(words):
            chunk = words[cursor : cursor + target_words_per_line]
            text = " ".join(chunk).strip()
            if cursor + target_words_per_line < len(words):
                text = text.rstrip(",.;:")
            output.append(text)
            cursor += target_words_per_line
    return output


def assign_beats(lines: list[str], speech_rate_wpm: float) -> list[dict[str, Any]]:
    beats: list[dict[str, Any]] = []
    cursor = 0.0
    for index, line in enumerate(lines):
        duration = max(1.0, estimate_spoken_seconds(line, speech_rate_wpm))
        beats.append(
            {
                "index": index + 1,
                "start_seconds": round(cursor, 2),
                "end_seconds": round(cursor + duration, 2),
                "text": line,
            }
        )
        cursor += duration
    return beats


def generate_short_form_script(
    brief_claims: BriefClaims,
    source_analysis: SourceAnalysis,
    active_profiles: list[CreatorProfile],
    min_seconds: float,
    max_seconds: float,
    speech_rate_wpm: float,
) -> dict[str, Any]:
    hook_options = build_hook_options(brief_claims, active_profiles, source_analysis.structure_template)
    selected_hook = choose_main_hook(hook_options, source_analysis)
    bank = line_bank_from_brief(brief_claims)

    if source_analysis.structure_template == "listicle-reframe":
        core_lines = [
            selected_hook,
            "First, the assistant forgets the real context.",
            "Second, every team invents its own rules in the chat window.",
            "Third, the same failures keep getting solved from scratch.",
            bank["solution"],
            bank["mechanism"],
            bank["payoff"],
        ]
    elif source_analysis.structure_template == "story-turn-reveal":
        core_lines = [
            selected_hook,
            "It looks like a model problem at first.",
            "But the real issue is everything the model can't carry forward on its own.",
            bank["solution"],
            bank["mechanism"],
            bank["payoff"],
        ]
    else:
        core_lines = [
            selected_hook,
            "So teams keep reloading context, re-solving the same failures, and re-paying the same tax.",
            bank["solution"],
            bank["mechanism"],
            bank["positioning"],
            bank["payoff"],
        ]

    optional_lines = [
        "That means less rework, faster onboarding, and fewer guessy answers across the team.",
        "Your engineers keep the assistants they already use. The system just gets smarter around them.",
        "It turns scattered chats into memory tied to the work that actually happened.",
    ]
    fitted_lines = fit_script_lines_to_duration(
        core_lines,
        optional_lines,
        min_seconds=min_seconds,
        max_seconds=max_seconds,
        speech_rate_wpm=speech_rate_wpm,
    )
    line_target = max(6, min(10, source_analysis.cadence_hint_words_per_line))
    final_lines = break_long_lines(fitted_lines, line_target)
    beats = assign_beats(final_lines, speech_rate_wpm)
    script_text = "\n".join(final_lines)
    estimated_seconds = beats[-1]["end_seconds"] if beats else 0.0
    return {
        "selected_hook": selected_hook,
        "hook_options": hook_options[:4],
        "script_lines": final_lines,
        "script_text": script_text,
        "estimated_duration_seconds": round(estimated_seconds, 2),
        "beats": beats,
        "delivery_notes": [
            "Land the hook in one breath. No soft intro before it.",
            "Keep each line punchy enough to survive caption-only viewing.",
            "Treat the middle as one tension arc: hidden tax -> missing layer -> payoff.",
        ],
    }


def render_markdown_report(
    metadata: SourceMetadata,
    source_analysis: SourceAnalysis,
    creator_profiles: list[CreatorProfile],
    skipped_profiles: list[CreatorProfile],
    brief_claims: BriefClaims,
    generated: dict[str, Any],
) -> str:
    lines: list[str] = []
    lines.append(f"# {metadata.title} -> MATRX Short-Form Script")
    lines.append("")
    lines.append("## Source")
    lines.append(f"- Source: `{metadata.source}`")
    lines.append(f"- Uploader: `{metadata.uploader}`")
    if metadata.duration_seconds is not None:
        lines.append(f"- Source duration: `{round(float(metadata.duration_seconds), 2)}s`")
    lines.append(f"- Transcript source: `{metadata.transcript_source}`")
    lines.append("")
    lines.append("## Short-Form Read")
    lines.append(f"- Structure template: `{source_analysis.structure_template}`")
    lines.append(f"- Opening hook observed: {source_analysis.hook_text}")
    detected_names = ", ".join(f"{item.name} ({item.score})" for item in source_analysis.detected_elements[:5]) or "none"
    lines.append(f"- Detected viral elements: {detected_names}")
    lines.append("")
    lines.append("## Creator Blend")
    for profile in creator_profiles:
        lines.append(f"- `{profile.slug}`: {profile.platform_fit}")
    for profile in skipped_profiles:
        lines.append(f"- `{profile.slug}` skipped: needs example clips or transcripts before it can be modeled accurately")
    lines.append("")
    lines.append("## MATRX Angle")
    lines.append(f"- Problem: {brief_claims.problem}")
    lines.append(f"- Solution: {brief_claims.solution}")
    lines.append(f"- Mechanism: {brief_claims.mechanism}")
    lines.append(f"- Payoff: {brief_claims.payoff}")
    lines.append("")
    lines.append("## Hook Options")
    for hook in generated["hook_options"]:
        lines.append(f"- {hook}")
    lines.append("")
    lines.append("## Recommended Script")
    lines.append("```text")
    lines.extend(generated["script_lines"])
    lines.append("```")
    lines.append("")
    lines.append(f"Estimated spoken duration: `{generated['estimated_duration_seconds']}s`")
    lines.append("")
    lines.append("## Beat Map")
    for beat in generated["beats"]:
        lines.append(
            f"- `{beat['start_seconds']:.2f}s` to `{beat['end_seconds']:.2f}s`: {beat['text']}"
        )
    lines.append("")
    lines.append("## Delivery Notes")
    for note in generated["delivery_notes"]:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def resolve_output_path(source_name: str, output: Path | None) -> Path:
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        return output
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_OUTPUT_DIR / f"{slugify(source_name)}_matrx_script.md"


def resolve_json_output_path(markdown_output: Path, json_output: Path | None) -> Path:
    if json_output is not None:
        json_output.parent.mkdir(parents=True, exist_ok=True)
        return json_output
    return markdown_output.with_suffix(".json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a short-form Reels/TikTok script pack for MATRX by extracting "
            "the transcript and viral structure of a source video."
        )
    )
    parser.add_argument("source", help="Video URL or local transcript file.")
    parser.add_argument(
        "--brief",
        type=Path,
        default=DEFAULT_BRIEF_PATH,
        help="Brand brief or positioning file that the new script should map onto.",
    )
    parser.add_argument(
        "--creator",
        action="append",
        dest="creators",
        help="Creator style profile to blend into the short-form rewrite. Can be passed multiple times.",
    )
    parser.add_argument(
        "--creator-profiles",
        type=Path,
        default=DEFAULT_CREATOR_PROFILES_PATH,
        help="Path to the creator profile JSON library.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Markdown output path. Defaults to analysis/scripts/<source>_matrx_script.md.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional JSON sidecar path. Defaults beside the markdown output.",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Preferred transcript language when downloading captions.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_TRANSCRIBE_MODEL,
        help="Whisper model used only if the source has no usable subtitle track.",
    )
    parser.add_argument(
        "--target-seconds",
        type=float,
        default=DEFAULT_TARGET_SECONDS,
        help="Preferred script duration. Defaults to 30 seconds.",
    )
    parser.add_argument(
        "--min-seconds",
        type=float,
        default=DEFAULT_MIN_SECONDS,
        help="Minimum duration target. Defaults to 20 seconds.",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=DEFAULT_MAX_SECONDS,
        help="Maximum duration target. Defaults to 45 seconds.",
    )
    parser.add_argument(
        "--speech-rate-wpm",
        type=float,
        default=DEFAULT_SPEECH_RATE_WPM,
        help="Estimated speaking rate for duration planning.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    creator_profiles = load_creator_profiles(args.creator_profiles)
    creator_slugs = args.creators or list(DEFAULT_CREATORS)

    active_profiles: list[CreatorProfile] = []
    skipped_profiles: list[CreatorProfile] = []
    for slug in creator_slugs:
        if slug not in creator_profiles:
            raise SystemExit(f"Unknown creator profile: {slug}")
        profile = creator_profiles[slug]
        if profile.research_status != "researched":
            skipped_profiles.append(profile)
            continue
        active_profiles.append(profile)
    if not active_profiles:
        active_profiles = [creator_profiles[slug] for slug in DEFAULT_CREATORS]

    brief_text = read_text(args.brief)
    claims = extract_brief_claims(brief_text)

    if is_url(args.source):
        cache_dir = ROOT / ".script_cache"
        metadata, segments = fetch_transcript_from_url(
            args.source,
            language=args.language,
            model_name=args.model,
            cache_dir=cache_dir,
        )
    else:
        source_path = Path(args.source).expanduser().resolve()
        if not source_path.exists():
            raise SystemExit(f"Source not found: {source_path}")
        segments = load_transcript_from_file(source_path, args.speech_rate_wpm)
        metadata = SourceMetadata(
            source=str(source_path),
            title=source_path.stem,
            uploader="local",
            duration_seconds=segments[-1].end_seconds if segments else None,
            transcript_source=f"file:{source_path.suffix.lower() or '.txt'}",
        )

    if not segments:
        raise SystemExit("No transcript text could be extracted from the source.")

    source_analysis = analyze_source_transcript(segments, args.speech_rate_wpm)
    generated = generate_short_form_script(
        brief_claims=claims,
        source_analysis=source_analysis,
        active_profiles=active_profiles,
        min_seconds=min(args.min_seconds, args.target_seconds),
        max_seconds=max(args.max_seconds, args.target_seconds),
        speech_rate_wpm=args.speech_rate_wpm,
    )

    output_path = resolve_output_path(metadata.title, args.output)
    json_output_path = resolve_json_output_path(output_path, args.json_output)

    report = render_markdown_report(
        metadata=metadata,
        source_analysis=source_analysis,
        creator_profiles=active_profiles,
        skipped_profiles=skipped_profiles,
        brief_claims=claims,
        generated=generated,
    )
    output_path.write_text(report, encoding="utf-8")

    json_payload = {
        "source": asdict(metadata),
        "analysis": {
            "structure_template": source_analysis.structure_template,
            "hook_text": source_analysis.hook_text,
            "source_word_count": source_analysis.source_word_count,
            "estimated_source_seconds": round(source_analysis.estimated_source_seconds, 2),
            "sentence_count": source_analysis.sentence_count,
            "cadence_hint_words_per_line": source_analysis.cadence_hint_words_per_line,
            "detected_elements": [asdict(item) for item in source_analysis.detected_elements],
        },
        "creator_profiles": [asdict(profile) for profile in active_profiles],
        "skipped_profiles": [asdict(profile) for profile in skipped_profiles],
        "brief_claims": asdict(claims),
        "generated": generated,
    }
    json_output_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    print(f"Created short-form script pack: {output_path}")
    print(f"JSON sidecar: {json_output_path}")
    print(f"Estimated duration: {generated['estimated_duration_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
