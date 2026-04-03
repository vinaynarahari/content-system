from pathlib import Path

from video_script import (
    BriefClaims,
    CreatorProfile,
    TranscriptSegment,
    analyze_source_transcript,
    estimate_spoken_seconds,
    extract_brief_claims,
    generate_short_form_script,
    parse_json3,
    parse_vtt,
)


def test_parse_vtt_extracts_segments() -> None:
    content = """WEBVTT

00:00:00.000 --> 00:00:01.200
This is the hook.

00:00:01.200 --> 00:00:02.700
This is the payoff.
"""
    segments = parse_vtt(content)
    assert len(segments) == 2
    assert segments[0].text == "This is the hook."
    assert segments[1].end_seconds == 2.7


def test_parse_json3_extracts_segments() -> None:
    content = """
{
  "events": [
    {"tStartMs": 0, "dDurationMs": 800, "segs": [{"utf8": "Hook line"}]},
    {"tStartMs": 800, "dDurationMs": 1200, "segs": [{"utf8": "Second line"}]}
  ]
}
"""
    segments = parse_json3(content)
    assert [segment.text for segment in segments] == ["Hook line", "Second line"]
    assert segments[1].start_seconds == 0.8


def test_extract_brief_claims_pulls_key_sentences() -> None:
    brief = """
MTRX is the layer between your team and the assistants you already use.
Teams keep re-explaining the same context because the assistant forgets.
It learns from what worked and stays inside your rules.
MTRX turns AI coding assistants into a managed company capability.
"""
    claims = extract_brief_claims(brief)
    assert "layer between your team" in claims.solution.lower()
    assert "managed company capability" in claims.positioning.lower()


def test_analyze_source_transcript_detects_contrarian_hook() -> None:
    segments = [
        TranscriptSegment(0.0, 1.5, "The problem isn't the model. It's the missing layer."),
        TranscriptSegment(1.5, 3.2, "Every team keeps paying the same hidden tax."),
        TranscriptSegment(3.2, 6.0, "Here's how that gets fixed."),
    ]
    analysis = analyze_source_transcript(segments, 185.0)
    names = [item.name for item in analysis.detected_elements]
    assert "contrarian" in names
    assert analysis.structure_template == "contrarian-problem-solution"


def test_generate_short_form_script_stays_in_duration_window() -> None:
    analysis = analyze_source_transcript(
        [
            TranscriptSegment(0.0, 1.0, "The problem isn't intelligence. It's memory."),
            TranscriptSegment(1.0, 3.0, "Teams keep redoing the same work because the assistant forgets."),
        ],
        185.0,
    )
    claims = BriefClaims(
        problem="AI assistants forget the context, so teams keep repeating the same work.",
        solution="MTRX is the layer between your team and the assistants you already use.",
        mechanism="It remembers what matters, keeps the assistant inside your rules, and learns from what worked.",
        payoff="So AI stops acting like a clever chat and starts behaving like company infrastructure.",
        positioning="MTRX turns AI coding assistants into a managed company capability.",
    )
    profiles = [
        CreatorProfile(
            slug="alex-hormozi",
            display_name="Alex Hormozi",
            research_status="researched",
            platform_fit="Direct-response business short form.",
            voice_traits=[],
            hook_patterns=[],
            script_rules=[],
            avoid=[],
        )
    ]
    generated = generate_short_form_script(
        brief_claims=claims,
        source_analysis=analysis,
        active_profiles=profiles,
        min_seconds=20.0,
        max_seconds=30.0,
        speech_rate_wpm=185.0,
    )
    duration = generated["estimated_duration_seconds"]
    assert 20.0 <= duration <= 30.0
    assert generated["script_lines"]
