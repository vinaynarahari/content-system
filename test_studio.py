import sqlite3
from pathlib import Path

from studio import (
    build_broll_plan,
    build_hashtag_recommendations,
    build_music_recommendations,
    build_post_caption,
    build_training_state,
    compute_content_id,
    compute_ingestion_completeness,
    compute_scores,
    detect_platform,
    extract_metrics_from_text,
    extract_platform_post_id,
    normalize_url,
    parse_compact_number,
    parse_seconds_value,
)


def make_item(
    content_id: str,
    owner_type: str,
    transcript_text: str,
    duration_seconds: float,
    views: int,
    average_watch_time: float | None = None,
    completion_rate: float | None = None,
    shares: int | None = None,
    saves: int | None = None,
    follows: int | None = None,
) -> dict:
    return {
        "content_id": content_id,
        "owner_type": owner_type,
        "platform": "instagram",
        "platform_post_id": content_id,
        "url": f"https://example.com/{content_id}",
        "title": content_id,
        "raw_caption": transcript_text,
        "transcript_text": transcript_text,
        "hook_text": "",
        "structure_type": "",
        "duration_seconds": duration_seconds,
        "views": views,
        "likes": 100,
        "comments": 10,
        "shares": shares,
        "saves": saves,
        "average_watch_time": average_watch_time,
        "completion_rate": completion_rate,
        "accounts_reached": None,
        "follows": follows,
        "profile_visits": None,
    }


def test_url_helpers_work_for_instagram_and_tiktok() -> None:
    instagram = "https://www.instagram.com/reel/ABC123/?utm_source=ig_web_copy_link"
    tiktok = "https://www.tiktok.com/@name/video/1234567890?lang=en"
    assert normalize_url(instagram) == "https://www.instagram.com/reel/ABC123"
    assert detect_platform(instagram) == "instagram"
    assert detect_platform(tiktok) == "tiktok"
    assert extract_platform_post_id(instagram) == "ABC123"
    assert extract_platform_post_id(tiktok) == "1234567890"


def test_parse_helpers_extract_counts_and_seconds() -> None:
    assert parse_compact_number("1.2K") == 1200
    assert parse_compact_number("3M") == 3000000
    assert parse_seconds_value("1m 12s") == 72
    assert parse_seconds_value("00:31") == 31


def test_extract_metrics_from_text_parses_insight_lines() -> None:
    text = """
    Views 12.4K
    Likes 810
    Shares 47
    Saves 38
    Average watch time 12s
    Completion rate 44%
    Follows 9
    """
    metrics = extract_metrics_from_text(text)
    assert metrics["views"] == 12400
    assert metrics["likes"] == 810
    assert metrics["shares"] == 47
    assert metrics["saves"] == 38
    assert metrics["average_watch_time"] == 12
    assert metrics["completion_rate"] == 0.44
    assert metrics["follows"] == 9


def test_compute_scores_prefers_stronger_retention_and_shares() -> None:
    strong = make_item(
        "strong",
        "first_party_matrix",
        "The problem isn't the model. It's memory.",
        duration_seconds=30,
        views=10_000,
        average_watch_time=19,
        completion_rate=0.52,
        shares=120,
        saves=90,
        follows=40,
    )
    weak = make_item(
        "weak",
        "first_party_matrix",
        "AI is changing everything and here's what we built.",
        duration_seconds=30,
        views=10_000,
        average_watch_time=8,
        completion_rate=0.18,
        shares=9,
        saves=4,
        follows=2,
    )
    scores = compute_scores([strong, weak])
    assert scores["strong"]["performance_score"] > scores["weak"]["performance_score"]


def test_build_training_state_prefers_best_structure() -> None:
    items = [
        make_item(
            "one",
            "first_party_matrix",
            "The problem isn't the model. It's the missing layer. Teams keep paying the tax.",
            duration_seconds=28,
            views=10_000,
            average_watch_time=18,
            completion_rate=0.49,
            shares=90,
            saves=70,
            follows=28,
        ),
        make_item(
            "two",
            "first_party_matrix",
            "The problem isn't the model. It's memory. Here's why teams keep redoing work.",
            duration_seconds=31,
            views=11_000,
            average_watch_time=20,
            completion_rate=0.55,
            shares=110,
            saves=84,
            follows=34,
        ),
        make_item(
            "voice",
            "first_party_voice",
            "Most people think they need more motivation. They need a system.",
            duration_seconds=24,
            views=8_000,
            average_watch_time=13,
            completion_rate=0.33,
            shares=30,
            saves=20,
            follows=12,
        ),
    ]
    state = build_training_state(items, speech_rate_wpm=185.0)
    assert state["preferred_structures"]
    assert state["preferred_structures"][0]["name"] == "contrarian-problem-solution"
    assert 20 <= state["target_duration_seconds"] <= 35


def test_broll_and_posting_helpers_return_structured_output() -> None:
    beats = [
        {"index": 1, "start_seconds": 0.0, "end_seconds": 2.2, "text": "The problem isn't the model."},
        {"index": 2, "start_seconds": 2.2, "end_seconds": 4.8, "text": "MTRX is the missing layer."},
        {"index": 3, "start_seconds": 4.8, "end_seconds": 7.0, "text": "Track what works and stop repeating failures."},
    ]
    plan = build_broll_plan(beats)
    assert plan["brand_reveal_insert_after_beat"] == 2
    assert len(plan["beats"]) == 3
    hashtags = build_hashtag_recommendations("MTRX helps AI coding assistants remember context.")
    assert hashtags
    music = build_music_recommendations("contrarian-problem-solution")
    assert music
    caption = build_post_caption(["The problem isn't the model.", "MTRX is the missing layer."])
    assert "Follow for systems" in caption


def test_ingestion_completeness_rewards_transcript_and_metrics() -> None:
    payload = {
        "title": "Test",
        "duration_seconds": 30,
        "raw_caption": "caption",
        "transcript_text": "full transcript",
        "views": 1000,
        "likes": 100,
        "shares": 12,
        "average_watch_time": 11,
        "completion_rate": 0.31,
        "follows": 4,
    }
    completeness = compute_ingestion_completeness(payload)
    assert completeness > 0.6
