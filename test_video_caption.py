from video_caption import (
    FRAME_TICKS,
    FONT_COLOR,
    POSITION,
    TitleChunk,
    chunk_end_ticks,
    normalize_caption_chunks,
    seconds_to_ticks,
)


def make_chunk(start_seconds: float, duration_seconds: float, text: str) -> TitleChunk:
    return TitleChunk(
        text=text,
        start_ticks=seconds_to_ticks(start_seconds),
        duration_ticks=seconds_to_ticks(duration_seconds),
        style_name="classic",
        font_color=FONT_COLOR,
        font_scale=1.0,
        position=POSITION,
        alignment="center",
    )


def test_normalize_caption_chunks_trims_overlap_with_clearance() -> None:
    chunks = [
        make_chunk(0.0, 0.60, "FIRST"),
        make_chunk(0.55, 0.25, "SECOND"),
    ]

    normalized = normalize_caption_chunks(chunks, seconds_to_ticks(0.1))

    assert chunk_end_ticks(normalized[0]) == seconds_to_ticks(0.45)
    assert normalized[1] == chunks[1]


def test_normalize_caption_chunks_preserves_frame_when_clearance_cannot_fit() -> None:
    chunks = [
        make_chunk(0.0, 0.20, "FIRST"),
        make_chunk(FRAME_TICKS / 30000, 0.20, "SECOND"),
    ]

    normalized = normalize_caption_chunks(chunks, seconds_to_ticks(0.1))

    assert normalized[0].duration_ticks == FRAME_TICKS
    assert chunk_end_ticks(normalized[0]) == FRAME_TICKS


def test_normalize_caption_chunks_extends_to_next_start_with_clearance() -> None:
    chunks = [
        make_chunk(0.0, 0.20, "FIRST"),
        make_chunk(0.50, 0.20, "SECOND"),
    ]

    normalized = normalize_caption_chunks(chunks, seconds_to_ticks(0.1))

    assert chunk_end_ticks(normalized[0]) == seconds_to_ticks(0.4)
    assert normalized[1] == chunks[1]


def test_normalize_caption_chunks_holds_until_next_start_without_clearance() -> None:
    chunks = [
        make_chunk(0.0, 0.20, "FIRST"),
        make_chunk(0.50, 0.20, "SECOND"),
    ]

    normalized = normalize_caption_chunks(chunks, 0)

    assert chunk_end_ticks(normalized[0]) == seconds_to_ticks(0.5)
    assert normalized[1] == chunks[1]


def test_normalize_caption_chunks_flattens_same_start_runs() -> None:
    chunks = [
        make_chunk(1.0, 0.04, "ONE"),
        make_chunk(1.0, 0.04, "TWO"),
        make_chunk(1.0, 0.20, "THREE"),
        make_chunk(1.40, 0.20, "FOUR"),
    ]

    normalized = normalize_caption_chunks(chunks, seconds_to_ticks(0.1))

    assert normalized[0].start_ticks == seconds_to_ticks(1.0)
    assert normalized[1].start_ticks > normalized[0].start_ticks
    assert normalized[2].start_ticks > normalized[1].start_ticks
    assert chunk_end_ticks(normalized[0]) <= normalized[1].start_ticks
    assert chunk_end_ticks(normalized[1]) <= normalized[2].start_ticks
    assert chunk_end_ticks(normalized[2]) <= normalized[3].start_ticks


def test_normalize_caption_chunks_shifts_subframe_starts_forward() -> None:
    chunks = [
        make_chunk(2.0, FRAME_TICKS / 30000, "ONE"),
        make_chunk(2.0 + 0.02, 0.08, "TWO"),
        make_chunk(2.20, 0.10, "THREE"),
    ]

    normalized = normalize_caption_chunks(chunks, seconds_to_ticks(0.1))

    assert chunk_end_ticks(normalized[0]) <= normalized[1].start_ticks
    assert chunk_end_ticks(normalized[1]) <= normalized[2].start_ticks
