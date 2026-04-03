from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont


WIDTH = 1080
HEIGHT = 1920
FPS = 30
DURATION_SECONDS = 2.95
FRAME_COUNT = round(FPS * DURATION_SECONDS)

ROOT = Path(__file__).resolve().parent
LOGO_PATH = ROOT / "mtrx-logo.png"
TEXT_PATH = ROOT / "mtrx-text.png"
OUTPUT_MOV = ROOT / "Completed" / "matrx_brand_reveal_vertical.mov"
OUTPUT_MP4 = ROOT / "Completed" / "matrx_brand_reveal_vertical_preview.mp4"

FONT_PATH = "/System/Library/Fonts/SFNS.ttf"
TAGLINE = "Infrastructure for intelligence"

BG_COLOR = (0, 0, 0)
MARK_COLOR = (248, 249, 251)
TAGLINE_COLOR = (210, 214, 220)


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def ease_out_cubic(value: float) -> float:
    value = clamp(value)
    return 1.0 - ((1.0 - value) ** 3)


def ease_in_out_cubic(value: float) -> float:
    value = clamp(value)
    if value < 0.5:
        return 4.0 * value * value * value
    return 1.0 - ((-2.0 * value + 2.0) ** 3) / 2.0


def linear_progress(start: float, end: float, value: float) -> float:
    if end <= start:
        return 1.0
    return clamp((value - start) / (end - start))


def smoothstep(edge0: float, edge1: float, value: float) -> float:
    if edge1 <= edge0:
        return 1.0
    return ease_in_out_cubic((value - edge0) / (edge1 - edge0))


def load_alpha(path: Path, threshold: int = 20) -> Image.Image:
    image = Image.open(path).convert("RGBA")
    alpha = image.getchannel("A")
    bbox = alpha.point(lambda px: 255 if px > threshold else 0).getbbox()
    if bbox is None:
        return alpha
    return alpha.crop(bbox)


def resize_alpha(alpha: Image.Image, width: int) -> Image.Image:
    aspect_ratio = alpha.height / alpha.width
    height = max(1, round(width * aspect_ratio))
    return alpha.resize((width, height), Image.Resampling.LANCZOS)


def visual_center_offset(alpha: Image.Image) -> tuple[float, float]:
    total = 0
    sum_x = 0.0
    sum_y = 0.0
    for y in range(alpha.height):
        for x in range(alpha.width):
            opacity = alpha.getpixel((x, y))
            if opacity <= 0:
                continue
            total += opacity
            sum_x += x * opacity
            sum_y += y * opacity

    if total == 0:
        return 0.0, 0.0

    center_x = sum_x / total
    center_y = sum_y / total
    return center_x - (alpha.width / 2), center_y - (alpha.height / 2)


def tint_alpha(alpha: Image.Image, color: tuple[int, int, int], opacity: float) -> Image.Image:
    layer = Image.new("RGBA", alpha.size, color + (0,))
    layer.putalpha(alpha.point(lambda px: round(px * clamp(opacity))))
    return layer


def paste_center(
    base: Image.Image,
    layer: Image.Image,
    center_x: float,
    top_y: float,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> None:
    x = round(center_x - (layer.width / 2) - offset_x)
    y = round(top_y - offset_y)
    base.alpha_composite(layer, (x, y))


def draw_tracking_text(
    image: Image.Image,
    text: str,
    font: ImageFont.FreeTypeFont,
    center_x: float,
    top_y: float,
    tracking: float,
    fill: tuple[int, int, int, int],
) -> None:
    draw = ImageDraw.Draw(image)
    widths = [draw.textbbox((0, 0), char, font=font)[2] for char in text]
    total_width = sum(widths) + max(0, len(text) - 1) * tracking
    cursor_x = center_x - (total_width / 2)
    for index, char in enumerate(text):
        draw.text((cursor_x, top_y), char, font=font, fill=fill)
        cursor_x += widths[index] + tracking


def make_background(reveal: float, center_x: float, logo_center_y: float) -> Image.Image:
    background = Image.new("RGBA", (WIDTH, HEIGHT), BG_COLOR + (255,))
    if reveal <= 0.0:
        return background

    # Tight aperture-style backlight behind the mark only.
    spotlight = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(spotlight)
    radius_x = round(140 + (70 * reveal))
    radius_y = round(90 + (45 * reveal))
    alpha = round(14 + (20 * reveal))
    draw.ellipse(
        (
            round(center_x - radius_x),
            round(logo_center_y - radius_y),
            round(center_x + radius_x),
            round(logo_center_y + radius_y),
        ),
        fill=(255, 255, 255, alpha),
    )
    spotlight = spotlight.filter(ImageFilter.GaussianBlur(70))
    background.alpha_composite(spotlight)
    return background


def render_frame(
    frame_index: int,
    logo_alpha: Image.Image,
    wordmark_alpha: Image.Image,
    logo_center_offset: tuple[float, float],
    wordmark_center_offset: tuple[float, float],
    tagline_font: ImageFont.FreeTypeFont,
) -> Image.Image:
    t = frame_index / FPS

    logo_reveal = ease_out_cubic(smoothstep(0.14, 1.44, t))
    wordmark_reveal = ease_out_cubic(smoothstep(0.30, 1.72, t))
    tagline_reveal = ease_out_cubic(smoothstep(1.26, 2.36, t))

    center_x = WIDTH / 2
    logo_center_y = HEIGHT * 0.390
    wordmark_top_y = HEIGHT * 0.512
    tagline_top_y = HEIGHT * 0.662

    logo_scale = 0.975 + (0.025 * logo_reveal)
    wordmark_scale = 0.988 + (0.012 * wordmark_reveal)
    logo_width = round(418 * logo_scale)
    wordmark_width = round(430 * wordmark_scale)

    background = make_background(logo_reveal * 0.92, center_x, logo_center_y)

    logo_alpha_resized = resize_alpha(logo_alpha, logo_width)
    logo_layer = tint_alpha(logo_alpha_resized, MARK_COLOR, logo_reveal)
    logo_blur = max(0.0, 8.0 * ((1.0 - logo_reveal) ** 1.1))
    if logo_blur > 0.0:
        logo_layer = logo_layer.filter(ImageFilter.GaussianBlur(logo_blur))
    logo_offset_x = logo_center_offset[0] * (logo_alpha_resized.width / logo_alpha.width)
    logo_offset_y = logo_center_offset[1] * (logo_alpha_resized.height / logo_alpha.height)
    paste_center(
        background,
        logo_layer,
        center_x,
        logo_center_y - (logo_layer.height / 2) + round((1.0 - logo_reveal) * 16),
        offset_x=logo_offset_x,
        offset_y=logo_offset_y,
    )

    wordmark_alpha_resized = resize_alpha(wordmark_alpha, wordmark_width)
    wordmark_layer = tint_alpha(wordmark_alpha_resized, MARK_COLOR, wordmark_reveal)
    wordmark_blur = max(0.0, 6.0 * ((1.0 - wordmark_reveal) ** 1.15))
    if wordmark_blur > 0.0:
        wordmark_layer = wordmark_layer.filter(ImageFilter.GaussianBlur(wordmark_blur))
    wordmark_offset_x = wordmark_center_offset[0] * (
        wordmark_alpha_resized.width / wordmark_alpha.width
    )
    wordmark_offset_y = wordmark_center_offset[1] * (
        wordmark_alpha_resized.height / wordmark_alpha.height
    )
    paste_center(
        background,
        wordmark_layer,
        center_x,
        wordmark_top_y + round((1.0 - wordmark_reveal) * 14),
        offset_x=wordmark_offset_x,
        offset_y=wordmark_offset_y,
    )

    if tagline_reveal > 0.0:
        tagline_layer = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
        draw_tracking_text(
            tagline_layer,
            TAGLINE,
            tagline_font,
            center_x=center_x,
            top_y=tagline_top_y + round((1.0 - tagline_reveal) * 10),
            tracking=1.4,
            fill=TAGLINE_COLOR + (round(255 * tagline_reveal),),
        )
        background.alpha_composite(tagline_layer)

    return background


def encode_video(frame_dir: Path, output_path: Path, codec_args: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(FPS),
        "-i",
        str(frame_dir / "frame_%04d.png"),
        *codec_args,
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    if not LOGO_PATH.exists() or not TEXT_PATH.exists():
        raise FileNotFoundError("Expected MATRX assets were not found in the workspace.")

    logo_alpha = load_alpha(LOGO_PATH)
    wordmark_alpha = load_alpha(TEXT_PATH)
    logo_center_offset = visual_center_offset(logo_alpha)
    wordmark_center_offset = visual_center_offset(wordmark_alpha)
    tagline_font = ImageFont.truetype(FONT_PATH, 44)

    with tempfile.TemporaryDirectory(prefix="matrx_reveal_frames_") as tmp_dir_name:
        frame_dir = Path(tmp_dir_name)
        for frame_index in range(FRAME_COUNT):
            frame = render_frame(
                frame_index=frame_index,
                logo_alpha=logo_alpha,
                wordmark_alpha=wordmark_alpha,
                logo_center_offset=logo_center_offset,
                wordmark_center_offset=wordmark_center_offset,
                tagline_font=tagline_font,
            )
            frame.save(frame_dir / f"frame_{frame_index:04d}.png")

        encode_video(
            frame_dir,
            OUTPUT_MOV,
            [
                "-c:v",
                "prores_ks",
                "-profile:v",
                "2",
                "-pix_fmt",
                "yuv422p10le",
            ],
        )
        encode_video(
            frame_dir,
            OUTPUT_MP4,
            [
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "16",
                "-preset",
                "slow",
                "-movflags",
                "+faststart",
            ],
        )

    print(f"Wrote {OUTPUT_MOV}")
    print(f"Wrote {OUTPUT_MP4}")


if __name__ == "__main__":
    main()
