#!/usr/bin/env python3
import argparse
import html
import json
import os
import re
import shutil
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
)


def fetch_text(url):
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")


def fetch_binary(url, destination):
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=60) as response:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            shutil.copyfileobj(response, handle)


def extract_coverr_mp4(page_html):
    match = re.search(r'"contentUrl":"([^"]+\.mp4)"', page_html)
    if match:
        return html.unescape(match.group(1).replace("\\/", "/"))

    match = re.search(r'src="(https://cdn\.coverr\.co/videos/[^"]+/360p\.mp4)"', page_html)
    if match:
        preview_url = html.unescape(match.group(1))
        return preview_url.replace("/360p.mp4", "/1080p.mp4")

    raise ValueError("Could not resolve Coverr MP4 URL from page HTML.")


def extract_mixkit_mp4(page_html):
    match = re.search(r'https://assets\.mixkit\.co/videos/[^"\']+\.mp4', page_html)
    if match:
        return html.unescape(match.group(0))

    raise ValueError("Could not resolve Mixkit MP4 URL from page HTML.")


def load_manifest(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_args():
    parser = argparse.ArgumentParser(description="Download stock B-roll clips from a local manifest.")
    parser.add_argument(
        "--manifest",
        default="stock_broll/manifest.json",
        help="Path to the manifest JSON file.",
    )
    parser.add_argument(
        "--base-dir",
        default="stock_broll",
        help="Base directory for downloaded clips.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download clips even if a local file already exists.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    base_dir = Path(args.base_dir).expanduser().resolve()
    payload = load_manifest(manifest_path)
    clips = payload.get("clips", [])

    if not clips:
        print("No clips found in manifest.", file=sys.stderr)
        return 1

    downloaded = 0
    skipped = 0
    failures = 0

    for clip in clips:
        source_url = clip["source_url"]
        provider = clip.get("provider", "").lower()
        group = clip["group"]
        filename = clip["filename"]
        destination = base_dir / group / filename

        if destination.exists() and destination.stat().st_size > 0 and not args.overwrite:
            skipped += 1
            print(f"SKIP  {destination}")
            continue

        try:
            page_html = fetch_text(source_url)
            if provider == "coverr":
                media_url = extract_coverr_mp4(page_html)
            elif provider == "mixkit":
                media_url = extract_mixkit_mp4(page_html)
            else:
                raise ValueError(f"Unsupported provider: {provider}")

            tmp_destination = destination.with_suffix(destination.suffix + ".part")
            if tmp_destination.exists():
                tmp_destination.unlink()

            print(f"GET   {source_url}")
            print(f"      -> {media_url}")
            fetch_binary(media_url, tmp_destination)
            tmp_destination.replace(destination)
            downloaded += 1
        except (HTTPError, URLError, TimeoutError, ValueError, OSError) as exc:
            failures += 1
            print(f"FAIL  {source_url} :: {exc}", file=sys.stderr)
            part_path = destination.with_suffix(destination.suffix + ".part")
            if part_path.exists():
                part_path.unlink()

    print(
        f"\nDone. downloaded={downloaded} skipped={skipped} failures={failures} "
        f"base_dir={base_dir}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
