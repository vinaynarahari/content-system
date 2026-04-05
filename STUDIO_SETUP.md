# Studio Setup

## One-time setup

1. Copy [studio_config.json.example](/Users/vinaynarahari/B-Roll/studio_config.json.example) to `studio_config.json`.
2. Add your Instagram and TikTok handles under `first_party_handles`.
3. If you want better access to private or login-gated content:
   - set `yt_dlp.cookies_from_browser` to your browser name, for example `chrome` or `safari`
   - or point `yt_dlp.cookie_file` to an exported cookies file
4. If you want authenticated page capture:
   - create a Playwright storage state file for your logged-in session
   - set `browser_session.storage_state_path` to that file

## Historical ingest

Use link plus screenshots only:

```bash
./studio ingest "https://www.instagram.com/reel/..." --screenshots /path/to/insights1.png
./studio ingest "https://www.tiktok.com/@user/video/..." --screenshots /path/to/insights2.png
```

Then train:

```bash
./studio train
```

## New content workflow

Generate a new script package:

```bash
./studio ideate "MTRX is the missing layer for AI coding assistants"
```

Produce a fuller project package:

```bash
./studio produce briefs/matrx.md --fcpxml /path/to/project.fcpxml
```

Review a historical item:

```bash
./studio review <content-id-or-link>
```
