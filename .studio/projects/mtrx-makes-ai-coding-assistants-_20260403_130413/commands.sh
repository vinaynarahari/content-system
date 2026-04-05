#!/usr/bin/env bash
set -euo pipefail

python3 make_matrx_reveal.py
./broll 11.1 --style builder
open /Users/vinaynarahari/B-Roll/.studio/projects/mtrx-makes-ai-coding-assistants-_20260403_130413