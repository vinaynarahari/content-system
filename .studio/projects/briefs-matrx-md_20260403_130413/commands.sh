#!/usr/bin/env bash
set -euo pipefail

python3 make_matrx_reveal.py
./broll 11.1 --style builder
open /Users/vinaynarahari/B-Roll/.studio/projects/briefs-matrx-md_20260403_130413