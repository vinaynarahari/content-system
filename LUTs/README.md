Drop your `.cube` LUT files in this folder.

Examples:

```bash
python3 color_grade.py IMG_4780.MOV --lut my_look.cube
python3 color_grade.py IMG_4780.MOV --lut my_look --working-space rec709
python3 color_grade.py /Users/vinaynarahari/B-Roll --lut my_look --working-space flat709
python3 color_grade.py IMG_4780.MOV --look custom1
python3 color_grade.py IMG_4780.MOV --look custom2
python3 color_grade.py IMG_4780.MOV --look custom3
python3 color_grade.py IMG_4780.MOV --look custom4
```

Notes:

- `rec709` is the safer default when a LUT expects standard SDR footage.
- `flat709` is a softer, lower-contrast base. It is not true camera log.
- The script auto-normalizes iPhone HLG HDR clips into `bt709` before applying the LUT.
- `custom1` is the original adaptive talking-head look.
- `custom2` is a cleaner, brighter, more vibrant built-in grade with less cool/dull bias.
- `custom3` is a warmer, sunnier built-in grade aimed at a California / golden-hour vibe.
- `custom4` builds from a flatter working image and then grades up more gently for talking-head footage.
