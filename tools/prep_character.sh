#!/usr/bin/env bash
# Stage a folder of raw photos into a captioned training dataset and produce
# contact sheets for review/captioning.
#
#   tools/prep_character.sh <name> <raw_photo_dir>
#   tools/prep_character.sh rocket ~/photos/rocket_v2
#
# Produces:
#   ~/datasets/<name>/img001.jpg ...        clean RGB, EXIF-rotated, <=2048px
#   ~/datasets/<name>/metadata.jsonl        skeleton (filename + empty caption)
#   /tmp/<name>_contact/sheet_NN.jpg        labeled montages (6 per sheet)
#
# Then captions get written into metadata.jsonl (Claude reads the sheets and
# fills them in), and tools/train_character.sh <name> ~/datasets/<name> "<phrase>"
# trains on the captioned set.
set -euo pipefail
cd "$(dirname "$0")/.."

NAME="${1:?usage: prep_character.sh <name> <raw_photo_dir>}"
RAW="${2:?raw photo dir required}"
DATASET="$HOME/datasets/$NAME"
CONTACT="/tmp/${NAME}_contact"
PY=.venv/bin/python
[ -d "$RAW" ] || { echo "photo dir not found: $RAW" >&2; exit 1; }

rm -rf "$DATASET" "$CONTACT" && mkdir -p "$DATASET" "$CONTACT"
$PY - "$RAW" "$DATASET" "$CONTACT" <<'PYEOF'
import os, sys, json
from PIL import Image, ImageOps, ImageDraw
try:
    import pillow_heif; pillow_heif.register_heif_opener()
except ImportError:
    pass
raw, dataset, contact = sys.argv[1], sys.argv[2], sys.argv[3]

# 1. stage clean images
staged = []
for fn in sorted(os.listdir(raw)):
    p = os.path.join(raw, fn)
    if not os.path.isfile(p) or fn.startswith("."):
        continue
    try:
        img = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
    except Exception:
        print(f"  skip (not an image): {fn}"); continue
    if max(img.size) > 2048:
        s = 2048 / max(img.size)
        img = img.resize((round(img.width*s), round(img.height*s)), Image.LANCZOS)
    out = f"img{len(staged)+1:03d}.jpg"
    img.save(os.path.join(dataset, out), quality=95)
    staged.append(out)
print(f"staged {len(staged)} images -> {dataset}")

# 2. skeleton metadata.jsonl (empty captions to fill in)
with open(os.path.join(dataset, "metadata.jsonl"), "w") as f:
    for out in staged:
        f.write(json.dumps({"file_name": out, "caption": ""}) + "\n")
print(f"wrote skeleton metadata.jsonl ({len(staged)} rows)")

# 3. labeled contact sheets, 6 tiles (3x2) per sheet, 512px tiles
TILE, COLS, ROWS = 512, 3, 2
per = COLS * ROWS
for s in range((len(staged) + per - 1) // per):
    sheet = Image.new("RGB", (COLS*TILE, ROWS*TILE), (20, 20, 24))
    d = ImageDraw.Draw(sheet)
    for i in range(per):
        idx = s*per + i
        if idx >= len(staged): break
        im = Image.open(os.path.join(dataset, staged[idx])).convert("RGB")
        im.thumbnail((TILE-8, TILE-8), Image.LANCZOS)
        cx, cy = (i % COLS)*TILE, (i // COLS)*TILE
        sheet.paste(im, (cx + (TILE-im.width)//2, cy + (TILE-im.height)//2))
        d.rectangle([cx+2, cy+2, cx+150, cy+24], fill=(0, 0, 0))
        d.text((cx+6, cy+6), staged[idx], fill=(255, 255, 0))
    sheet.save(os.path.join(contact, f"sheet_{s+1:02d}.jpg"), quality=88)
print(f"wrote {(len(staged)+per-1)//per} contact sheets -> {contact}")
PYEOF
echo "[prep] DONE. Review sheets in $CONTACT, then captions go into $DATASET/metadata.jsonl"
