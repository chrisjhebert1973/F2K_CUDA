#!/usr/bin/env bash
# Bake several character-LoRA checkpoints, quantize each to MXFP8 F2K, generate a
# fixed prompt set per candidate, and assemble one contact sheet per prompt so the
# best (rank, lr, step, scale) can be picked by eye WITHOUT retraining.
#
#   tools/compare_character.sh <label> <base_model_root> "<trigger phrase>" \
#       <lora1.safetensors>[:scale] <lora2.safetensors>[:scale] ...
#
# Each LoRA arg is a pytorch_lora_weights.safetensors (a sweep checkpoint-N/ file
# works directly), optionally suffixed ":0.8" to bake at reduced strength. The
# candidate label is derived from the path (…/r16_lr5e-5/checkpoint-500 -> the two
# trailing path components) plus the scale.
#
# Output: /tmp/cmp_<label>/  — per-candidate PNGs + grid_p<N>.png contact sheets.
# Prompts test identity (close portrait) and flexibility (novel scene); edit below.
set -euo pipefail
cd "$(dirname "$0")/.."

LABEL="${1:?usage: compare_character.sh <label> <base_model_root> \"<phrase>\" <lora[:scale]>...}"
BASE="${2:?base model root required (e.g. ~/models/flux2-klein-4B)}"
PHRASE="${3:?trigger phrase required}"
shift 3
[ "$#" -ge 1 ] || { echo "need at least one LoRA candidate" >&2; exit 1; }

PY=.venv/bin/python
OUT="/tmp/cmp_${LABEL}"; rm -rf "$OUT"; mkdir -p "$OUT"
SEED="${SEED:-0xcafebabe}"
RES="${RES:-1024}"
STEPS="${STEPS:-8}"

# Fixed prompts (identity-tight, then flexibility). Override by exporting PROMPTS
# as a newline-separated list before calling.
if [ -z "${PROMPTS:-}" ]; then
  PROMPTS="a close-up portrait photo of $PHRASE, soft natural light, sharp focus
a photo of $PHRASE, wearing a wizard robe in a candlelit library, cinematic"
fi

mapfile -t PROMPT_ARR <<< "$PROMPTS"
echo "[cmp] $LABEL: $# candidate(s) x ${#PROMPT_ARR[@]} prompt(s), base=$BASE res=$RES steps=$STEPS"

LABELS=()
for spec in "$@"; do
  LORA="${spec%%:*}"
  SCALE="1.0"; [[ "$spec" == *:* ]] && SCALE="${spec##*:}"
  [ -f "$LORA" ] || { echo "[cmp] missing LoRA: $LORA" >&2; exit 1; }
  # label = last two DIRECTORY components (config + checkpoint) + scale, sanitized.
  # The LoRA filename is the generic pytorch_lora_weights.safetensors, so it carries
  # no info — derive the tag from the dir so different configs don't collide.
  ldir="$(dirname "$LORA")"
  tag="$(echo "$ldir" | awk -F/ '{print $(NF-1)"_"$NF}' | sed 's/[^A-Za-z0-9._-]/_/g')"
  [ "$SCALE" != "1.0" ] && tag="${tag}_s${SCALE}"
  LABELS+=("$tag")
  CAND="$OUT/$tag"; mkdir -p "$CAND/transformer_mxfp8"
  echo "[cmp] === $tag (scale=$SCALE) ==="
  MERGED="/tmp/cmp_${LABEL}_${tag}.safetensors"
  $PY tools/merge_lora.py "$BASE/transformer" "$LORA" "$MERGED" --scale "$SCALE" | tail -1
  ./build/f2k_convert "$MERGED" "$CAND/transformer_mxfp8/shard-00001.f2k1" --quant mxfp8 | tail -1
  rm -f "$MERGED"
  pi=0
  for p in "${PROMPT_ARR[@]}"; do
    [ -z "$p" ] && continue
    ./build/generate --model "$BASE" --transformer "$CAND/transformer_mxfp8" \
      --precision fp8 --res "$RES" --steps "$STEPS" --seed "$SEED" \
      --prompt "$p" --out "$CAND/p${pi}.png" > "$CAND/gen_p${pi}.log" 2>&1
    echo "[cmp]   p$pi -> $CAND/p${pi}.png"
    pi=$((pi+1))
  done
done

# Contact sheets: one row per candidate, one column per prompt, labeled.
$PY - "$OUT" "${LABELS[@]}" <<'PYEOF'
import os, sys, glob
from PIL import Image, ImageDraw, ImageFont
out = sys.argv[1]; labels = sys.argv[2:]
nprompt = max(len(glob.glob(os.path.join(out, l, "p*.png"))) for l in labels)
for pi in range(nprompt):
    imgs = []
    for l in labels:
        fp = os.path.join(out, l, f"p{pi}.png")
        imgs.append((l, Image.open(fp).convert("RGB") if os.path.exists(fp) else None))
    w = max((im.width for _, im in imgs if im), default=512)
    h = max((im.height for _, im in imgs if im), default=512)
    pad, lh = 8, 22
    sheet = Image.new("RGB", (w + 2*pad, (h+lh+pad)*len(imgs) + pad), (20,20,20))
    d = ImageDraw.Draw(sheet)
    y = pad
    for l, im in imgs:
        d.text((pad, y), l, fill=(230,230,230))
        if im: sheet.paste(im.resize((w,h)), (pad, y+lh))
        y += h+lh+pad
    sp = os.path.join(out, f"grid_p{pi}.png"); sheet.save(sp)
    print("wrote", sp)
PYEOF
echo "[cmp] DONE -> $OUT/grid_p*.png"
