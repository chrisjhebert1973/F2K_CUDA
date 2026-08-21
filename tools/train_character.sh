#!/usr/bin/env bash
# Train a character LoRA on FLUX.2-klein-4B and install it as a selectable
# web-UI checkpoint — photos in, dropdown entry out.
#
#   tools/train_character.sh <name> <dir> "<trigger phrase>" [steps] [rank] [lr]
#
# Two input modes, auto-detected:
#   * CAPTIONED (preferred): <dir> contains metadata.jsonl (from prep_character.sh,
#     captions filled in). Trains with per-image captions — best identity, least
#     background bleed. <dir> is used as-is (already staged).
#   * SIMPLE: <dir> is a raw photo folder with no metadata.jsonl. Staged to /tmp
#     and trained with one shared "<trigger phrase>" for every image.
#
# "<trigger phrase>" = the subject described with its rare trigger token and class,
# e.g. "rkt9d, a black and white Akita husky dog with one blue eye and one amber eye".
# Pick a token NOT near a real word (r0cket collided with 'rocket' = spaceship).
# It seeds the @mention (trigger.json) and, in SIMPLE mode, the instance prompt.
#
# Checkpoints every 250 steps — quality peaks mid-run then overfits, so the best
# step is picked afterwards (bake each, scale-sweep, choose by eye). Output:
# ~/models/flux2-klein-4B-<name>/ ; the web UI discovers it automatically.
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

NAME="${1:?usage: train_character.sh <name> <dir> \"<trigger phrase>\" [steps] [rank] [lr]}"
INPUT="${2:?dataset/photo dir required}"
PHRASE="${3:?trigger phrase (token + class) required}"
STEPS="${4:-1000}"
RANK="${5:-16}"
LR="${6:-1e-4}"

BASE="$HOME/models/flux2-klein-4B"
LORA_DIR="$HOME/loras/$NAME"
DEST="$HOME/models/flux2-klein-4B-$NAME"
PY=.venv/bin/python
[ -d "$INPUT" ] || { echo "input dir not found: $INPUT" >&2; exit 1; }

# Mode detection: captioned dataset (metadata.jsonl present) vs raw photo folder.
DATA_ARGS=()
if [ -f "$INPUT/metadata.jsonl" ]; then
  N_IMG=$(grep -c . "$INPUT/metadata.jsonl")
  N_EMPTY=$($PY - "$INPUT/metadata.jsonl" <<'PYEOF'
import json,sys
print(sum(1 for l in open(sys.argv[1]) if l.strip() and not json.loads(l).get("caption","").strip()))
PYEOF
)
  [ "$N_EMPTY" -eq 0 ] || { echo "$N_EMPTY captions still empty in $INPUT/metadata.jsonl — fill them first" >&2; exit 1; }
  echo "[train] CAPTIONED mode: $N_IMG images, $STEPS steps, rank $RANK, lr $LR"
  DATA_ARGS=(--dataset_name "$INPUT" --caption_column caption)
else
  # SIMPLE mode: stage clean RGB images (HEIC via pillow-heif), single prompt.
  STAGE="/tmp/dataset_$NAME"; rm -rf "$STAGE" && mkdir -p "$STAGE"
  $PY - "$INPUT" "$STAGE" <<'PYEOF'
import os, sys
from PIL import Image, ImageOps
try:
    import pillow_heif; pillow_heif.register_heif_opener()
except ImportError:
    pass
src, dst = sys.argv[1], sys.argv[2]
n = 0
for fn in sorted(os.listdir(src)):
    p = os.path.join(src, fn)
    if not os.path.isfile(p) or fn.startswith("."): continue
    try: img = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
    except Exception: print(f"  skip (not an image): {fn}"); continue
    if max(img.size) > 2048:
        s = 2048/max(img.size); img = img.resize((round(img.width*s), round(img.height*s)), Image.LANCZOS)
    n += 1; img.save(os.path.join(dst, f"img{n:03d}.jpg"), quality=95)
print(f"staged {n} images")
PYEOF
  N_IMG=$(ls "$STAGE" | wc -l)
  [ "$N_IMG" -ge 5 ] || { echo "need at least 5 usable images in $INPUT" >&2; exit 1; }
  echo "[train] SIMPLE mode: $N_IMG images, $STEPS steps, rank $RANK, lr $LR"
  DATA_ARGS=(--instance_data_dir "$STAGE")
fi

echo "[train] 1/4 training LoRA (log: /tmp/train_$NAME.log)..."
HF_HOME=/tmp/hfhome .venv/bin/accelerate launch --num_processes 1 --mixed_precision bf16 \
  tools/training/train_dreambooth_lora_flux2_klein.py \
  --pretrained_model_name_or_path "$BASE" \
  "${DATA_ARGS[@]}" \
  --output_dir "$LORA_DIR" \
  --instance_prompt "a photo of $PHRASE" \
  --resolution 1024 --train_batch_size 1 --guidance_scale 1 \
  --gradient_accumulation_steps 4 --optimizer adamW --learning_rate "$LR" \
  --lr_scheduler constant --lr_warmup_steps 0 \
  --max_train_steps "$STEPS" --checkpointing_steps 250 \
  --rank "$RANK" --lora_alpha "$RANK" \
  --gradient_checkpointing --cache_latents --mixed_precision bf16 \
  --seed 0 > "/tmp/train_$NAME.log" 2>&1
LORA="$LORA_DIR/pytorch_lora_weights.safetensors"
[ -f "$LORA" ] || { echo "training produced no LoRA — see /tmp/train_$NAME.log" >&2; exit 1; }
echo "[train] checkpoints saved every 250 steps in $LORA_DIR (bake + scale-sweep to pick the best)"

echo "[train] 2/4 merging LoRA into BF16 base..."
MERGED="/tmp/${NAME}_merged.safetensors"
$PY tools/merge_lora.py "$BASE/transformer" "$LORA" "$MERGED"

echo "[train] 3/4 quantizing to MXFP8 F2K..."
mkdir -p "$DEST/transformer_mxfp8"
./build/f2k_convert "$MERGED" "$DEST/transformer_mxfp8/shard-00001.f2k1" --quant mxfp8 | tail -2
rm -f "$MERGED"

echo "[train] 4/4 installing model root $DEST..."
cp "$BASE/f2k_model.json" "$DEST/"
for d in qwen3_f2k vae_f2k tokenizer; do ln -sfn "$BASE/$d" "$DEST/$d"; done

# Register the @mention -> trigger-phrase the web UI substitutes ($PHRASE already
# carries the trigger token + class anchor, so it drops cleanly into any sentence).
$PY - "$DEST/trigger.json" "$NAME" "$PHRASE" <<'PYEOF'
import json, sys
json.dump({"mention": sys.argv[2], "phrase": sys.argv[3]}, open(sys.argv[1], "w"), indent=2)
PYEOF
echo "[train] web UI: type @$NAME in a prompt to summon this character."

echo "[train] DONE — '$(basename "$DEST")' is now in the web UI model dropdown."
echo "[train] NOTE: this baked the FINAL step. Quality usually peaks earlier —"
echo "       bake checkpoints in $LORA_DIR (merge_lora.py + f2k_convert) and"
echo "       scale-sweep (merge_lora.py --scale 0.7..0.9) to pick the best."
echo "[train] Try: ./build/generate --model $DEST --prompt \"a photo of $PHRASE, on the beach at sunset\" --res 1024 --precision fp8 --steps 8 --out /tmp/${NAME}_test.png"
