#!/usr/bin/env bash
# Hyperparameter sweep for a character LoRA: train several (rank, lr) configs on
# the same captioned dataset, each saving checkpoints every 250 steps. Produces
# candidate LoRAs to bake + compare; does NOT install (use train_character.sh or
# a manual bake to promote the winner).
#
#   tools/sweep_character.sh <name> <captioned_dataset_dir> "<trigger phrase>" [steps]
#   tools/sweep_character.sh rocket3 ~/datasets/rocket3 \
#       "rkt9d, a black and white Akita husky dog with one blue eye and one amber eye" 1000
#
# Default matrix (edit CONFIGS below): A=r16/5e-5  B=r16/1e-4  C=r32/5e-5.
#
# Parallelism: by default runs configs sequentially on THIS box. To farm across
# Sparks, run one config per machine — pass a single "rank:lr" via $ONLY, e.g.
#   ONLY=16:5e-5 tools/sweep_character.sh rocket3 ~/datasets/rocket3 "..." 1000
# (the dataset must exist at the same path on each machine).
set -euo pipefail
cd "$(dirname "$0")/.."

NAME="${1:?usage: sweep_character.sh <name> <captioned_dataset_dir> \"<trigger phrase>\" [steps]}"
DATASET="${2:?captioned dataset dir required}"
PHRASE="${3:?trigger phrase required}"
STEPS="${4:-1000}"
CONFIGS=("16:5e-5" "16:1e-4" "32:5e-5")   # rank:lr
[ -n "${ONLY:-}" ] && CONFIGS=("$ONLY")

BASE="${BASE_MODEL:-$HOME/models/flux2-klein-4B}"   # BASE_MODEL=~/models/flux2-klein-9B to train 9B
PY=.venv/bin/python
[ -f "$DATASET/metadata.jsonl" ] || { echo "no metadata.jsonl in $DATASET — run prep_character.sh and fill captions first" >&2; exit 1; }
N_EMPTY=$($PY - "$DATASET/metadata.jsonl" <<'PYEOF'
import json,sys
print(sum(1 for l in open(sys.argv[1]) if l.strip() and not json.loads(l).get("caption","").strip()))
PYEOF
)
[ "$N_EMPTY" -eq 0 ] || { echo "$N_EMPTY captions still empty — fill them first" >&2; exit 1; }

echo "[sweep] $NAME: ${#CONFIGS[@]} config(s), $STEPS steps each, dataset $DATASET"
for cfg in "${CONFIGS[@]}"; do
  RANK="${cfg%%:*}"; LR="${cfg##*:}"
  OUT="$HOME/loras/${NAME}_sweep/r${RANK}_lr${LR}"
  LOG="/tmp/sweep_${NAME}_r${RANK}_lr${LR}.log"
  echo "[sweep] === rank $RANK, lr $LR -> $OUT (log $LOG) ==="
  HF_HOME=/tmp/hfhome .venv/bin/accelerate launch --num_processes 1 --mixed_precision bf16 \
    tools/training/train_dreambooth_lora_flux2_klein.py \
    --pretrained_model_name_or_path "$BASE" \
    --dataset_name "$DATASET" --caption_column caption \
    --output_dir "$OUT" \
    --instance_prompt "a photo of $PHRASE" \
    --resolution 1024 --train_batch_size 1 --guidance_scale 1 \
    --gradient_accumulation_steps 4 --optimizer adamW --learning_rate "$LR" \
    --lr_scheduler constant --lr_warmup_steps 0 \
    --max_train_steps "$STEPS" --checkpointing_steps 250 \
    --rank "$RANK" --lora_alpha "$RANK" \
    --gradient_checkpointing --cache_latents --mixed_precision bf16 \
    --seed 0 > "$LOG" 2>&1
  echo "[sweep] done rank $RANK lr $LR: $(ls -d "$OUT"/checkpoint-* 2>/dev/null | wc -l) checkpoints"
done
echo "[sweep] ALL DONE. Candidate LoRAs under ~/loras/${NAME}_sweep/. Bake + compare to pick the winner."
