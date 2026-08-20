#!/usr/bin/env bash
# Train ONE character-LoRA config (captioned dataset) to a local output dir.
# Used to fan single configs across the Spark cluster — see project_spark_cluster.
#   launch_one.sh <base> <dataset> <out> <lr> <rank> <steps> <ckpt_every> [instance_token]
# instance_prompt is unused in captioned mode (dataset captions win) — pass a
# no-space token to avoid SSH quoting headaches.
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root
BASE="$1"; DATASET="$2"; OUT="$3"; LR="$4"; RANK="$5"; STEPS="$6"; CKPT="$7"; IP="${8:-subject}"
HF_HOME=/tmp/hfhome .venv/bin/accelerate launch --num_processes 1 --mixed_precision bf16 \
  tools/training/train_dreambooth_lora_flux2_klein.py \
  --pretrained_model_name_or_path "$BASE" \
  --dataset_name "$DATASET" --caption_column caption \
  --output_dir "$OUT" \
  --instance_prompt "a photo of $IP" \
  --resolution 1024 --train_batch_size 1 --guidance_scale 1 \
  --gradient_accumulation_steps 4 --optimizer adamW --learning_rate "$LR" \
  --lr_scheduler constant --lr_warmup_steps 0 \
  --max_train_steps "$STEPS" --checkpointing_steps "$CKPT" \
  --rank "$RANK" --lora_alpha "$RANK" \
  --gradient_checkpointing --cache_latents --seed 0
echo "[launch_one] DONE -> $OUT"
