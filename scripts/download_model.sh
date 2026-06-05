#!/usr/bin/env bash
# Pull FLUX.2-klein-9B from Hugging Face into ~/models/.
#
# Prerequisites (one-time):
#   1. pip install --user "huggingface_hub[cli]"   # gives you the `hf` CLI
#   2. Browser: accept the gated license at
#      https://huggingface.co/black-forest-labs/FLUX.2-klein-9B
#   3. Auth — pick one:
#        - export HF_TOKEN=hf_xxx                  (env var, preferred)
#        - hf auth login                           (interactive)
#
# Usage:
#   scripts/download_model.sh                 # full repo (~25 GB)
#   scripts/download_model.sh transformer     # just MMDiT weights (~18 GB)
#   scripts/download_model.sh vae             # just VAE (~300 MB)
#   scripts/download_model.sh text_encoder    # just T5 (~5 GB)
#
# Output lands in: $MODEL_DIR (default ~/models/flux2-klein-9B)

set -euo pipefail

REPO="black-forest-labs/FLUX.2-klein-9B"
MODEL_DIR="${MODEL_DIR:-$HOME/models/flux2-klein-9B}"
SCOPE="${1:-all}"

if ! command -v hf >/dev/null 2>&1; then
    echo "ERROR: 'hf' CLI not installed."
    echo "  Run:  pip install --user 'huggingface_hub[cli]'"
    echo "  Then ensure ~/.local/bin is on PATH (already is, in this shell)."
    exit 1
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
    # No env-var token — verify an interactive login is in place.
    if ! hf auth whoami >/dev/null 2>&1; then
        echo "ERROR: not authenticated."
        echo "  Either:  export HF_TOKEN=hf_xxx"
        echo "  Or:      hf auth login"
        exit 1
    fi
fi

mkdir -p "$MODEL_DIR"

case "$SCOPE" in
    all)
        echo ">>> Downloading entire $REPO (~25 GB) to $MODEL_DIR"
        hf download "$REPO" --local-dir "$MODEL_DIR"
        ;;
    transformer)
        echo ">>> Downloading transformer/ (~18 GB) to $MODEL_DIR"
        hf download "$REPO" --include "transformer/*" --local-dir "$MODEL_DIR"
        ;;
    vae)
        echo ">>> Downloading vae/ (~300 MB) to $MODEL_DIR"
        hf download "$REPO" --include "vae/*" --local-dir "$MODEL_DIR"
        ;;
    text_encoder|t5)
        echo ">>> Downloading text_encoder/ + tokenizer/ (~5 GB) to $MODEL_DIR"
        hf download "$REPO" --include "text_encoder/*" --include "tokenizer/*" \
            --local-dir "$MODEL_DIR"
        ;;
    *)
        echo "Unknown scope: $SCOPE  (try: all | transformer | vae | text_encoder)"
        exit 1
        ;;
esac

echo ""
echo ">>> Done. Top-level layout:"
find "$MODEL_DIR" -maxdepth 2 -type f \( -name "*.safetensors" -o -name "*.json" \) \
    -exec ls -lh {} \; | head -20
