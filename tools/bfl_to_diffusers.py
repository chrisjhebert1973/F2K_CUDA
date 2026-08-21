#!/usr/bin/env python3
"""Remap a BFL/ComfyUI single-file FLUX.2-klein transformer checkpoint to the
diffusers tensor naming that f2k_convert / TensorRouter expect.

Differences handled:
  - double_blocks.N.{img,txt}_attn.qkv  ->  split into to_q/to_k/to_v (img)
    and add_{q,k,v}_proj (txt)
  - final_layer.adaLN_modulation.1      ->  norm_out.linear with the
    shift/scale halves swapped (BFL stores [shift;scale], diffusers [scale;shift])
  - plain renames for everything else (see MAP/PATTERNS below)
  - norm tensors named either *.scale (BFL original) or *.weight (some finetunes)

Usage:
  bfl_to_diffusers.py in.safetensors out.safetensors
  bfl_to_diffusers.py in.safetensors --verify ~/models/flux2-klein-9B/transformer
"""
import argparse, glob, json, os, re, struct, sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

GLOBALS = {
    "img_in.weight":                        "x_embedder.weight",
    "txt_in.weight":                        "context_embedder.weight",
    "time_in.in_layer.weight":              "time_guidance_embed.timestep_embedder.linear_1.weight",
    "time_in.out_layer.weight":             "time_guidance_embed.timestep_embedder.linear_2.weight",
    "double_stream_modulation_img.lin.weight": "double_stream_modulation_img.linear.weight",
    "double_stream_modulation_txt.lin.weight": "double_stream_modulation_txt.linear.weight",
    "single_stream_modulation.lin.weight":  "single_stream_modulation.linear.weight",
    "final_layer.linear.weight":            "proj_out.weight",
    # final_layer.adaLN_modulation.1.weight handled specially (swap halves)
}

DBL = {  # double_blocks.N.<key> -> transformer_blocks.N.<value>
    "img_attn.proj.weight":       "attn.to_out.0.weight",
    "txt_attn.proj.weight":       "attn.to_add_out.weight",
    "img_attn.norm.query_norm":   "attn.norm_q.weight",
    "img_attn.norm.key_norm":     "attn.norm_k.weight",
    "txt_attn.norm.query_norm":   "attn.norm_added_q.weight",
    "txt_attn.norm.key_norm":     "attn.norm_added_k.weight",
    "img_mlp.0.weight":           "ff.linear_in.weight",
    "img_mlp.2.weight":           "ff.linear_out.weight",
    "txt_mlp.0.weight":           "ff_context.linear_in.weight",
    "txt_mlp.2.weight":           "ff_context.linear_out.weight",
}

SGL = {  # single_blocks.N.<key> -> single_transformer_blocks.N.<value>
    "linear1.weight":             "attn.to_qkv_mlp_proj.weight",
    "linear2.weight":             "attn.to_out.weight",
    "norm.query_norm":            "attn.norm_q.weight",
    "norm.key_norm":              "attn.norm_k.weight",
}


def swap_scale_shift(w):
    shift, scale = w.chunk(2, dim=0)
    return torch.cat([scale, shift], dim=0)


def remap(src_path):
    out = {}
    with safe_open(src_path, framework="pt") as f:
        names = list(f.keys())
        for name in names:
            t = f.get_tensor(name)
            key = re.sub(r"\.(scale|weight)$", "", name)  # match norms by stem

            if name in GLOBALS:
                out[GLOBALS[name]] = t
                continue
            if name == "final_layer.adaLN_modulation.1.weight":
                out["norm_out.linear.weight"] = swap_scale_shift(t)
                continue

            m = re.match(r"(double|single)_blocks\.(\d+)\.(.+)", key)
            if not m:
                raise SystemExit(f"unmapped tensor: {name}")
            kind, idx, rest = m.group(1), m.group(2), m.group(3)
            rest_w = rest + ".weight" if name.endswith(".weight") and not rest.endswith("_norm") else rest

            if kind == "double":
                if rest == "img_attn.qkv":
                    q, k, v = t.chunk(3, dim=0)
                    p = f"transformer_blocks.{idx}.attn."
                    out[p + "to_q.weight"], out[p + "to_k.weight"], out[p + "to_v.weight"] = q, k, v
                elif rest == "txt_attn.qkv":
                    q, k, v = t.chunk(3, dim=0)
                    p = f"transformer_blocks.{idx}.attn."
                    out[p + "add_q_proj.weight"], out[p + "add_k_proj.weight"], out[p + "add_v_proj.weight"] = q, k, v
                else:
                    suffix = DBL.get(rest_w) or DBL.get(rest)
                    if suffix is None:
                        raise SystemExit(f"unmapped tensor: {name}")
                    out[f"transformer_blocks.{idx}.{suffix}"] = t
            else:
                suffix = SGL.get(rest_w) or SGL.get(rest)
                if suffix is None:
                    raise SystemExit(f"unmapped tensor: {name}")
                out[f"single_transformer_blocks.{idx}.{suffix}"] = t
    return out


def load_reference(ref_dir):
    ref = {}
    for shard in sorted(glob.glob(os.path.join(ref_dir, "*.safetensors"))):
        with safe_open(shard, framework="pt") as f:
            for k in f.keys():
                ref[k] = f.get_tensor(k)
    return ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output", nargs="?")
    ap.add_argument("--verify", metavar="DIFFUSERS_DIR",
                    help="compare remapped tensors bit-exactly against a diffusers transformer dir")
    args = ap.parse_args()
    if not args.output and not args.verify:
        ap.error("need an output path or --verify")

    out = remap(args.input)
    print(f"remapped {args.input}: {len(out)} tensors")

    if args.verify:
        ref = load_reference(args.verify)
        missing = sorted(set(ref) - set(out))
        extra = sorted(set(out) - set(ref))
        bad = []
        for k in sorted(set(ref) & set(out)):
            if ref[k].shape != out[k].shape:
                bad.append((k, "shape", tuple(out[k].shape), tuple(ref[k].shape)))
            elif not torch.equal(ref[k], out[k]):
                nbad = (ref[k] != out[k]).sum().item()
                bad.append((k, f"{nbad}/{ref[k].numel()} elems differ", None, None))
        for k in missing: print(f"  MISSING  {k}")
        for k in extra:   print(f"  EXTRA    {k}")
        for k, why, a, b in bad: print(f"  MISMATCH {k}: {why}" + (f" got {a} want {b}" if a else ""))
        if missing or extra or bad:
            sys.exit(f"verify FAILED ({len(missing)} missing, {len(extra)} extra, {len(bad)} mismatched)")
        print(f"verify OK: all {len(ref)} tensors bit-exact")

    if args.output:
        save_file(out, args.output)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
