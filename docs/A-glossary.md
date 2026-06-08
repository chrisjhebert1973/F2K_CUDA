# Appendix A — Glossary

Terms and acronyms used in the course, with the chapter that introduces each.

## Models & math

- **Diffusion model** — generative model that learns to reverse a noising process;
  samples by denoising from noise. (Ch 1)
- **Score** — $\nabla_x \log p(x)$, the gradient of log-density; the quantity a
  diffusion model implicitly learns. (Ch 1)
- **Tweedie's formula** — relates the optimal denoiser (posterior mean of clean given
  noisy) to the score. (Ch 1)
- **DSM** — denoising score matching; the training objective equivalence that makes
  diffusion trainable. (Ch 1)
- **Flow matching** — learn a velocity field whose flow transports noise to data; the
  framework FLUX trains with. (Ch 2)
- **Rectified flow** — flow matching with straight-line conditional paths; constant
  velocity target $x_1-x_0$. (Ch 2)
- **Probability-flow ODE** — deterministic ODE with the same marginals as the reverse
  diffusion SDE; sampling = integrating it. (Ch 1–2)
- **Euler step** — first-order ODE integration step; `latent += dt·v`. (Ch 2)
- **Distillation** — training a student to match many teacher steps in few; why klein
  needs 4 steps. (Ch 2)
- **Dynamic shift** — resolution-aware remapping of the timestep schedule. (Ch 2)
- **VAE** — variational autoencoder; compresses image↔latent. (Ch 3)
- **ELBO** — evidence lower bound; the VAE training objective. (Ch 3)
- **Latent** — the compressed representation diffusion operates in ($[32,H/8,W/8]$). (Ch 3)

## Architecture

- **MMDiT** — Multi-Modal Diffusion Transformer; FLUX's two-stream architecture. (Ch 4)
- **Token** — a vector the transformer processes (image patch or text vector). (Ch 4)
- **Patchify** — pack a $2\times2$ latent block into one 128-dim token. (Ch 3)
- **Double-stream block** — separate text/image weights, joint attention; ×8. (Ch 8)
- **Single-stream block** — unified weights, fused parallel attn+MLP; ×24. (Ch 9)
- **Joint attention** — attention over concatenated `[txt‖img]` tokens. (Ch 8)
- **RoPE** — rotary position embedding; rotate Q/K by position-dependent angles. (Ch 6)
- **4-axis RoPE** — RoPE partitioned into (T,H,W,L) axes for 2D images + text. (Ch 6)
- **AdaLN / modulation** — adaptive layer norm; per-block scale/shift/gate from the
  timestep. (Ch 7)
- **QK-norm** — per-head RMS-normalization of Q and K before attention. (Ch 8)
- **SwiGLU** — gated MLP activation, $\text{SiLU}(g)\cdot u$. (Ch 8, 15)
- **RMSNorm / LayerNorm / GroupNorm** — normalizations (Ch 15).
- **GQA** — grouped-query attention; fewer KV heads than query heads (Qwen3: 32/8). (Ch 21)
- **Context embedder** — Linear projecting `[512×12288]` conditioning → hidden 4096. (Ch 5)

## GPU / numerics

- **GB10 / Blackwell / sm_121** — the target GPU and its compute capability. (Ch 11)
- **Unified memory** — coherent CPU+GPU memory; no PCIe copy of weights. (Ch 11, 14)
- **Roofline** — performance bound $\min(P_{\max}, B_{\max}\cdot I)$. (Ch 11)
- **Arithmetic intensity ($I$)** — FLOPs per byte of HBM traffic. (Ch 11)
- **Ridge point** — $I_{\text{ridge}}=P_{\max}/B_{\max}$; memory- vs compute-bound
  boundary. (Ch 11)
- **Memory-bound / compute-bound / occupancy-bound** — which ceiling limits a kernel.
  (Ch 11, 17)
- **BF16** — 16-bit float, E8M7; activation baseline. (Ch 12)
- **FP8 E4M3 / MXFP8** — 8-bit float; MXFP8 = E4M3 values + per-32 UE8M0 scales. (Ch 12)
- **NVFP4 / E2M1** — 4-bit float (values $\{0,.5,1,1.5,2,3,4,6\}$) + per-16 E4M3
  scales. (Ch 12)
- **Block / microblock scaling** — per-small-group scales so low precision survives
  outliers. (Ch 12)
- **Quantization** — mapping high-precision values to a low-precision grid + scale.
  (Ch 12)
- **GEMM** — general matrix multiply; what every `Linear`/conv reduces to. (Ch 13, 19)
- **Tensor core / MMA** — hardware matrix-multiply-accumulate unit/instruction. (Ch 13, 18)
- **CUTLASS / CuTe** — NVIDIA's GEMM template library / its tensor-layout layer. (Ch 13)
- **`mma.sync`** — raw tensor-core MMA PTX instruction (m16n8k16 here). (Ch 18)
- **`ldmatrix`** — cooperative shared-memory→fragment load PTX. (Ch 18)
- **Fragment** — a thread-distributed tile of a matrix in registers. (Ch 18)
- **Occupancy / CTA / SM** — resident blocks per streaming multiprocessor. (Ch 11, 17)
- **Online softmax** — single-pass softmax with running max/sum/output. (Ch 16)
- **Flash attention** — tiled attention that streams K/V once, never materializing
  $P$. (Ch 16–17)
- **cuDNN graph API** — the modern cuDNN interface reaching Blackwell conv engines. (Ch 19)
- **NHWC / NCHW / KRSC** — tensor/filter memory layouts. (Ch 19)

## Project-specific

- **F2K (`.f2k1`)** — the custom mmap-and-go weight container. (Ch 14)
- **TensorRouter** — maps tensor names → structural roles. (Ch 14)
- **`Linear`** — the project's quantized linear layer over `FP4Gemm`/`FP8Gemm`. (Ch 13)
- **`FlowMatchScheduler`** — the sampler schedule (`flux2_dynamic`). (Ch 2)
- **`generate`** — the end-to-end CLI: prompt → PNG. (Ch 23)
- **`f2k_convert`** — offline safetensors → quantized F2K. (Ch 12, 14)
- **`bench_attention` / `mma_unit`** — the attention benchmark / PTX-layout validator.
  (Ch 17, 18)
- **`cmp_transformer` / `*_golden` / `diffusers_*`** — reference-comparison tools. (Ch 25)
- **Capture layers `{8,17,26}`** — Qwen3 layer outputs concatenated into conditioning.
  (Ch 5, 21)
- **Zero-gate identity test** — bit-exact wiring test using AdaLN-Zero. (Ch 8, 25)
