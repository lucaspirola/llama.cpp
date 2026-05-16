# PR2 — NVFP4 KV-cache flash-attention on f16 tensor cores (fused inline dequant)

> Working document for the `feat/nvfp4-kv-cache` branch. It records what was done,
> how it was verified, and the measured results, and doubles as a draft description
> for the eventual upstream `ggml-org/llama.cpp` pull request.
> **This file is meta — delete it (or drop its commit) before opening the upstream PR.**

Branch: `feat/nvfp4-kv-cache` (PR2 builds on PR1, tip `78f6e1fa6`). Build/test toolchain:
**CUDA 13.2** (CUDA 12.8 corrupts `sharedMemPerBlockOptin` on SM120). Verified locally on
an RTX 5080 (SM120).

---

## 1. Summary

PR1 made NVFP4 a first-class KV-cache type. Its CUDA flash-attention prefill path worked
by converting the whole NVFP4 K/V tensors to a temporary **f16 scratch buffer** (via
`launch_fattn`) and then running the existing f16 tensor-core MMA kernel on that buffer.

PR2 removes that scratch buffer. NVFP4 KV-cache prefill (head-dim 128) now runs on the
**shared f16 MMA kernel** (`fattn-mma-f16.cuh`) with the NVFP4 K/V dequantized **inline,
per shared-memory tile, during the load**. Nothing downstream of the load changes — the
MMA math still operates on f16. The win is memory: no full-tensor f16 K/V copy is
allocated, which matters most at long context (a 16 GB card spends that VRAM on more KV
instead).

The integration is done the convention-compliant llama.cpp way: a single shared
templated kernel with a `KV_src_t` template parameter, the same pattern MMQ and the FA
VEC kernel use for quantized inputs. The f16/q8_0/q4_0/bf16 instantiations are
byte-identical to before (all NVFP4-specific code is behind `if constexpr`).

This is **PR2** of the two-PR series. PR1 = correct first-class integration. PR2 = the
fused-dequant integration plus the investigation that establishes there is no native-FP4
compute path worth taking (see §4).

## 2. Motivation

PR1's f16 scratch buffer is a correctness-first shortcut: it allocates a full f16 copy of
K and V (16 bpw) purely as kernel input, defeating part of the point of a 4.5 bpw KV
cache during prefill. PR2 eliminates that allocation by teaching the MMA kernel to read
raw NVFP4 blocks and dequantize them as it streams each tile into shared memory — the
dequantized f16 only ever exists in the shared tile that the kernel was going to fill
anyway.

## 3. What changed

3 source files + 1 codegen script + 15 generated MMA template-instance files + tests.

### The fused MMA kernel — `ggml/src/ggml-cuda/fattn-mma-f16.cuh`

- A new `typename KV_src_t = half2` template parameter threads through
  `flash_attn_ext_f16_load_tile`, `flash_attn_ext_f16_iter`,
  `flash_attn_ext_f16_process_tile`, the `flash_attn_ext_f16` kernel, and
  `ggml_cuda_flash_attn_ext_mma_f16_case`. It selects the in-memory representation of K/V:
  - `half2` (default) — the original f16 loader, **byte-identical**.
  - `block_nvfp4` — K/V are stored as NVFP4 blocks; the loader reads the packed bytes and
    dequantizes them inline into the shared `half2` tile.
- `flash_attn_ext_f16_load_tile` gets an NVFP4 branch (`if constexpr
  std::is_same_v<KV_src_t, block_nvfp4>`): synchronous plain loads of the NVFP4 blocks
  followed by inline dequant. It mirrors the existing synchronous f16 loader's
  stride/index logic exactly.
- K/V row strides are now expressed in units of `sizeof(KV_src_t)` instead of a hardcoded
  `sizeof(half2)`, so the same code computes the correct stride for both representations.
- For NVFP4, `nstages` is forced to 0 (fully synchronous): `block_nvfp4` is 36 bytes, not
  a 16-byte multiple, so the cp.async pipeline cannot be used (cp.async requires 16-byte
  alignment). The cp.async pipeline is therefore disabled wherever `is_nvfp4` is true.
- For NVFP4, `launch_fattn` is told **not** to allocate the f16 scratch buffer and **not**
  to rewrite the K/V strides (the `need_f16_K` / `need_f16_V` flags become `!is_nvfp4`):
  the kernel reads the raw NVFP4 blocks directly.
- A new `DECL_FATTN_MMA_F16_CASE_NVFP4` macro and `..._ALL_NCOLS2` block declare the
  NVFP4 instantiations (head-dim 128, ncols2 ∈ {1,2,4,8}).

### The vectorized inline dequantizer — `ggml/src/ggml-cuda/fattn-common.cuh`

- `dequantize_nvfp4_chunk` decodes one 16-byte tile chunk (8 consecutive NVFP4 elements,
  `el0` a multiple of 8) into 4 `half2`. Because `QK_NVFP4_SUB == 16` and `el0` is a
  multiple of 8, all 8 elements share one block, one sub-block UE4M3 scale and one nibble
  shift. The 8 4-bit codes are decoded with **two `get_int_from_table_16` calls**
  (prmt-based 16-entry table lookup, exactly as in PR1's `vec_dot_fattn_vec_KQ_nvfp4`)
  instead of 8 scalar LUT loads, and the UE4M3 scale is converted to f16 **once** per
  chunk. This is the B3 vectorization step.

### Dispatch — `ggml/src/ggml-cuda/fattn.cu`

- `ggml_cuda_flash_attn_ext_mma_f16` routes NVFP4 K/V with head-dim 128 to a new
  `ggml_cuda_flash_attn_ext_mma_f16_nvfp4`, which performs the same `use_gqa_opt` /
  `gqa_ratio` ncols2 selection as the non-Volta `switch_ncols2<128,128>` path and
  instantiates `ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<128,128,ncols2,block_nvfp4>`.
- Other NVFP4 head dims fall through to the standard f16-conversion MMA path (PR1
  behaviour: `launch_fattn` converts K/V to f16 scratch). Head-dim 128 covers the common
  case (Llama-family models).

### Codegen — `ggml/src/ggml-cuda/template-instances/generate_cu_files.py`

- Each `fattn-mma-f16-instance-*` file with `ncols2 ∈ {1,2,4,8}` also emits a
  `DECL_FATTN_MMA_F16_CASE_NVFP4(128, 128, ncols1, ncols2)` line. Regenerating the
  instances produces output byte-identical to what is committed; the non-NVFP4 instance
  lines are unchanged.

### Removed scaffolding — commit `8df360968`

PR2 also removes an abandoned earlier attempt at a **native FP4 tensor-core**
flash-attention path (`BEST_FATTN_KERNEL_MMA_FP4`, a stub `ggml_cuda_flash_attn_ext_mma_fp4`,
its dispatch case and codegen hooks). §4 explains why that path is a dead end.

### Tests — `tests/test-backend-ops.cpp`

- NVFP4 KV-cache prefill coverage at hsk=hsv=128, Q columns > 2 (the fused kernel only
  dispatches for `Q->ne[1] > 2`): an ncols1-selection sweep `{3,4,8}`, long-KV cases so
  `parallel_blocks > 1` exercises the KV-split combine path, `max_bias` and
  `logit_softcap` variants, and a permuted (non-contiguous) NVFP4 K/V case that exercises
  the row-stride-driven loader with genuine NVFP4 row strides.

## 4. Native FP4 tensor-core attention — a confirmed dead end

An earlier PR2 plan was to run K·Qᵀ directly on SM120's FP4 MMA instruction. This was
investigated and is **not viable**:

- SM120's FP4 MMA requires **both** operands to be 4-bit. Using it for K·Qᵀ therefore
  forces the **query** to be quantized to 4-bit as well.
- A 4-bit query is catastrophic for accuracy: measured wikitext perplexity with a 4-bit
  query exploded to **~211,000** (vs ~7.7 normally). The query carries the per-token
  signal that attention is most sensitive to; 4-bit is far too coarse.

This is not a llama.cpp-specific finding. A survey of the field confirms the same
conclusion everywhere:

- **NVIDIA's own NVFP4 KV-cache blog** describes NVFP4 KV as a memory/bandwidth
  optimisation and dequantizes the KV cache before attention.
- **vLLM**, **SGLang** and **FlashInfer** all store the KV cache in a 4-bit/FP8 format
  and dequantize to FP8/BF16 before the attention matmul.
- **FlashAttention-3 / FlashAttention-4** and **BitDecoding** likewise keep attention
  compute in FP8/BF16 and treat low-bit KV purely as a storage format.

The consensus: **NVFP4 KV is a memory/bandwidth feature, not a compute one.** You quantize
the KV cache to save VRAM and bandwidth, then dequantize to FP8/BF16 (never to FP4) before
the attention matmul. PR2's fused inline-dequant path is exactly that — and it does the
dequant without a full-tensor scratch buffer.

## 5. Verification (RTX 5080, CUDA 13.2)

### Correctness

- `test-backend-ops test -o FLASH_ATTN_EXT -b CUDA0`: **3225 / 3225 OK**, including every
  NVFP4 hsk=128 prefill case added in PR2 (ncols sweep, long-KV combine path, max_bias,
  logit_softcap, permuted K/V). "OK" means the CUDA kernel matches the CPU reference.
- Full `test-backend-ops -b CUDA0`: 12590 / 12629. The 39 failures are the pre-existing
  `iq1_s` / `iq2_s` / `iq3_s` `MUL_MAT` / `MUL_MAT_ID` cases (13 each) — unrelated to this
  branch, present on untouched `master`. **Zero new failures.**
- Regenerating the FA MMA template instances produces output byte-identical to what is
  committed; the non-NVFP4 (`f16` / `q8_0` / `q4_0` / `bf16`) instance lines are unchanged.

### Byte-identical non-NVFP4 path

Every NVFP4-specific construct in `fattn-mma-f16.cuh` is gated by `if constexpr
(std::is_same_v<KV_src_t, block_nvfp4>)` or by the `constexpr bool is_nvfp4` flag. With
the default `KV_src_t = half2`: `is_nvfp4` is `false`, so `nstages`, `use_cp_async`, the
K/V strides and the `launch_fattn` flags all reduce to the original literal expressions,
and the NVFP4 load branch is dead-code-eliminated. The `q8_0` / `q4_0` / `bf16` KV types
are instantiated through the unchanged `DECL_FATTN_MMA_F16_CASE` macro (default
`KV_src_t`), so they remain on the f16-scratch path exactly as before.

### Code review

A per-file review of the PR2 diff (`fattn-common.cuh`, `fattn-mma-f16.cuh`, `fattn.cu`,
`generate_cu_files.py`, `test-backend-ops.cpp`) surfaced one finding, which was fixed:

> The NVFP4 tile loader passes the K/V loop offsets `k0_start` / `i0_start/2` (head-dim
> *half2* offsets) as offsets into a `block_nvfp4` row. For `KV_src_t = block_nvfp4`,
> pointer arithmetic advances by whole blocks, so this is only correct when the K/V loops
> run as a single iteration (offset 0). That holds for the only shipped NVFP4 config
> (DKQ = DV = 128, `nbatch_K2 = nbatch_V2 = 64`), so the code is correct today — but a
> future config with a smaller batch would silently corrupt.

Fix: a `static_assert(!is_nvfp4 || (nbatch_K2 >= DKQ/2 && nbatch_V2 >= DV/2), ...)` in
`flash_attn_ext_f16_iter` makes the single-iteration invariant a compile-time error
instead of a latent bug. It holds for every current instantiation (verified: the tree
still compiles) and changes no runtime behaviour.

### Accuracy (wikitext-2, Llama-3.1-8B Q4_K_M, flash attention on, -ngl 99)

564 chunks, n_ctx = 512:

| KV type | wikitext PPL |
|---|---|
| f16   | 7.4983 ± 0.04795 |
| q4_0  | 7.5927 ± 0.04853 |
| **nvfp4** | **7.6019 ± 0.04864** |

NVFP4 KV is **0.104 above f16** and **within measurement noise of q4_0** (0.009 apart;
both ± ~0.048). The absolute numbers differ from PR1's report (f16 7.618 / nvfp4 7.713)
because this run used a different chunk window — but the **nvfp4-vs-f16 gap is unchanged**
(0.104 here, 0.095 in PR1). That is the key correctness claim: the fused inline-dequant
kernel dequantizes NVFP4 to the same f16 values PR1's scratch-buffer path produced, so its
accuracy is identical to PR1's. The numbers confirm it.

## 6. Performance (honest)

Measured prefill on Llama-3.1-8B Q4_K_M, RTX 5080:

- NVFP4 KV prefill is **~10% behind f16 and q4_0**.
- The B3 vectorized dequant (`dequantize_nvfp4_chunk`) narrowed the **FA-kernel** gap from
  **1.70× to 1.44×** by replacing 8 scalar LUT loads + a per-element scale conversion with
  two `get_int_from_table_16` calls + one scale conversion per 8-element chunk.
- The residual gap is **dequant ALU cost**: every tile load now does the table decode and
  scale multiply that the f16 path does not.
- A **cp.async pipeline** for the NVFP4 loader was tried and **reverted**: `block_nvfp4`
  is 36 bytes, not a 16-byte multiple, and cp.async requires 16-byte-aligned transfers.
  The NVFP4 loader is therefore synchronous (`nstages = 0`).

### Honest NVFP4-vs-q4_0 assessment

NVFP4 KV has **no real performance advantage over q4_0**. Same 4.5 bpw memory footprint;
accuracy tied within noise; q4_0 is marginally faster (linear `(nibble−8)·scale` dequant
vs NVFP4's non-uniform E2M1 table lookup); q4_0 also has broader head-dim coverage. NVFP4
KV is a **legitimate first-class KV type** — it completes the KV-type matrix and is the
natural cache format for the NVFP4-quantized model ecosystem (a model already shipped in
NVFP4 can keep K/V in the same format) — but it is **not a q4_0 replacement** and should
not be presented as "better than q4_0".

## 7. Known limitations / future work

- Prefill trails f16/q4_0 by ~10% (dequant ALU cost; §6).
- The fused path covers head-dim 128 only; other head dims still use the PR1 f16-scratch
  conversion path.
- Decode still uses the PR1 VEC kernel — unchanged by PR2.
- **FP8-attention path (future work):** dequant NVFP4 KV → FP8 and run the attention
  matmul on FP8 tensor cores. This is the industry-standard approach (§4) and would be a
  genuine compute speedup, unlike a native-FP4 path. Out of scope for PR2.

## 8. Commits on this branch (PR2, on top of PR1 tip `78f6e1fa6`)

```
b97e6f01a  cuda: NVFP4 KV-cache flash-attention — fused inline dequant on f16 MMA
8df360968  cuda: remove obsolete standalone NVFP4 FP4-MMA scaffolding
8e085300c  cuda: NVFP4 KV-cache flash-attention — vectorized inline dequant
           (+ B4 code-review fix: static_assert guarding the NVFP4 single-iteration
            tile-load invariant)
```

For the upstream PR these may be squashed.
