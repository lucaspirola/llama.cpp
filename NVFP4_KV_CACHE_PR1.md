# PR1 — NVFP4 as a first-class KV-cache type (CUDA)

> Working document for the `feat/nvfp4-kv-cache` branch. It records what was done,
> how it was verified, and the measured results, and doubles as a draft description
> for the eventual upstream `ggml-org/llama.cpp` pull request.
> **This file is meta — delete it (or drop its commit) before opening the upstream PR.**

Branch: `feat/nvfp4-kv-cache` (off `master`). Build/test toolchain: **CUDA 13.2** (CUDA 12.8
corrupts `sharedMemPerBlockOptin` on SM120). Verified locally on an RTX 5080 (SM120).

---

## 1. Summary

Adds **NVFP4** (`GGML_TYPE_NVFP4`) as a fully supported KV-cache weight type, so
`--cache-type-k nvfp4 --cache-type-v nvfp4` works end to end. NVFP4 already existed as a
*weight* quantization type; this wires it into the **KV-cache path** and the CUDA
flash-attention kernels to the same integration depth as `q8_0` — every backend guard,
every FA code path, test coverage, and graceful fallback.

NVFP4 KV cache is 4.5 bits/value vs 16 for f16 (~3.5× smaller): on a 16 GB card it is the
difference between OOM and long-context capability.

This is **PR1** of a planned two-PR series. PR1 = correct, first-class integration with
software dequant. **PR2** = SM120 native FP4 tensor-core flash-attention (the hardware
acceleration path) — out of scope here.

## 2. Motivation

A prior MXFP4 KV-cache attempt (PR #21055) was closed unmerged for being a CUDA-only
bolt-on with no test coverage and no guards on other backends. PR1 is built to *not*
repeat that: NVFP4 KV is backend-complete-or-cleanly-guarded, covered by the standard
test suite, and degrades gracefully.

## 3. What changed

15 source files + 15 generated flash-attention template instances.

### KV type registration
- `common/arg.cpp` — `GGML_TYPE_NVFP4` added to `kv_cache_types` (drives `--cache-type-k/v`
  parsing and `--help`).

### KV write path (quantize-on-write), CUDA
- `ggml-cuda/cpy-utils.cuh` — `quantize_f32_nvfp4_block` + `cpy_blck_f32_nvfp4`.
- `ggml-cuda/cpy.cu` — `ggml_cpy_f32_nvfp4_cuda` + dispatcher branch (F32→NVFP4).
- `ggml-cuda/set-rows.cu` — NVFP4 branch in the set-rows dispatcher.
- `ggml-cuda/ggml-cuda.cu` — `supports_op` entries for `GGML_OP_CPY` (F32→NVFP4) and
  `GGML_OP_SET_ROWS`. Forward-only, matching the IQ4_NL precedent.

### KV read path — CUDA flash attention
- `ggml-cuda/fattn-common.cuh` — `vec_dot_fattn_vec_KQ_nvfp4` (K·Q dot) and
  `dequantize_V_nvfp4` (V dequant) for the VEC kernel; registered in `get_vec_dot_KQ` /
  `get_dequantize_V`.
- `ggml-cuda/fattn-vec.cuh` — `EXTERN_DECL_FATTN_VEC_CASES` for NVFP4.
- `ggml-cuda/fattn.cu` — `GGML_TYPE_NVFP4` in the `switch (K->type)` kernel-selection gate
  and in the `FATTN_VEC_CASES` lists (default minimal set + `GGML_CUDA_FA_ALL_QUANTS`).
- `ggml-cuda/template-instances/generate_cu_files.py` — NVFP4 added to `TYPES_KV`;
  15 generated `fattn-vec-instance-*nvfp4*.cu` files.
- `ggml-cuda/CMakeLists.txt` — `fattn-vec-instance-nvfp4-nvfp4.cu` in the minimal build set.
- `ggml-cuda/convert.cu` — non-contiguous `dequantize_row_nvfp4_nc_cuda` registered in
  `ggml_get_to_fp16/bf16/fp32_nc_cuda`. The MMA/tile/WMMA FA paths convert quantized K/V
  to f16 first; for a non-contiguous (view) K the nc converter is required — without it
  `ggml_get_to_fp16_nc_cuda` returned `nullptr` and the MMA path would crash.

### Determinism / portability
- `ggml-cuda/common.cuh` — `ggml_cuda_fp32_to_ue4m3` is now an unconditional software
  encoder (was Blackwell-only `NO_DEVICE_CODE`). The hardware `__nv_fp8_e4m3` path used
  round-to-nearest-even and would diverge from the round-half-up CPU reference, breaking
  the bit-exact CPU/GPU parity the NVFP4 quantizer relies on.

### Quantizer accuracy improvement
- `ggml-quants.c` (`quantize_row_nvfp4_ref`) and the CUDA mirror — the per-sub-block
  UE4M3 scale is now chosen by a small min-SSE search around `amax/6` instead of a fixed
  value. NVFP4's 8-bit UE4M3 scale is the dominant error source vs q4_0's f16 scale; the
  search recovers most of the gap. Also benefits NVFP4 weight quantization.

### Tooling / tests
- `tests/test-backend-ops.cpp` — NVFP4 in the flash-attention `type_KV` enumeration plus
  standalone hsk=128 and mixed NVFP4/F16 cases.
- `tools/llama-bench/llama-bench.cpp` — `nvfp4` accepted by `--cache-type-k/-v`.

### Other backends
No code change needed. CPU computes NVFP4 KV via the existing `dequantize_row_nvfp4`
fallback. Vulkan / SYCL / Metal already cleanly reject NVFP4 KV via their flash-attention
type whitelists (verified) — no crash.

## 4. Verification (RTX 5080, CUDA 13.2)

### Correctness
- `test-backend-ops -b CUDA0`: NVFP4 flash-attention **333 OK / 0 fail**; CPY 5 OK;
  SET_ROWS 12 OK. The FA "OK" verdicts mean CUDA matches the CPU reference.
- Full `test-backend-ops` on CUDA0: the only failures are 42 pre-existing
  `iq1_s/iq2_s/iq3_s` `MUL_MAT` cases — **verified identical on an untouched `master`
  build**, i.e. zero regressions from this branch.
- e2e: `llama-cli` / `llama-completion` produce coherent output with `--cache-type-k/v
  nvfp4` on GPU and on CPU (`-ngl 0` fallback).

### Code review
4-round per-file code-review loop over all 15 source files + 15 generated templates
until every file returned no findings (commits `97e010b8` / `c868004b` / `8eb2b72a`).

### Accuracy (wikitext-2, flash attention on)
| KV type | Llama-3.1-8B Q4_K_M | Llama-3.2-1B Q8_0 |
|---|---|---|
| f16   | 7.618 | 13.996 |
| q8_0  | 7.619 | 13.999 |
| q4_0  | 7.704 | 14.652 |
| **nvfp4** | **7.713** | **14.590** |

With the min-SSE scale search, NVFP4 KV **beats q4_0 on 1B and ties it on 8B**;
q8_0 KV remains near-lossless.

### Memory
NVFP4 KV = 4.5 bpw. Demonstrated: 8B model at 131072 context on a 16 GB RTX 5080 —
f16 KV OOMs (16 GiB KV alloc fails), NVFP4 KV loads and runs.

## 5. Performance (llama-bench, Llama-3.1-8B Q4_K_M, FA on, -r 5)

| KV type | prefill pp512 (t/s) | decode tg128 (t/s) |
|---|---|---|
| f16   | 8696 | 163.4 |
| q8_0  | 8436 | 152.3 |
| q4_0  | 8449 | 152.5 |
| nvfp4 | 7895 | 100.7 |

Prefill is ~6% slower than q4_0; decode is ~34% slower. This is **structural**: NVFP4's
non-uniform E2M1 values require a table lookup (`get_int_from_table_16`) in the VEC kernel
that q4_0's linear `(nibble−8)·scale` dequant does not. A safe instruction-level
micro-optimization of the software-dequant path was investigated and does not close the
gap (the lookup, not the loads, dominates). Decode-speed work belongs in **PR2**, where
the SM120 FP4 tensor-core path removes software dequant entirely.

## 6. Known limitations / out of scope for PR1

- Decode throughput trails q4_0 (see §5) — PR2.
- SM120 native FP4 tensor-core flash attention — PR2.
- NVFP4 KV on Vulkan/Metal/SYCL is cleanly rejected, not accelerated (acceptable; the
  brief explicitly allows clean rejection).

## 7. Upstream-readiness checklist

- [x] Backend-complete or cleanly guarded — CUDA full; CPU via dequant fallback;
      Vulkan/SYCL/Metal cleanly reject.
- [x] Correct on all CUDA FA paths — VEC native; MMA/tile/WMMA via f16 conversion
      (contiguous + non-contiguous).
- [x] Graceful fallback — verified CPU-only and non-CUDA.
- [x] Test coverage — `test-backend-ops` flash-attention exercises NVFP4 K/V.
- [x] Documented — `--cache-type-k/v` help auto-lists `nvfp4`.
- [x] Minimal, idiomatic diff — matches the `q8_0` / `iq4_nl` patterns; no dead code,
      no "experimental" naming.
- [x] No regressions — full `test-backend-ops` diffed against `master`.

## 8. Commits on this branch

```
8747dbdbb  cuda: add NVFP4 as a first-class KV-cache type
97e010b82  cuda: address NVFP4 KV-cache code-review round 1
c868004ba  cuda: address NVFP4 KV-cache code-review round 2
8eb2b72ae  cuda: address NVFP4 KV-cache code-review round 3
12bf31131  llama-bench: accept nvfp4 for --cache-type-k/-v
```

For the upstream PR these may be squashed; the code-review rounds need not appear as
separate commits in upstream history.
