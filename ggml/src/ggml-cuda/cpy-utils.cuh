#pragma once

#include "ggml-common.h"
#include "convert.cuh"

static __device__ __forceinline__ int best_index_int8(int n, const int8_t * val, float x) {
    if (x <= val[0]) return 0;
    if (x >= val[n-1]) return n-1;
    int ml = 0, mu = n-1;
    while (mu-ml > 1) {
        int mav = (ml+mu)/2;
        if (x < val[mav]) mu = mav; else ml = mav;
    }
    return x - val[mu-1] < val[mu] - x ? mu-1 : mu;
}

static __device__ void quantize_f32_q4_0_block(const float * __restrict__ x, block_q4_0 * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK4_0; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    const float d  = vmax / -8;
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    for (int j = 0; j < QK4_0/2; ++j) {
        const float x0 = x[0       + j]*id;
        const float x1 = x[QK4_0/2 + j]*id;

        const uint8_t xi0 = min(15, (int8_t)(x0 + 8.5f));
        const uint8_t xi1 = min(15, (int8_t)(x1 + 8.5f));

        y->qs[j]  = xi0;
        y->qs[j] |= xi1 << 4;
    }
}

static __device__ void quantize_f32_q4_1_block(const float * __restrict__ x, block_q4_1 * __restrict__ y) {
    float vmin = FLT_MAX;
    float vmax = -FLT_MAX;

    for (int j = 0; j < QK4_1; ++j) {
        const float v = x[j];
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
    }

    const float d  = (vmax - vmin) / ((1 << 4) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    y->dm.x = d;
    y->dm.y = vmin;

    for (int j = 0; j < QK4_1/2; ++j) {
        const float x0 = (x[0       + j] - vmin)*id;
        const float x1 = (x[QK4_1/2 + j] - vmin)*id;

        const uint8_t xi0 = min(15, (int8_t)(x0 + 0.5f));
        const uint8_t xi1 = min(15, (int8_t)(x1 + 0.5f));

        y->qs[j]  = xi0;
        y->qs[j] |= xi1 << 4;
    }
}

static __device__ void quantize_f32_q5_0_block(const float * __restrict__ x, block_q5_0 * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK5_0; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    const float d  = vmax / -16;
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_0/2; ++j) {
        const float x0 = x[0       + j]*id;
        const float x1 = x[QK5_0/2 + j]*id;

        const uint8_t xi0 = min(31, (int8_t)(x0 + 16.5f));
        const uint8_t xi1 = min(31, (int8_t)(x1 + 16.5f));

        y->qs[j]  = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0/2);
    }
    memcpy(y->qh, &qh, sizeof(qh));
}

static __device__ void quantize_f32_q5_1_block(const float * __restrict__ x, block_q5_1 * __restrict__ y) {
    float min = x[0];
    float max = x[0];

    for (int j = 1; j < QK5_1; ++j) {
        const float v = x[j];
        min = v < min ? v : min;
        max = v > max ? v : max;
    }

    const float d  = (max - min) / 31;
    const float id = d ? 1.0f/d : 0.0f;

    y->dm.x = d;
    y->dm.y = min;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_1/2; ++j) {
        const float x0 = (x[0       + j] - min)*id;
        const float x1 = (x[QK5_1/2 + j] - min)*id;

        const uint8_t xi0 = (uint8_t)(x0 + 0.5f);
        const uint8_t xi1 = (uint8_t)(x1 + 0.5f);

        y->qs[j]  = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1/2);
    }
    memcpy(y->qh, &qh, sizeof(qh));
}

static __device__ void quantize_f32_q8_0_block(const float * __restrict__ x, block_q8_0 * __restrict__ y) {
    float amax = 0.0f; // absolute max

    for (int j = 0; j < QK8_0; j++) {
        const float v = x[j];
        amax = fmaxf(amax, fabsf(v));
    }

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    for (int j = 0; j < QK8_0; ++j) {
        const float x0 = x[j]*id;
        y->qs[j] = roundf(x0);
    }
}

static __device__ void quantize_f32_iq4_nl_block(const float * __restrict__ x, block_iq4_nl * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK4_NL; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    float d = vmax / kvalues_iq4nl[0];
    const float id = d ? 1.0f/d : 0.0f;

    float sumqx = 0, sumq2 = 0;
    for (int j = 0; j < QK4_NL/2; ++j) {
        const float x0 = x[0        + j]*id;
        const float x1 = x[QK4_NL/2 + j]*id;
        const uint8_t xi0 = best_index_int8(16, kvalues_iq4nl, x0);
        const uint8_t xi1 = best_index_int8(16, kvalues_iq4nl, x1);
        y->qs[j] = xi0 | (xi1 << 4);
        const float v0 = kvalues_iq4nl[xi0];
        const float v1 = kvalues_iq4nl[xi1];
        const float w0 = x[0        + j]*x[0        + j];
        const float w1 = x[QK4_NL/2 + j]*x[QK4_NL/2 + j];
        sumqx += w0*v0*x[j] + w1*v1*x[QK4_NL/2 + j];
        sumq2 += w0*v0*v0 + w1*v1*v1;
    }

    y->d = sumq2 > 0 ? sumqx/sumq2 : d;
}

// Pick the 4-bit E2M1 code (index into kvalues_mxfp4) closest to x for a sub-block
// scaled by e. Mirrors the CPU best_index_mxfp4() used by quantize_row_nvfp4_ref():
// same linear scan, same strict-less-than tie-break (so ties pick the lower code).
// The return type is uint8_t here vs int on the CPU, but the result is always 0..15.
static __device__ __forceinline__ uint8_t best_index_mxfp4(float x, float e) {
    uint8_t best_index = 0;
    float   best_err   = fabsf(kvalues_mxfp4[0]*e - x);
    for (int i = 1; i < 16; ++i) {
        const float err = fabsf(kvalues_mxfp4[i]*e - x);
        if (err < best_err) {
            best_err   = err;
            best_index = i;
        }
    }
    return best_index;
}

static __device__ void quantize_f32_nvfp4_block(const float * __restrict__ x, block_nvfp4 * __restrict__ y) {
    constexpr int n_sub = QK_NVFP4 / QK_NVFP4_SUB; // 4 sub-blocks of 16 values

    for (int s = 0; s < n_sub; ++s) {
        const float * xb = x + s*QK_NVFP4_SUB;

        // Use the same scalar comparison as the CPU reference rather than fmaxf(): under
        // CUDA's default flush-to-zero mode fmaxf() would flush a denormal amax to 0,
        // diverging from the CPU and silently zeroing a (tiny but non-zero) sub-block.
        float amax = 0.0f;
        for (int j = 0; j < QK_NVFP4_SUB; ++j) {
            if (amax < fabsf(xb[j])) {
                amax = fabsf(xb[j]);
            }
        }

        if (amax == 0.0f) {
            y->d[s] = 0;
            for (int j = 0; j < QK_NVFP4_SUB/2; ++j) {
                y->qs[s*(QK_NVFP4_SUB/2) + j] = 0;
            }
            continue;
        }

        // Mirror of quantize_row_nvfp4_ref() in ggml-quants.c -- must stay bit-identical.
        // amax/6 maps the largest E2M1 magnitude (6.0) to amax, but the UE4M3 sub-block
        // scale has only a 3-bit mantissa, so the code nearest to amax/6 is rarely the
        // one that minimises reconstruction error. Scan every valid UE4M3 scale code and
        // keep the lowest-error one. Codes in [1, 0x7E] all decode to a finite non-zero
        // scale (only code 0 and the NaN sentinel 0x7F decode to 0), so no zero-scale
        // guard is needed inside the loop. The scan is ascending with a strict-less-than
        // test, so on equal error the lower code wins.
        uint8_t best_ue  = 1;
        float   best_err = INFINITY;
        for (int uec = 1; uec <= 0x7E; ++uec) {
            const float dc = ggml_cuda_ue4m3_to_fp32((uint8_t) uec);
            float err = 0.0f;
            for (int j = 0; j < QK_NVFP4_SUB; ++j) {
                const float r = kvalues_mxfp4[best_index_mxfp4(xb[j], dc)]*dc - xb[j];
                err += r*r;
            }
            if (err < best_err) {
                best_err = err;
                best_ue  = (uint8_t) uec;
            }
        }

        y->d[s] = best_ue;
        const float d = ggml_cuda_ue4m3_to_fp32(best_ue);

        for (int j = 0; j < QK_NVFP4_SUB/2; ++j) {
            const uint8_t x0 = best_index_mxfp4(xb[0              + j], d);
            const uint8_t x1 = best_index_mxfp4(xb[QK_NVFP4_SUB/2 + j], d);

            y->qs[s*(QK_NVFP4_SUB/2) + j] = x0 | (x1 << 4);
        }
    }
}

// Pack a (squared-error, UE4M3 code) pair into one uint64 ordered for argmin.
// Squared error is a non-negative IEEE float, so its raw bits sort as a uint32 in
// the same order as the float value; placing it in the high 32 bits and the code
// in the low 8 bits makes a plain uint64 compare order by error first, and on an
// exact error tie pick the lower code -- matching the CPU scan's tie-break.
static __device__ __forceinline__ uint64_t pack_err_uec(float err, uint8_t uec) {
    return ((uint64_t) __float_as_uint(err) << 32) | (uint64_t) uec;
}

// Warp-wide argmin over (err, uec): every lane contributes its local best, the
// result (the lowest-error code, lower code on ties) is returned to every lane.
static __device__ __forceinline__ uint8_t warp_argmin_err_uec(float err, uint8_t uec) {
    uint64_t val = pack_err_uec(err, uec);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const uint64_t o = __shfl_xor_sync(0xFFFFFFFF, val, offset, 32);
        if (o < val) {
            val = o;
        }
    }
    return (uint8_t) (val & 0xFF);
}

// Warp-cooperative NVFP4 quantizer: one 32-lane warp per block_nvfp4. The 25x
// heavier full-range UE4M3 scan is split across the 32 lanes (4 codes each), so
// the per-block cost stays close to the old narrow-window scalar version. The
// per-code SSE sum keeps the exact CPU op order (in-order j=0..15) so the result
// is bit-identical to quantize_f32_nvfp4_block() / the CPU reference.
static __device__ void quantize_f32_nvfp4_block_warp(const float * __restrict__ x,
                                                     block_nvfp4 * __restrict__ y) {
    constexpr int n_sub = QK_NVFP4 / QK_NVFP4_SUB; // 4 sub-blocks of 16 values

    const int lane = threadIdx.x; // 0..31, one warp per block

    for (int s = 0; s < n_sub; ++s) {
        const float * xb = x + s*QK_NVFP4_SUB;

        // Lane 0 computes amax with the same scalar comparison as the CPU reference
        // (not fmaxf(): FTZ would flush a denormal amax to 0), then broadcasts it.
        float amax = 0.0f;
        if (lane == 0) {
            for (int j = 0; j < QK_NVFP4_SUB; ++j) {
                if (amax < fabsf(xb[j])) {
                    amax = fabsf(xb[j]);
                }
            }
        }
        amax = __shfl_sync(0xFFFFFFFF, amax, 0, 32);

        if (amax == 0.0f) {
            if (lane == 0) {
                y->d[s] = 0;
                for (int j = 0; j < QK_NVFP4_SUB/2; ++j) {
                    y->qs[s*(QK_NVFP4_SUB/2) + j] = 0;
                }
            }
            continue;
        }

        // Each lane evaluates codes uec = 1 + lane + k*32 for k = 0..3, covering the
        // full [1, 0x7E] range across the warp. Each code uses the same in-order SSE
        // sum as the CPU so the bits match.
        float   local_best_err = INFINITY;
        uint8_t local_best_ue  = 1;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int uec = 1 + lane + k*32;
            if (uec > 0x7E) {
                continue;
            }
            const float dc = ggml_cuda_ue4m3_to_fp32((uint8_t) uec);
            float err = 0.0f;
            for (int j = 0; j < QK_NVFP4_SUB; ++j) {
                const float r = kvalues_mxfp4[best_index_mxfp4(xb[j], dc)]*dc - xb[j];
                err += r*r;
            }
            if (err < local_best_err) {
                local_best_err = err;
                local_best_ue  = (uint8_t) uec;
            }
        }

        const uint8_t best_ue = warp_argmin_err_uec(local_best_err, local_best_ue);

        if (lane == 0) {
            y->d[s] = best_ue;
            const float d = ggml_cuda_ue4m3_to_fp32(best_ue);
            for (int j = 0; j < QK_NVFP4_SUB/2; ++j) {
                const uint8_t x0 = best_index_mxfp4(xb[0              + j], d);
                const uint8_t x1 = best_index_mxfp4(xb[QK_NVFP4_SUB/2 + j], d);
                y->qs[s*(QK_NVFP4_SUB/2) + j] = x0 | (x1 << 4);
            }
        }
    }
}

// Wrapper functions for cpy.cu compatibility
static __device__ void cpy_blck_f32_q4_0(const char * cxi, char * cdsti) {
    quantize_f32_q4_0_block((const float *)cxi, (block_q4_0 *)cdsti);
}

static __device__ void cpy_blck_f32_q4_1(const char * cxi, char * cdsti) {
    quantize_f32_q4_1_block((const float *)cxi, (block_q4_1 *)cdsti);
}

static __device__ void cpy_blck_f32_q5_0(const char * cxi, char * cdsti) {
    quantize_f32_q5_0_block((const float *)cxi, (block_q5_0 *)cdsti);
}

static __device__ void cpy_blck_f32_q5_1(const char * cxi, char * cdsti) {
    quantize_f32_q5_1_block((const float *)cxi, (block_q5_1 *)cdsti);
}

static __device__ void cpy_blck_f32_q8_0(const char * cxi, char * cdsti) {
    quantize_f32_q8_0_block((const float *)cxi, (block_q8_0 *)cdsti);
}

static __device__ void cpy_blck_f32_iq4_nl(const char * cxi, char * cdsti) {
    quantize_f32_iq4_nl_block((const float *)cxi, (block_iq4_nl *)cdsti);
}

static __device__ void cpy_blck_f32_nvfp4(const char * cxi, char * cdsti) {
    quantize_f32_nvfp4_block((const float *)cxi, (block_nvfp4 *)cdsti);
}

static __device__ void quantize_f32_f8_e4m3_block(const float * __restrict__ x, block_f8_e4m3 * __restrict__ y) {
    // Scalar comparison rather than fmaxf(): under CUDA's flush-to-zero mode fmaxf() would
    // flush a denormal amax to 0, diverging from the CPU reference quantize_row_f8_e4m3_ref().
    float amax = 0.0f;
    for (int j = 0; j < QK_F8_E4M3; ++j) {
        const float ax = fabsf(x[j]);
        if (amax < ax) {
            amax = ax;
        }
    }

    const float d  = amax / 448.0f;
    const float id = d != 0.0f ? 1.0f/d : 0.0f;

    y->d = d;

    for (int j = 0; j < QK_F8_E4M3; ++j) {
        y->qs[j] = ggml_cuda_fp32_to_se4m3(x[j]*id);
    }
}

static __device__ void cpy_blck_f32_f8_e4m3(const char * cxi, char * cdsti) {
    quantize_f32_f8_e4m3_block((const float *)cxi, (block_f8_e4m3 *)cdsti);
}

template<typename src_t, typename dst_t>
static __device__ void cpy_1_scalar(const char * cxi, char * cdsti) {
    *(dst_t *) cdsti = ggml_cuda_cast<dst_t>(*(const src_t *) cxi);
}
