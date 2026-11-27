// 16-bit storage helpers (BF16 / FP16) & mma.sync.m16n8k16 Tensor Core primitives for [N, 16, 64] tensors (LST_ACT_HALF=1|2).
// Mode 1: __nv_bfloat16 (identical FP32 exponent range -> zero overflow/underflow risk).
// Mode 2: __half (10-bit mantissa -> 8x finer precision where values fit in [6e-8, 65504]).
// All accumulations, weights, [N, 64] features, forces, loss, gradients, and optimizer stay 100% FP32.
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <type_traits>

namespace lst {
namespace ah {

template <typename H>
__device__ __forceinline__ float to_f(H x) {
    if constexpr (std::is_same<H, float>::value) return x;
    else if constexpr (std::is_same<H, __nv_bfloat16>::value) return __bfloat162float(x);
    else if constexpr (std::is_same<H, __half>::value) return __half2float(x);
    else return static_cast<float>(x);
}

template <typename T, typename H>
__device__ __forceinline__ T ld_a16(const H* __restrict__ p, long idx) {
    if constexpr (std::is_same<H, T>::value) return p[idx];
    else return static_cast<T>(to_f(p[idx]));
}

template <typename H, typename T>
__device__ __forceinline__ void st_a16(H* __restrict__ p, long idx, T v) {
    if constexpr (std::is_same<H, T>::value) p[idx] = v;
    else if constexpr (std::is_same<H, __nv_bfloat16>::value) p[idx] = __float2bfloat16_rn(static_cast<float>(v));
    else if constexpr (std::is_same<H, __half>::value) p[idx] = __float2half_rn(static_cast<float>(v));
    else p[idx] = static_cast<H>(v);
}

template <typename H>
__device__ __forceinline__ H from_f(float v) {
    if constexpr (std::is_same<H, __nv_bfloat16>::value) return __float2bfloat16_rn(v);
    else if constexpr (std::is_same<H, __half>::value) return __float2half_rn(v);
    else return static_cast<H>(v);
}

template <typename H>
__device__ __forceinline__ uint32_t pack2(float lo, float hi) {
    if constexpr (std::is_same<H, __nv_bfloat16>::value) {
        const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
        return *reinterpret_cast<const uint32_t*>(&v);
    } else {
        const __half2 v = __floats2half2_rn(lo, hi);
        return *reinterpret_cast<const uint32_t*>(&v);
    }
}

template <typename H>
__device__ __forceinline__ float2 unpack2(uint32_t u) {
    if constexpr (std::is_same<H, __nv_bfloat16>::value) {
        return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u));
    } else {
        return __half22float2(*reinterpret_cast<const __half2*>(&u));
    }
}

// Wave 26: Per-(node, m) 8-bit quantization helpers for backward-only GA_buf (k_slot_xd_opt_v3).
// FP8 E4M3 uses 1-cycle native hardware cvt on sm_89/sm_120; INT8 uses symmetric [-127, 127].
__device__ __forceinline__ uint16_t pack_ga_fp8(float g_scaled, float a_scaled) {
    return static_cast<uint16_t>(
        __nv_cvt_float2_to_fp8x2(make_float2(g_scaled, a_scaled), __NV_SATFINITE, __NV_E4M3));
}
__device__ __forceinline__ __half2 unpack_ga_fp8_h2(uint16_t u) {
    return static_cast<__half2>(
        __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(u), __NV_E4M3));
}
__device__ __forceinline__ float2 unpack_ga_fp8(uint16_t u) {
    return __half22float2(unpack_ga_fp8_h2(u));
}
__device__ __forceinline__ uint16_t pack_ga_i8(float g_scaled, float a_scaled) {
    const int gi = max(-127, min(127, __float2int_rn(g_scaled)));
    const int ai = max(-127, min(127, __float2int_rn(a_scaled)));
    return static_cast<uint16_t>(static_cast<uint8_t>(static_cast<int8_t>(gi))) |
           (static_cast<uint16_t>(static_cast<uint8_t>(static_cast<int8_t>(ai))) << 8);
}
__device__ __forceinline__ float2 unpack_ga_i8(uint16_t u) {
    const int8_t gi = static_cast<int8_t>(u & 0xffu);
    const int8_t ai = static_cast<int8_t>(u >> 8);
    return make_float2(static_cast<float>(gi), static_cast<float>(ai));
}

template <typename H>
__device__ __forceinline__ uint32_t pack2_raw(H lo, H hi) {
    static_assert(sizeof(H) == 2, "pack2_raw expects 16-bit half/bfloat16");
    if constexpr (std::is_same<H, __nv_bfloat16>::value) {
        const __nv_bfloat162 v = __halves2bfloat162(lo, hi);
        return *reinterpret_cast<const uint32_t*>(&v);
    } else {
        const __half2 v = __halves2half2(lo, hi);
        return *reinterpret_cast<const uint32_t*>(&v);
    }
}

template <typename H>
__device__ __forceinline__ uint32_t ld_h2(const H* __restrict__ p, long pair_idx) {
    return reinterpret_cast<const uint32_t*>(p)[pair_idx];
}

template <typename H>
__device__ __forceinline__ void st_h2(H* __restrict__ p, long pair_idx, uint32_t v) {
    reinterpret_cast<uint32_t*>(p)[pair_idx] = v;
}

template <typename H>
__device__ __forceinline__ void mma_16x8x16(float c[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                            uint32_t b0, uint32_t b1) {
    if constexpr (std::is_same<H, __nv_bfloat16>::value) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                     "{%0,%1,%2,%3};"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                     : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    } else {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                     "{%0,%1,%2,%3};"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                     : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
}

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
    const uint32_t addr = smem_u32(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
                                                  const void* p) {
    const uint32_t addr = smem_u32(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

__device__ __forceinline__ uint32_t cvt_tf32(float x) {
    uint32_t r;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
    return r;
}

__device__ __forceinline__ void mma_tf32_16x8x8(float c[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

}  // namespace ah
}  // namespace lst
