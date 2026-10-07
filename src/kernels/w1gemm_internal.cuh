// Private helpers of the w1gemm kernels (w1gemm.cu): the packed bit layout, the k order
// inside a group, and the bit -> bf16x2 dequant. Not part of the public API.
//
// The packed layout is the plain one of include/spark/kernels.h: uint32 [N][K * bits / 32],
// row n's k ascending from the lowest bits of its lowest word. bits == 1 stores the sign
// (code 1 = +s, 0 = -s) in bit k % 32 of word k / 32; bits == 2 stores the ternary code
// (0 = -s, 1 = 0, 2 = +s) in bits 2 (k % 16), 2 (k % 16) + 1 of word k / 16.
//
// The kernels compute C^T = W^T A^T on the tensor cores the way w4gemm does: 16 output
// columns of W are the rows of the mma.sync.m16n8k16 A operand and 8 tokens are its n8 B
// operand. With g = lane / 4 and c = lane % 4, the lane's A fragment of a 16 x 16 block is
//     a0 = (row g,     kappa 2c, 2c+1)    a1 = (row g + 8, kappa 2c, 2c+1)
//     a2 = (row g,     kappa 2c+8, 2c+9)  a3 = (row g + 8, kappa 2c+8, 2c+9)
// and its B fragment is b0 = kappa (2c, 2c+1), b1 = (2c+8, 2c+9) of token g.
//
// kappa is the k index as the mma sees it. Inside every 128-k group the kernels feed k16 step
// j (0..7) the matrix k
//     k(j, kappa) = 32 (j / 2) + 8 c + 4 (j % 2) + 2 (kappa / 8) + kappa % 2,  c = (kappa % 8) / 2
// on both operands (w4gemm's order), so lane (g, c) needs for steps 2i and 2i + 1 the eight
// consecutive k 32 i + 8 c .. 32 i + 8 c + 7 of its two rows: for bits == 1 that is byte c of
// word 4 grp + i of the row, for bits == 2 the 16-bit half c % 2 of word 8 grp + 2 i + c / 2.
// One 16-byte load per row covers a whole group at 1 bit (two at 2 bits), and the
// activations are the same one 16-byte load per lane and step pair as in w4gemm.
//
// Within the lane's 8 codes t = 0..7 (k 32 i + 8 c + t): t = 0, 1 are a0 / b0 of step 2i,
// t = 2, 3 are a2 / b1 of step 2i, t = 4..7 the same for step 2i + 1.
#pragma once

#include <cuda_bf16.h>

#include <cstdint>

namespace spark::w1 {

// Words per row of the packed matrix.
__host__ __device__ __forceinline__ int words_per_row(int K, int bits) {
    return K * bits / 32;
}

// Matrix k within a 128-k group of mma step j (0..7) and mma index kappa (0..15).
__host__ __device__ __forceinline__ int group_k(int j, int kappa) {
    const int c = (kappa % 8) / 2;
    return 32 * (j / 2) + 8 * c + 4 * (j % 2) + 2 * (kappa / 8) + kappa % 2;
}

// The code of weight (n, k) from the packed array.
__host__ __device__ __forceinline__ int code_at(const uint32_t* packed, int n, int k, int K,
                                                int bits) {
    const int wpr = words_per_row(K, bits);
    if (bits == 1) return (packed[static_cast<size_t>(n) * wpr + k / 32] >> (k % 32)) & 1;
    return (packed[static_cast<size_t>(n) * wpr + k / 16] >> (2 * (k % 16))) & 3;
}

// bf16 bits in both halves of a 32-bit register.
__device__ __forceinline__ unsigned splat(unsigned short h) {
    return static_cast<unsigned>(h) | (static_cast<unsigned>(h) << 16);
}

// Sign bits of a bf16x2 register.
constexpr unsigned kSignPair = 0x80008000u;

// bits == 1: codes (b0, b1) in bits 0 and 1 of q -> bf16x2 (+-s, +-s), s >= 0 in both halves
// of ss: code 1 keeps s, code 0 flips the sign. One LOP3-able mask and one XOR: the negated
// bits, placed on the two sign positions.
__device__ __forceinline__ unsigned sign_pair(unsigned q, unsigned ss) {
    const unsigned nq = ~q;
    const unsigned m = ((nq & 1u) << 15) | ((nq & 2u) << 30);
    return ss ^ m;
}

// bits == 2: codes (q0, q1) in bits 0..1 and 2..3 of q -> bf16x2 (s (q0 - 1), s (q1 - 1)).
// Bit 0 of a code says "zero" (code 1), bit 1 says "positive" (code 2): the sign flip of
// sign_pair on the second bit, then the halves of the zero codes cleared.
__device__ __forceinline__ unsigned ternary_pair(unsigned q, unsigned ss) {
    const unsigned nq = ~q;
    const unsigned m = ((nq & 2u) << 14) | ((nq & 8u) << 28);
    const unsigned zero = ((q & 1u) * 0xFFFFu) | (((q >> 2) & 1u) * 0xFFFF0000u);
    return (ss ^ m) & ~zero;
}

// The lane's A fragment of one k16 step from its 8 codes of the step pair: `q` holds the
// 4 BITS-bit codes t = 0..3 of the step in its low 4 BITS bits (t = 0, 1 -> a0 / a1, t = 2, 3
// -> a2 / a3) for row g (q_lo) and row g + 8 (q_hi), ss_* the bf16x2 (s, s) of each row.
template <int BITS>
__device__ __forceinline__ void dequant(unsigned q_lo, unsigned q_hi, unsigned ss_lo,
                                        unsigned ss_hi, unsigned (&a)[4]) {
    if constexpr (BITS == 1) {
        a[0] = sign_pair(q_lo, ss_lo);
        a[1] = sign_pair(q_hi, ss_hi);
        a[2] = sign_pair(q_lo >> 2, ss_lo);
        a[3] = sign_pair(q_hi >> 2, ss_hi);
    } else {
        a[0] = ternary_pair(q_lo, ss_lo);
        a[1] = ternary_pair(q_hi, ss_hi);
        a[2] = ternary_pair(q_lo >> 4, ss_lo);
        a[3] = ternary_pair(q_hi >> 4, ss_hi);
    }
}

}  // namespace spark::w1
