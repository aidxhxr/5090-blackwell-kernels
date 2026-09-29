// Private helpers of the w4gemm kernels (w4gemm.cu) and the repack (w4gemm_quant.cu): the
// packed weight layout, the k order inside a group, and the register dequant. Not part of
// the public API.
//
// The kernels compute C^T = W^T A^T on the tensor cores: 16 output columns of W are the rows
// of the mma.sync.m16n8k16 A operand and 8 tokens of the activations are its n8 B operand.
// With g = lane / 4 and c = lane % 4 (PTX ISA fragment layouts), the lane's A fragment of a
// 16 x 16 block is 8 weights, one 32-bit word of nibbles:
//     a0 = (row g,     kappa 2c, 2c+1)    a1 = (row g + 8, kappa 2c, 2c+1)
//     a2 = (row g,     kappa 2c+8, 2c+9)  a3 = (row g + 8, kappa 2c+8, 2c+9)
// and its B fragment is 4 activations of token g: b0 = kappa (2c, 2c+1), b1 = (2c+8, 2c+9).
//
// kappa is the k index as the mma sees it, not the k of the matrix. A dot product does not
// care in which order it visits k, so inside every 128-k group the kernels feed k16 step j
// (j = 0..7) the matrix k
//     k(j, kappa) = 32 (j / 2) + 8 c + 4 (j % 2) + 2 (kappa / 8) + kappa % 2,  c = (kappa % 8) / 2
// on both operands. Lane (g, c)'s activations for steps 2i and 2i + 1 are then the eight
// consecutive k 32 i + 8 c .. 32 i + 8 c + 7 of token g: one 16-byte load gives b0 and b1 of
// both steps, and the four lanes of a row read 64 contiguous bytes. That is what lets the
// small-M kernel load activations straight into registers, and the large-M kernel read them
// from shared memory with LDS.128 instead of ldmatrix. Scales stay per 128 matrix k.
//
// The repacked layout stores, for every 16 columns x 64 k (steps 4u .. 4u + 3 of a group),
// 32 lanes x 4 words (one per step), so a lane's 16 bytes are the four A fragments of those
// 64 k and a warp's 512 bytes are one contiguous block. Blocks run along K within a
// 16-column strip: int32 [N/16][K/64][32][4].
//
// Within a word, nibble p (bits 4p..4p+3) holds
//     p = 0: (g, kappa 2c)      p = 4: (g, 2c+1)
//     p = 1: (g+8, 2c)          p = 5: (g+8, 2c+1)
//     p = 2: (g, 2c+8)          p = 6: (g, 2c+9)
//     p = 3: (g+8, 2c+8)        p = 7: (g+8, 2c+9)
// because the dequant extracts nibbles p and p + 4 together, into the low and high halves of
// one bf16x2 register: (word >> 4p) & 0x000F000F puts them in the low 4 mantissa bits of two
// bf16 lanes, and OR-ing in the exponent of 128 (0x4300) makes each lane exactly 128 + q.
#pragma once

#include <cuda_bf16.h>

#include <cstdint>

namespace spark::w4 {

// Index of the 512-byte block of 16-column strip t and 64-k unit u (strip-major).
__host__ __device__ __forceinline__ size_t block_index(int t, int u, int units) {
    return static_cast<size_t>(t) * units + u;
}

// Matrix k within a 128-k group of mma step j (0..7) and mma index kappa (0..15).
__host__ __device__ __forceinline__ int group_k(int j, int kappa) {
    const int c = (kappa % 8) / 2;
    return 32 * (j / 2) + 8 * c + 4 * (j % 2) + 2 * (kappa / 8) + kappa % 2;
}

// (row within the 16-column block, matrix k within the 64-k block) of nibble p of word j of
// `lane` in 64-k unit u (u % 2 is the half of the 128-k group it covers).
__host__ __device__ __forceinline__ void nibble_coords(int lane, int u, int j, int p, int& dn,
                                                       int& dk) {
    const int g = lane >> 2, c = lane & 3;
    const int hi_row = p & 1, hi_k = (p >> 1) & 1, odd = p >> 2;
    dn = g + 8 * hi_row;
    dk = group_k(4 * (u % 2) + j, 8 * hi_k + 2 * c + odd) - 64 * (u % 2);
}

// Inverse: where matrix (n, k) lives: word index in the packed array and nibble position.
__host__ __device__ __forceinline__ void locate(int n, int k, int K, size_t& word, int& p) {
    const int a = k % 128;
    const int i = a / 32, c = (a % 32) / 8, jodd = (a % 8) / 4, hi_k = (a % 4) / 2, odd = a % 2;
    const int j = 2 * i + jodd;  // step within the group
    const int u = (k / 128) * 2 + j / 4;
    const int r = n % 16;
    const int lane = (r % 8) * 4 + c;
    p = r / 8 + 2 * hi_k + 4 * odd;
    word = (block_index(n / 16, u, K / 64) * 32 + lane) * 4 + j % 4;
}

// (a & b) | c in one LOP3.
__device__ __forceinline__ unsigned lop3_and_or(unsigned a, unsigned b, unsigned c) {
    unsigned d;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;\n" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

__device__ __forceinline__ unsigned bf16x2_sub(unsigned a, unsigned b) {
    unsigned d;
    asm("sub.rn.bf16x2 %0, %1, %2;\n" : "=r"(d) : "r"(a), "r"(b));
    return d;
}

__device__ __forceinline__ unsigned bf16x2_mul(unsigned a, unsigned b) {
    unsigned d;
    asm("mul.rn.bf16x2 %0, %1, %2;\n" : "=r"(d) : "r"(a), "r"(b));
    return d;
}

// bf16 bits of 128 + z in both halves (z in 0..15): the offset the dequant subtracts.
__device__ __forceinline__ unsigned zero_pair(unsigned z) {
    return 0x43004300u | z | (z << 16);
}
constexpr unsigned kSymZeroPair = 0x43084308u;  // 136.0 = 128 + 8 in both halves

// One packed word -> the lane's A fragment of one 16 x 16 block: (q - z) * s per weight,
// q - z exact (a difference of two integers in [128, 143]) and the product rounded once to
// bf16, which is what bf16((q - z) * s) computed in fp32 gives. zz_*, ss_* are bf16x2 pairs
// (the same value twice) for rows g (lo) and g + 8 (hi). 4 LOP3 + 3 shifts + 4 SUB + 4 MUL.
__device__ __forceinline__ void dequant(unsigned q, unsigned zz_lo, unsigned zz_hi, unsigned ss_lo,
                                        unsigned ss_hi, unsigned (&a)[4]) {
    constexpr unsigned kMask = 0x000F000Fu, kExp = 0x43004300u;
    const unsigned t0 = lop3_and_or(q, kMask, kExp);
    const unsigned t1 = lop3_and_or(q >> 4, kMask, kExp);
    const unsigned t2 = lop3_and_or(q >> 8, kMask, kExp);
    const unsigned t3 = lop3_and_or(q >> 12, kMask, kExp);
    a[0] = bf16x2_mul(bf16x2_sub(t0, zz_lo), ss_lo);
    a[1] = bf16x2_mul(bf16x2_sub(t1, zz_hi), ss_hi);
    a[2] = bf16x2_mul(bf16x2_sub(t2, zz_lo), ss_lo);
    a[3] = bf16x2_mul(bf16x2_sub(t3, zz_hi), ss_hi);
}

// bf16 bits in both halves of a 32-bit register.
__device__ __forceinline__ unsigned splat(unsigned short h) {
    return static_cast<unsigned>(h) | (static_cast<unsigned>(h) << 16);
}

}  // namespace spark::w4
