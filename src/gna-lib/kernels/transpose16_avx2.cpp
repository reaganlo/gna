/*
 INTEL CONFIDENTIAL
 Copyright 2017 Intel Corporation.

 The source code contained or described herein and all documents related
 to the source code ("Material") are owned by Intel Corporation or its suppliers
 or licensors. Title to the Material remains with Intel Corporation or its suppliers
 and licensors. The Material may contain trade secrets and proprietary
 and confidential information of Intel Corporation and its suppliers and licensors,
 and is protected by worldwide copyright and trade secret laws and treaty provisions.
 No part of the Material may be used, copied, reproduced, modified, published,
 uploaded, posted, transmitted, distributed, or disclosed in any way without Intel's
 prior express written permission.

 No license under any patent, copyright, trade secret or other intellectual
 property right is granted to or conferred upon you by disclosure or delivery
 of the Materials, either expressly, by implication, inducement, estoppel
 or otherwise. Any license under such intellectual property rights must
 be express and approved by Intel in writing.

 Unless otherwise agreed by Intel in writing, you may not remove or alter this notice
 or any other notice embedded in Materials by Intel or Intel's suppliers or licensors
 in any way.
*/

#include <string.h>

#include "igemv.h"
#include "igemv16.h"

void TransposeKernelImpl(TransposeConfig const * const cfg)
{
    uint32_t M = cfg->rowCount;
    uint32_t N = cfg->columnCount;
    const int16_t * const I = cfg->input;
    int16_t * const O = cfg->output;

    uint32_t i, j;

    // input matrix is a vector - copy 
    if (M == 1 || N == 1)
    {
        memcpy_s(O, M * N * sizeof(int16_t), I, M * N * sizeof(int16_t));
        return;
    }

    // very small matrix - generic tranpose
    // or N, M between 8 and 16 for large matrices - not implemented
    if (M * N < VEC_16CAP * VEC_16CAP
        || (N > 8 && N < 16) || (M > 8 && M < 16))
    {
        for (i = 0; i < M; i++)
        {
            for (j = 0; j < N; j++)
            {
                O[j * M + i] = I[i * N + j];
            }
        }

        return;
    }

    // M, N end element that will be loaded to AVX vector
    uint32_t N_VEC = N - N % VEC_16CAP;
    uint32_t M_VEC = M - M % VEC_16CAP;

    // first input vector pointer and input end pointer
    int16_t *in0 = const_cast<int16_t*>(I);
    int16_t *in_end = in0 + M * N;

    // for large matrices - some purpose in the future?
    if (N >= VEC_16CAP && M >= VEC_16CAP)
    {
        __m256i a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p;
        __m256i ab_lo, cd_lo, ef_lo, gh_lo, ij_lo, kl_lo, mn_lo, op_lo;
        __m256i ab_hi, cd_hi, ef_hi, gh_hi, ij_hi, kl_hi, mn_hi, op_hi;
        __m256i abcd1, abcd5, abcd9, abcd13,
            efgh1, efgh5, efgh9, efgh13,
            ijkl1, ijkl5, ijkl9, ijkl13,
            mnop1, mnop5, mnop9, mnop13,
            ah1, ah3, ah5, ah7, ah9, ah11, ah13, ah15,
            ip1, ip3, ip5, ip7, ip9, ip11, ip13, ip15,
            ap1, ap2, ap3, ap4, ap5, ap6, ap7, ap8, ap9,
            ap10, ap11, ap12, ap13, ap14, ap15, ap16;

        int16_t *in1, *in2, *in3, *in4, *in5, *in6, *in7, *in8,
            *in9, *in10, *in11, *in12, *in13, *in14, *in15;

        int16_t *out0, *out1, *out2, *out3, *out4, *out5, *out6, *out7, *out8,
            *out9, *out10, *out11, *out12, *out13, *out14, *out15;

        uint32_t ii, jj;
        out0 = O;
        in_end = in0 + M_VEC * N;
        for (; in0 < in_end;)
        {
            in1 = in0 + N,
                in2 = in1 + N,
                in3 = in2 + N,
                in4 = in3 + N,
                in5 = in4 + N,
                in6 = in5 + N,
                in7 = in6 + N,
                in8 = in7 + N,
                in9 = in8 + N,
                in10 = in9 + N,
                in11 = in10 + N,
                in12 = in11 + N,
                in13 = in12 + N,
                in14 = in13 + N,
                in15 = in14 + N;

            a = _mm256_lddqu_si256((__m256i*)in0);
            b = _mm256_lddqu_si256((__m256i*)in1);
            c = _mm256_lddqu_si256((__m256i*)in2);
            d = _mm256_lddqu_si256((__m256i*)in3);
            e = _mm256_lddqu_si256((__m256i*)in4);
            f = _mm256_lddqu_si256((__m256i*)in5);
            g = _mm256_lddqu_si256((__m256i*)in6);
            h = _mm256_lddqu_si256((__m256i*)in7);
            i = _mm256_lddqu_si256((__m256i*)in8);
            j = _mm256_lddqu_si256((__m256i*)in9);
            k = _mm256_lddqu_si256((__m256i*)in10);
            l = _mm256_lddqu_si256((__m256i*)in11);
            m = _mm256_lddqu_si256((__m256i*)in12);
            n = _mm256_lddqu_si256((__m256i*)in13);
            o = _mm256_lddqu_si256((__m256i*)in14);
            p = _mm256_lddqu_si256((__m256i*)in15);

            out1 = out0 + M,
                out2 = out1 + M,
                out3 = out2 + M,
                out4 = out3 + M,
                out5 = out4 + M,
                out6 = out5 + M,
                out7 = out6 + M,
                out8 = out7 + M,
                out9 = out8 + M,
                out10 = out9 + M,
                out11 = out10 + M,
                out12 = out11 + M,
                out13 = out12 + M,
                out14 = out13 + M,
                out15 = out14 + M;

            int16_t *col_end = in0 + N_VEC;

            for (; in0 < col_end;)
            {
                in0 += VEC_16CAP;
                in1 += VEC_16CAP;
                in2 += VEC_16CAP;
                in3 += VEC_16CAP;
                in4 += VEC_16CAP;
                in5 += VEC_16CAP;
                in6 += VEC_16CAP;
                in7 += VEC_16CAP;
                in8 += VEC_16CAP;
                in9 += VEC_16CAP;
                in10 += VEC_16CAP;
                in11 += VEC_16CAP;
                in12 += VEC_16CAP;
                in13 += VEC_16CAP;
                in14 += VEC_16CAP;
                in15 += VEC_16CAP;

                ab_lo = _mm256_unpacklo_epi16(a, b);
                cd_lo = _mm256_unpacklo_epi16(c, d);
                ef_lo = _mm256_unpacklo_epi16(e, f);
                gh_lo = _mm256_unpacklo_epi16(g, h);
                ij_lo = _mm256_unpacklo_epi16(i, j);
                kl_lo = _mm256_unpacklo_epi16(k, l);
                mn_lo = _mm256_unpacklo_epi16(m, n);
                op_lo = _mm256_unpacklo_epi16(o, p);

                ab_hi = _mm256_unpackhi_epi16(a, b);
                cd_hi = _mm256_unpackhi_epi16(c, d);
                ef_hi = _mm256_unpackhi_epi16(e, f);
                gh_hi = _mm256_unpackhi_epi16(g, h);
                ij_hi = _mm256_unpackhi_epi16(i, j);
                kl_hi = _mm256_unpackhi_epi16(k, l);
                mn_hi = _mm256_unpackhi_epi16(m, n);
                op_hi = _mm256_unpackhi_epi16(o, p);

                abcd1 = _mm256_unpacklo_epi32(ab_lo, cd_lo);
                abcd5 = _mm256_unpackhi_epi32(ab_lo, cd_lo);
                abcd9 = _mm256_unpacklo_epi32(ab_hi, cd_hi);
                abcd13 = _mm256_unpackhi_epi32(ab_hi, cd_hi);

                efgh1 = _mm256_unpacklo_epi32(ef_lo, gh_lo);
                efgh5 = _mm256_unpackhi_epi32(ef_lo, gh_lo);
                efgh9 = _mm256_unpacklo_epi32(ef_hi, gh_hi);
                efgh13 = _mm256_unpackhi_epi32(ef_hi, gh_hi);

                ijkl1 = _mm256_unpacklo_epi32(ij_lo, kl_lo);
                ijkl5 = _mm256_unpackhi_epi32(ij_lo, kl_lo);
                ijkl9 = _mm256_unpacklo_epi32(ij_hi, kl_hi);
                ijkl13 = _mm256_unpackhi_epi32(ij_hi, kl_hi);

                mnop1 = _mm256_unpacklo_epi32(mn_lo, op_lo);
                mnop5 = _mm256_unpackhi_epi32(mn_lo, op_lo);
                mnop9 = _mm256_unpacklo_epi32(mn_hi, op_hi);
                mnop13 = _mm256_unpackhi_epi32(mn_hi, op_hi);

                ah1 = _mm256_unpacklo_epi64(abcd1, efgh1);
                ah3 = _mm256_unpackhi_epi64(abcd1, efgh1);
                ah5 = _mm256_unpacklo_epi64(abcd5, efgh5);
                ah7 = _mm256_unpackhi_epi64(abcd5, efgh5);
                ah9 = _mm256_unpacklo_epi64(abcd9, efgh9);
                ah11 = _mm256_unpackhi_epi64(abcd9, efgh9);
                ah13 = _mm256_unpacklo_epi64(abcd13, efgh13);
                ah15 = _mm256_unpackhi_epi64(abcd13, efgh13);

                ip1 = _mm256_unpacklo_epi64(ijkl1, mnop1);
                ip3 = _mm256_unpackhi_epi64(ijkl1, mnop1);
                ip5 = _mm256_unpacklo_epi64(ijkl5, mnop5);
                ip7 = _mm256_unpackhi_epi64(ijkl5, mnop5);
                ip9 = _mm256_unpacklo_epi64(ijkl9, mnop9);
                ip11 = _mm256_unpackhi_epi64(ijkl9, mnop9);
                ip13 = _mm256_unpacklo_epi64(ijkl13, mnop13);
                ip15 = _mm256_unpackhi_epi64(ijkl13, mnop13);

                ap1 = _mm256_permute2x128_si256(ah1, ip1, 32);
                ap2 = _mm256_permute2x128_si256(ah1, ip1, 49);
                ap3 = _mm256_permute2x128_si256(ah3, ip3, 32);
                ap4 = _mm256_permute2x128_si256(ah3, ip3, 49);
                ap5 = _mm256_permute2x128_si256(ah5, ip5, 32);
                ap6 = _mm256_permute2x128_si256(ah5, ip5, 49);
                ap7 = _mm256_permute2x128_si256(ah7, ip7, 32);
                ap8 = _mm256_permute2x128_si256(ah7, ip7, 49);
                ap9 = _mm256_permute2x128_si256(ah9, ip9, 32);
                ap10 = _mm256_permute2x128_si256(ah9, ip9, 49);
                ap11 = _mm256_permute2x128_si256(ah11, ip11, 32);
                ap12 = _mm256_permute2x128_si256(ah11, ip11, 49);
                ap13 = _mm256_permute2x128_si256(ah13, ip13, 32);
                ap14 = _mm256_permute2x128_si256(ah13, ip13, 49);
                ap15 = _mm256_permute2x128_si256(ah15, ip15, 32);
                ap16 = _mm256_permute2x128_si256(ah15, ip15, 49);

                _mm256_storeu_si256((__m256i*)out0, ap1);
                _mm256_storeu_si256((__m256i*)out1, ap3);
                _mm256_storeu_si256((__m256i*)out2, ap5);
                _mm256_storeu_si256((__m256i*)out3, ap7);
                _mm256_storeu_si256((__m256i*)out4, ap9);
                _mm256_storeu_si256((__m256i*)out5, ap11);
                _mm256_storeu_si256((__m256i*)out6, ap13);
                _mm256_storeu_si256((__m256i*)out7, ap15);
                _mm256_storeu_si256((__m256i*)out8, ap2);
                _mm256_storeu_si256((__m256i*)out9, ap4);
                _mm256_storeu_si256((__m256i*)out10, ap6);
                _mm256_storeu_si256((__m256i*)out11, ap8);
                _mm256_storeu_si256((__m256i*)out12, ap10);
                _mm256_storeu_si256((__m256i*)out13, ap12);
                _mm256_storeu_si256((__m256i*)out14, ap14);
                _mm256_storeu_si256((__m256i*)out15, ap16);

                out0 += M * VEC_16CAP;
                out1 += M * VEC_16CAP;
                out2 += M * VEC_16CAP;
                out3 += M * VEC_16CAP;
                out4 += M * VEC_16CAP;
                out5 += M * VEC_16CAP;
                out6 += M * VEC_16CAP;
                out7 += M * VEC_16CAP;
                out8 += M * VEC_16CAP;
                out9 += M * VEC_16CAP;
                out10 += M * VEC_16CAP;
                out11 += M * VEC_16CAP;
                out12 += M * VEC_16CAP;
                out13 += M * VEC_16CAP;
                out14 += M * VEC_16CAP;
                out15 += M * VEC_16CAP;

                a = _mm256_lddqu_si256((__m256i*)in0);
                b = _mm256_lddqu_si256((__m256i*)in1);
                c = _mm256_lddqu_si256((__m256i*)in2);
                d = _mm256_lddqu_si256((__m256i*)in3);
                e = _mm256_lddqu_si256((__m256i*)in4);
                f = _mm256_lddqu_si256((__m256i*)in5);
                g = _mm256_lddqu_si256((__m256i*)in6);
                h = _mm256_lddqu_si256((__m256i*)in7);
                i = _mm256_lddqu_si256((__m256i*)in8);
                j = _mm256_lddqu_si256((__m256i*)in9);
                k = _mm256_lddqu_si256((__m256i*)in10);
                l = _mm256_lddqu_si256((__m256i*)in11);
                m = _mm256_lddqu_si256((__m256i*)in12);
                n = _mm256_lddqu_si256((__m256i*)in13);
                o = _mm256_lddqu_si256((__m256i*)in14);
                p = _mm256_lddqu_si256((__m256i*)in15);
            }

            in0 = in0 - N_VEC + VEC_16CAP * N;
            out0 = out0 - N_VEC * M + VEC_16CAP;
        }

        for (ii = M_VEC; ii < M; ii++)
        {
            for (jj = 0; jj < N_VEC; jj++)
            {
                O[jj * M + ii] = I[ii * N + jj];
            }

        }

        for (ii = N_VEC; ii < N; ii++)
        {
            for (jj = 0; jj < M; jj++)
            {
                O[ii * M + jj] = I[jj * N + ii];
            }
        }

        return;
    }

    // INTERLEAVE
    // MAX M is 8, MAX N is UINT16_MAX
    if (M == 2)
    {
        in_end = in0 + N_VEC;
        int16_t *in1 = in0 + N;
        int16_t *out0 = O,
                *out1 = out0 + 8,
                *out2 = out1 + 8,
                *out3 = out2 + 8;

        __m256i a = _mm256_lddqu_si256((__m256i*)in0);
        __m256i b = _mm256_lddqu_si256((__m256i*)in1);
        __m256i ab_lo, ab_hi;
        __m128i ab4, ab8, ab12, ab16;

        for (; in0 < in_end;)
        {
            in0 += VEC_16CAP;
            in1 += VEC_16CAP;

            ab_lo = _mm256_unpacklo_epi16(a, b);
            ab_hi = _mm256_unpackhi_epi16(a, b);

            ab4 = _mm256_castsi256_si128(ab_lo);
            ab8 = _mm256_castsi256_si128(ab_hi);

            ab12 = _mm256_extracti128_si256(ab_lo, 1);
            ab16 = _mm256_extracti128_si256(ab_hi, 1);

            _mm_storeu_si128((__m128i*)out0, ab4);
            _mm_storeu_si128((__m128i*)out1, ab8);
            _mm_storeu_si128((__m128i*)out2, ab12);
            _mm_storeu_si128((__m128i*)out3, ab16);

            out0 += M * VEC_16CAP;
            out1 += M * VEC_16CAP;
            out2 += M * VEC_16CAP;
            out3 += M * VEC_16CAP;

            a = _mm256_lddqu_si256((__m256i*)in0);
            b = _mm256_lddqu_si256((__m256i*)in1);
        }

        out1 = out0 + 1;
        for (i = N_VEC; i < N; i++)
        {
            *out0 = *in0++;
            *out1 = *in1++;

            out0 += M;
            out1 += M;
        }

        return;
    }

    if (M == 3)
    {
        int16_t *in1 = in0 + 4,
                *in2 = in1 + 4;

        const int32_t scale = 2;
        __m256i gather1 = _mm256_setr_epi32(0, N, 2 * N, 12, 2, N + 2, 2 * N + 2, N + 12);
        __m256i gather2 = _mm256_setr_epi32(0, N, 2 * N, 2 * N + 8, 2, N + 2, 2 * N + 2, 10);
        __m256i gather3 = _mm256_setr_epi32(0, N, 2 * N, N + 6, 2, N + 2, 2 * N + 2, 2 * N + 6);

        __m256i shuffle_mask = _mm256_setr_epi8(0, 1, 4, 5, 8, 9, 2, 3, 6, 7, 10, 11, 12, 13, 14, 15,
            0, 1, 4, 5, 8, 9, 2, 3, 6, 7, 10, 11, 12, 13, 14, 15);
        __m256i permute_mask = _mm256_setr_epi32(0, 1, 2, 4, 5, 6, 3, 7);
        __m256i permute_mask2 = _mm256_setr_epi32(0, 1, 2, 0, 3, 6, 7, 0);
        __m256i store_mask = _mm256_setr_epi64x(UINT64_MAX, UINT64_MAX, UINT64_MAX, 0);

        __m256i mix1 = _mm256_i32gather_epi32((int32_t*)in0, gather1, scale);
        __m256i mix2 = _mm256_i32gather_epi32((int32_t*)in1, gather2, scale);
        __m256i mix3 = _mm256_i32gather_epi32((int32_t*)in2, gather3, scale);

        __m256i abc1_4, abc7_10, abc13_16;
        __m256i abc_rem;

        int16_t *out0 = O,
            *out1 = out0 + 4 * M,
            *out2 = out1 + 4 * M,
            *out3 = out2 + 4 * M;

        in_end = in0 + N_VEC;
        for (; in0 < in_end;)
        {
            in0 += VEC_16CAP;
            in1 += VEC_16CAP;
            in2 += VEC_16CAP;

            abc1_4 = _mm256_shuffle_epi8(mix1, shuffle_mask);
            abc1_4 = _mm256_permutevar8x32_epi32(abc1_4, permute_mask);

            abc7_10 = _mm256_shuffle_epi8(mix2, shuffle_mask);
            abc7_10 = _mm256_permutevar8x32_epi32(abc7_10, permute_mask);

            abc13_16 = _mm256_shuffle_epi8(mix3, shuffle_mask);
            abc13_16 = _mm256_permutevar8x32_epi32(abc13_16, permute_mask);

            abc_rem = _mm256_permute2x128_si256(abc1_4, abc7_10, 49);
            abc_rem = _mm256_permute4x64_epi64(abc_rem, 13);
            abc_rem = _mm256_permute2x128_si256(abc_rem, abc13_16, 48);
            abc_rem = _mm256_permutevar8x32_epi32(abc_rem, permute_mask2);
            abc_rem = _mm256_shuffle_epi8(abc_rem, shuffle_mask);
            abc_rem = _mm256_permutevar8x32_epi32(abc_rem, permute_mask);

            _mm256_maskstore_epi64((long long*)out0, store_mask, abc1_4);
            _mm256_maskstore_epi64((long long *)out1, store_mask, abc7_10);
            _mm256_maskstore_epi64((long long *)out2, store_mask, abc13_16);
            _mm256_maskstore_epi64((long long *)out3, store_mask, abc_rem);

            out0 += VEC_16CAP * M;
            out1 += VEC_16CAP * M;
            out2 += VEC_16CAP * M;
            out3 += VEC_16CAP * M;

            mix1 = _mm256_i32gather_epi32((int32_t*)in0, gather1, scale);
            mix2 = _mm256_i32gather_epi32((int32_t*)in1, gather2, scale);
            mix3 = _mm256_i32gather_epi32((int32_t*)in2, gather3, scale);
        }

        out1 = out0 + 1;
        out2 = out1 + 1;
        in1 = in0 + N;
        in2 = in1 + N;
        for (i = N_VEC; i < N; i++)
        {
            *out0 = *in0++;
            *out1 = *in1++;
            *out2 = *in2++;

            out0 += M;
            out1 += M;
            out2 += M;
        }

        return;
    }

    if (M == 4)
    {
        in_end = in0 + N_VEC;
        int16_t *in1 = in0 + N,
                *in2 = in1 + N,
                *in3 = in2 + N;

        int16_t *out0 = O,
            *out1 = out0 + 8,
            *out2 = out1 + 8,
            *out3 = out2 + 8,
            *out4 = out3 + 8,
            *out5 = out4 + 8,
            *out6 = out5 + 8,
            *out7 = out6 + 8;

        __m256i a = _mm256_lddqu_si256((__m256i*)in0);
        __m256i b = _mm256_lddqu_si256((__m256i*)in1);
        __m256i c = _mm256_lddqu_si256((__m256i*)in2);
        __m256i d = _mm256_lddqu_si256((__m256i*)in3);

        __m256i ab_lo, ab_hi, cd_lo, cd_hi;
        __m256i abcd_lo, abcd_lo2, abcd_hi, abcd_hi2;
        __m128i abcd8, abcd16, abcd24, abcd32,
            abcd40, abcd48, abcd56, abcd64;

        for (; in0 < in_end;)
        {
            in0 += VEC_16CAP;
            in1 += VEC_16CAP;
            in2 += VEC_16CAP;
            in3 += VEC_16CAP;

            ab_lo = _mm256_unpacklo_epi16(a, b);
            ab_hi = _mm256_unpackhi_epi16(a, b);
            cd_lo = _mm256_unpacklo_epi16(c, d);
            cd_hi = _mm256_unpackhi_epi16(c, d);

            abcd_lo = _mm256_unpacklo_epi32(ab_lo, cd_lo);
            abcd_lo2 = _mm256_unpacklo_epi32(ab_hi, cd_hi);
            abcd_hi = _mm256_unpackhi_epi32(ab_lo, cd_lo);
            abcd_hi2 = _mm256_unpackhi_epi32(ab_hi, cd_hi);

            abcd8 = _mm256_castsi256_si128(abcd_lo);
            abcd16 = _mm256_castsi256_si128(abcd_hi);
            abcd24 = _mm256_castsi256_si128(abcd_lo2);
            abcd32 = _mm256_castsi256_si128(abcd_hi2);

            abcd40 = _mm256_extracti128_si256(abcd_lo, 1);
            abcd48 = _mm256_extracti128_si256(abcd_hi, 1);
            abcd56 = _mm256_extracti128_si256(abcd_lo2, 1);
            abcd64 = _mm256_extracti128_si256(abcd_hi2, 1);

            _mm_storeu_si128((__m128i*) out0, abcd8);
            _mm_storeu_si128((__m128i*) out1, abcd16);
            _mm_storeu_si128((__m128i*) out2, abcd24);
            _mm_storeu_si128((__m128i*) out3, abcd32);
            _mm_storeu_si128((__m128i*) out4, abcd40);
            _mm_storeu_si128((__m128i*) out5, abcd48);
            _mm_storeu_si128((__m128i*) out6, abcd56);
            _mm_storeu_si128((__m128i*) out7, abcd64);

            out0 += 64;
            out1 += 64;
            out2 += 64;
            out3 += 64;
            out4 += 64;
            out5 += 64;
            out6 += 64;
            out7 += 64;

            a = _mm256_lddqu_si256((__m256i*)in0);
            b = _mm256_lddqu_si256((__m256i*)in1);
            c = _mm256_lddqu_si256((__m256i*)in2);
            d = _mm256_lddqu_si256((__m256i*)in3);
        }

        out1 = out0 + 1;
        out2 = out1 + 1;
        out3 = out2 + 1;
        for (i = N_VEC; i < N; i++)
        {
            *out0 = *in0++;
            *out1 = *in1++;
            *out2 = *in2++;
            *out3 = *in3++;

            out0 += M;
            out1 += M;
            out2 += M;
            out3 += M;
        }

        return;
    }

    if (M == 5)
    {
        uint32_t i;
        __m256i a, b, c, d,
            ab_lo, ab_hi, cd_lo, cd_hi,
            abcd_lo, abcd_lohi, abcd_hilo, abcd_hi;
        __m128i v1, v2, v3, v4, v5, v6, v7, v8,
            v1_hi, v2_hi, v3_hi, v4_hi, v5_hi, v6_hi, v7_hi, v8_hi;

        in_end = in0 + N_VEC;
        int16_t *in1 = in0 + N,
            *in2 = in1 + N,
            *in3 = in2 + N,
            *in4 = in3 + N;

        int16_t *out0 = O,
            *out1 = out0 + M,
            *out2 = out1 + M,
            *out3 = out2 + M,
            *out4 = out3 + M,
            *out5 = out4 + M,
            *out6 = out5 + M,
            *out7 = out6 + M,
            *out8 = out7 + M,
            *out9 = out8 + M,
            *out10 = out9 + M,
            *out11 = out10 + M,
            *out12 = out11 + M,
            *out13 = out12 + M,
            *out14 = out13 + M,
            *out15 = out14 + M;

        a = _mm256_lddqu_si256((__m256i*) in0);
        b = _mm256_lddqu_si256((__m256i*) in1);
        c = _mm256_lddqu_si256((__m256i*) in2);
        d = _mm256_lddqu_si256((__m256i*) in3);

        for (; in0 < in_end;)
        {
            in0 += VEC_16CAP;
            in1 += VEC_16CAP;
            in2 += VEC_16CAP;
            in3 += VEC_16CAP;

            ab_lo = _mm256_unpacklo_epi16(a, b);
            ab_hi = _mm256_unpackhi_epi16(a, b);

            cd_lo = _mm256_unpacklo_epi16(c, d);
            cd_hi = _mm256_unpackhi_epi16(c, d);

            abcd_lo = _mm256_unpacklo_epi32(ab_lo, cd_lo);
            abcd_lohi = _mm256_unpackhi_epi32(ab_lo, cd_lo);
            abcd_hilo = _mm256_unpacklo_epi32(ab_hi, cd_hi);
            abcd_hi = _mm256_unpackhi_epi32(ab_hi, cd_hi);


            v1 = _mm256_castsi256_si128(abcd_lo);
            v2 = _mm256_castsi256_si128(abcd_lohi);
            v3 = _mm256_castsi256_si128(abcd_hilo);
            v4 = _mm256_castsi256_si128(abcd_hi);
            v5 = _mm256_extracti128_si256(abcd_lo, 1);
            v6 = _mm256_extracti128_si256(abcd_lohi, 1);
            v7 = _mm256_extracti128_si256(abcd_hilo, 1);
            v8 = _mm256_extracti128_si256(abcd_hi, 1);

            v1_hi = _mm_bsrli_si128(v1, 8);
            v2_hi = _mm_bsrli_si128(v2, 8);
            v3_hi = _mm_bsrli_si128(v3, 8);
            v4_hi = _mm_bsrli_si128(v4, 8);
            v5_hi = _mm_bsrli_si128(v5, 8);
            v6_hi = _mm_bsrli_si128(v6, 8);
            v7_hi = _mm_bsrli_si128(v7, 8);
            v8_hi = _mm_bsrli_si128(v8, 8);

            _mm_storeu_si64(out0, v1);
            *(out0 + 4) = *in4++;

            _mm_storeu_si64(out1, v1_hi);
            *(out1 + 4) = *in4++;

            _mm_storeu_si64(out2, v2);
            *(out2 + 4) = *in4++;

            _mm_storeu_si64(out3, v2_hi);
            *(out3 + 4) = *in4++;

            _mm_storeu_si64(out4, v3);
            *(out4 + 4) = *in4++;

            _mm_storeu_si64(out5, v3_hi);
            *(out5 + 4) = *in4++;

            _mm_storeu_si64(out6, v4);
            *(out6 + 4) = *in4++;

            _mm_storeu_si64(out7, v4_hi);
            *(out7 + 4) = *in4++;

            _mm_storeu_si64(out8, v5);
            *(out8 + 4) = *in4++;

            _mm_storeu_si64(out9, v5_hi);
            *(out9 + 4) = *in4++;

            _mm_storeu_si64(out10, v6);
            *(out10 + 4) = *in4++;

            _mm_storeu_si64(out11, v6_hi);
            *(out11 + 4) = *in4++;

            _mm_storeu_si64(out12, v7);
            *(out12 + 4) = *in4++;

            _mm_storeu_si64(out13, v7_hi);
            *(out13 + 4) = *in4++;

            _mm_storeu_si64(out14, v8);
            *(out14 + 4) = *in4++;

            _mm_storeu_si64(out15, v8_hi);
            *(out15 + 4) = *in4++;

            out0 +=  VEC_16CAP * M;
            out1 +=  VEC_16CAP * M;
            out2 +=  VEC_16CAP * M;
            out3 +=  VEC_16CAP * M;
            out4 +=  VEC_16CAP * M;
            out5 +=  VEC_16CAP * M;
            out6 +=  VEC_16CAP * M;
            out7 +=  VEC_16CAP * M;
            out8 +=  VEC_16CAP * M;
            out9 +=  VEC_16CAP * M;
            out10 += VEC_16CAP * M;
            out11 += VEC_16CAP * M;
            out12 += VEC_16CAP * M;
            out13 += VEC_16CAP * M;
            out14 += VEC_16CAP * M;
            out15 += VEC_16CAP * M;

            a = _mm256_lddqu_si256((__m256i*) in0);
            b = _mm256_lddqu_si256((__m256i*) in1);
            c = _mm256_lddqu_si256((__m256i*) in2);
            d = _mm256_lddqu_si256((__m256i*) in3);
        }

        out1 = out0 + 1;
        out2 = out1 + 1;
        out3 = out2 + 1;
        out4 = out3 + 1;
        in1 = in0 + N;
        in2 = in1 + N;
        in3 = in2 + N;
        in4 = in3 + N;
        for (i = N_VEC; i < N; i++)
        {
            *out0 = *in0++;
            *out1 = *in1++;
            *out2 = *in2++;
            *out3 = *in3++;
            *out4 = *in4++;

            out0 += M;
            out1 += M;
            out2 += M;
            out3 += M;
            out4 += M;
        }

        return;
    }

    if (M == 6)
    {
        uint32_t i;
        int16_t *in1 = in0 + N,
            *in2 = in1 + N,
            *in3 = in2 + N,
            *in4 = in3 + N,
            *in5 = in4 + N;

        int16_t *out0 = O,
            *out1 = out0 + 2 * M,
            *out2 = out1 + 2 * M,
            *out3 = out2 + 2 * M,
            *out4 = out3 + 2 * M,
            *out5 = out4 + 2 * M,
            *out6 = out5 + 2 * M,
            *out7 = out6 + 2 * M;

        __m256i a = _mm256_lddqu_si256((__m256i*)in0);
        __m256i b = _mm256_lddqu_si256((__m256i*)in1);
        __m256i c = _mm256_lddqu_si256((__m256i*)in2);
        __m256i d = _mm256_lddqu_si256((__m256i*)in3);
        __m256i e = _mm256_lddqu_si256((__m256i*)in4);
        __m256i f = _mm256_lddqu_si256((__m256i*)in5);

        __m256i ab_lo, ab_hi, cd_lo, cd_hi, ef_lo, ef_hi;
        __m256i abcd_lo, abcd_lo2, abcd_hi, abcd_hi2;
        __m256i mix1, mix2, mix3, mix4, mix5, mix6, mix7, mix8;

        __m256i permute_mask = _mm256_setr_epi32(0, 1, 4, 2, 3, 5, 6, 7);
        __m256i permute_mask2 = _mm256_setr_epi32(0, 1, 6, 2, 3, 7, 4, 5);
        __m256i store_mask = _mm256_setr_epi64x(UINT64_MAX, UINT64_MAX, UINT64_MAX, 0);

        in_end = in0 + N_VEC;
        for (; in0 < in_end;)
        {
            in0 += VEC_16CAP;
            in1 += VEC_16CAP;
            in2 += VEC_16CAP;
            in3 += VEC_16CAP;
            in4 += VEC_16CAP;
            in5 += VEC_16CAP;

            ab_lo = _mm256_unpacklo_epi16(a, b);
            ab_hi = _mm256_unpackhi_epi16(a, b);
            cd_lo = _mm256_unpacklo_epi16(c, d);
            cd_hi = _mm256_unpackhi_epi16(c, d);
            ef_lo = _mm256_unpacklo_epi16(e, f);
            ef_hi = _mm256_unpackhi_epi16(e, f);

            abcd_lo = _mm256_unpacklo_epi32(ab_lo, cd_lo);
            abcd_lo2 = _mm256_unpacklo_epi32(ab_hi, cd_hi);
            abcd_hi = _mm256_unpackhi_epi32(ab_lo, cd_lo);
            abcd_hi2 = _mm256_unpackhi_epi32(ab_hi, cd_hi);

            mix1 = _mm256_permute2x128_si256(abcd_lo, ef_lo, 32);
            mix2 = _mm256_permute2x128_si256(abcd_lo, ef_lo, 49);

            mix3 = _mm256_permute2x128_si256(abcd_hi, ef_lo, 32);
            mix4 = _mm256_permute2x128_si256(abcd_hi, ef_lo, 49);

            mix5 = _mm256_permute2x128_si256(abcd_lo2, ef_hi, 32);
            mix6 = _mm256_permute2x128_si256(abcd_lo2, ef_hi, 49);

            mix7 = _mm256_permute2x128_si256(abcd_hi2, ef_hi, 32);
            mix8 = _mm256_permute2x128_si256(abcd_hi2, ef_hi, 49);

            mix1 = _mm256_permutevar8x32_epi32(mix1, permute_mask);
            mix2 = _mm256_permutevar8x32_epi32(mix2, permute_mask);
            mix3 = _mm256_permutevar8x32_epi32(mix3, permute_mask2);
            mix4 = _mm256_permutevar8x32_epi32(mix4, permute_mask2);
            mix5 = _mm256_permutevar8x32_epi32(mix5, permute_mask);
            mix6 = _mm256_permutevar8x32_epi32(mix6, permute_mask);
            mix7 = _mm256_permutevar8x32_epi32(mix7, permute_mask2);
            mix8 = _mm256_permutevar8x32_epi32(mix8, permute_mask2);

            _mm256_maskstore_epi64((long long*)out0, store_mask, mix1);
            _mm256_maskstore_epi64((long long*)out1, store_mask, mix3);
            _mm256_maskstore_epi64((long long*)out2, store_mask, mix5);
            _mm256_maskstore_epi64((long long*)out3, store_mask, mix7);
            _mm256_maskstore_epi64((long long*)out4, store_mask, mix2);
            _mm256_maskstore_epi64((long long*)out5, store_mask, mix4);
            _mm256_maskstore_epi64((long long*)out6, store_mask, mix6);
            _mm256_maskstore_epi64((long long*)out7, store_mask, mix8);

            out0 += VEC_16CAP * M;
            out1 += VEC_16CAP * M;
            out2 += VEC_16CAP * M;
            out3 += VEC_16CAP * M;
            out4 += VEC_16CAP * M;
            out5 += VEC_16CAP * M;
            out6 += VEC_16CAP * M;
            out7 += VEC_16CAP * M;

            a = _mm256_lddqu_si256((__m256i*)in0);
            b = _mm256_lddqu_si256((__m256i*)in1);
            c = _mm256_lddqu_si256((__m256i*)in2);
            d = _mm256_lddqu_si256((__m256i*)in3);
            e = _mm256_lddqu_si256((__m256i*)in4);
            f = _mm256_lddqu_si256((__m256i*)in5);
        }

        out1 = out0 + 1;
        out2 = out1 + 1;
        out3 = out2 + 1;
        out4 = out3 + 1;
        out5 = out4 + 1;
        for (i = N_VEC; i < N; i++)
        {
            *out0 = *in0++;
            *out1 = *in1++;
            *out2 = *in2++;
            *out3 = *in3++;
            *out4 = *in4++;
            *out5 = *in5++;

            out0 += M;
            out1 += M;
            out2 += M;
            out3 += M;
            out4 += M;
            out5 += M;
        }

        return;
    }

    if (M == 7)
    {
        int16_t *in1 = in0 + 2,
            *in2 = in1 + 2,
            *in3 = in2 + 2,
            *in4 = in3 + 2,
            *in5 = in4 + 2,
            *in6 = in5 + 2,
            *in7 = in6 + 2;

        int16_t *out0 = O,
            *out1 = out0 + 2 * M,
            *out2 = out1 + 2 * M,
            *out3 = out2 + 2 * M,
            *out4 = out3 + 2 * M,
            *out5 = out4 + 2 * M,
            *out6 = out5 + 2 * M,
            *out7 = out6 + 2 * M;

        __m256i gather = _mm256_setr_epi32(0, N, 2 * N, 3 * N, 4 * N, 5 * N, 6 * N, 0);

        __m256i mix1, mix2, mix3, mix4, mix5, mix6, mix7, mix8;

        __m256i shuffle_mask = _mm256_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15,
                                                0, 1, 4, 5, 8, 9, 14, 15, 2, 3, 6, 7, 10, 11, 12, 13);
        __m256i permute_mask = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);
        __m256i store_mask = _mm256_setr_epi32(UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX,
            UINT32_MAX, UINT32_MAX, UINT32_MAX, 0);

        const int32_t scale = 2;

        mix1 = _mm256_i32gather_epi32((int*)in0, gather, scale);
        mix2 = _mm256_i32gather_epi32((int*)in1, gather, scale);
        mix3 = _mm256_i32gather_epi32((int*)in2, gather, scale);
        mix4 = _mm256_i32gather_epi32((int*)in3, gather, scale);
        mix5 = _mm256_i32gather_epi32((int*)in4, gather, scale);
        mix6 = _mm256_i32gather_epi32((int*)in5, gather, scale);
        mix7 = _mm256_i32gather_epi32((int*)in6, gather, scale);
        mix8 = _mm256_i32gather_epi32((int*)in7, gather, scale);

        in_end = in0 + N_VEC;
        for (; in0 < in_end;)
        {
            in0 += VEC_16CAP;
            in1 += VEC_16CAP;
            in2 += VEC_16CAP;
            in3 += VEC_16CAP;
            in4 += VEC_16CAP;
            in5 += VEC_16CAP;
            in6 += VEC_16CAP;
            in7 += VEC_16CAP;

            mix1 = _mm256_shuffle_epi8(mix1, shuffle_mask);
            mix1 = _mm256_permutevar8x32_epi32(mix1, permute_mask);
            mix1 = _mm256_inserti128_si256(
                mix1, _mm_bsrli_si128(_mm256_extracti128_si256(mix1, 1), 2), 1);

            mix2 = _mm256_shuffle_epi8(mix2, shuffle_mask);
            mix2 = _mm256_permutevar8x32_epi32(mix2, permute_mask);
            mix2 = _mm256_inserti128_si256(
                mix2, _mm_bsrli_si128(_mm256_extracti128_si256(mix2, 1), 2), 1);

            mix3 = _mm256_shuffle_epi8(mix3, shuffle_mask);
            mix3 = _mm256_permutevar8x32_epi32(mix3, permute_mask);
            mix3 = _mm256_inserti128_si256(
                mix3, _mm_bsrli_si128(_mm256_extracti128_si256(mix3, 1), 2), 1);

            mix4 = _mm256_shuffle_epi8(mix4, shuffle_mask);
            mix4 = _mm256_permutevar8x32_epi32(mix4, permute_mask);
            mix4 = _mm256_inserti128_si256(
                mix4, _mm_bsrli_si128(_mm256_extracti128_si256(mix4, 1), 2), 1);

            mix5 = _mm256_shuffle_epi8(mix5, shuffle_mask);
            mix5 = _mm256_permutevar8x32_epi32(mix5, permute_mask);
            mix5 = _mm256_inserti128_si256(
                mix5, _mm_bsrli_si128(_mm256_extracti128_si256(mix5, 1), 2), 1);

            mix6 = _mm256_shuffle_epi8(mix6, shuffle_mask);
            mix6 = _mm256_permutevar8x32_epi32(mix6, permute_mask);
            mix6 = _mm256_inserti128_si256(
                mix6, _mm_bsrli_si128(_mm256_extracti128_si256(mix6, 1), 2), 1);

            mix7 = _mm256_shuffle_epi8(mix7, shuffle_mask);
            mix7 = _mm256_permutevar8x32_epi32(mix7, permute_mask);
            mix7 = _mm256_inserti128_si256(
                mix7, _mm_bsrli_si128(_mm256_extracti128_si256(mix7, 1), 2), 1);

            mix8 = _mm256_shuffle_epi8(mix8, shuffle_mask);
            mix8 = _mm256_permutevar8x32_epi32(mix8, permute_mask);
            mix8 = _mm256_inserti128_si256(
                mix8, _mm_bsrli_si128(_mm256_extracti128_si256(mix8, 1), 2), 1);

            _mm256_maskstore_epi32((int*)out0, store_mask, mix1);
            _mm256_maskstore_epi32((int*)out1, store_mask, mix2);
            _mm256_maskstore_epi32((int*)out2, store_mask, mix3);
            _mm256_maskstore_epi32((int*)out3, store_mask, mix4);
            _mm256_maskstore_epi32((int*)out4, store_mask, mix5);
            _mm256_maskstore_epi32((int*)out5, store_mask, mix6);
            _mm256_maskstore_epi32((int*)out6, store_mask, mix7);
            _mm256_maskstore_epi32((int*)out7, store_mask, mix8);

            out0 += VEC_16CAP * M;
            out1 += VEC_16CAP * M;
            out2 += VEC_16CAP * M;
            out3 += VEC_16CAP * M;
            out4 += VEC_16CAP * M;
            out5 += VEC_16CAP * M;
            out6 += VEC_16CAP * M;
            out7 += VEC_16CAP * M;

            mix1 = _mm256_i32gather_epi32((int*)in0, gather, scale);
            mix2 = _mm256_i32gather_epi32((int*)in1, gather, scale);
            mix3 = _mm256_i32gather_epi32((int*)in2, gather, scale);
            mix4 = _mm256_i32gather_epi32((int*)in3, gather, scale);
            mix5 = _mm256_i32gather_epi32((int*)in4, gather, scale);
            mix6 = _mm256_i32gather_epi32((int*)in5, gather, scale);
            mix7 = _mm256_i32gather_epi32((int*)in6, gather, scale);
            mix8 = _mm256_i32gather_epi32((int*)in7, gather, scale);
        }

        out1 = out0 + 1;
        out2 = out1 + 1;
        out3 = out2 + 1;
        out4 = out3 + 1;
        out5 = out4 + 1;
        out6 = out5 + 1;
        in1 = in0 + N;
        in2 = in1 + N;
        in3 = in2 + N;
        in4 = in3 + N;
        in5 = in4 + N;
        in6 = in5 + N;
        for (i = N_VEC; i < N; i++)
        {
            *out0 = *in0++;
            *out1 = *in1++;
            *out2 = *in2++;
            *out3 = *in3++;
            *out4 = *in4++;
            *out5 = *in5++;
            *out6 = *in6++;

            out0 += M;
            out1 += M;
            out2 += M;
            out3 += M;
            out4 += M;
            out5 += M;
            out6 += M;
        }

        return;
    }

    if (M == 8)
    {
        int16_t *in1 = in0 + N,
            *in2 = in1 + N,
            *in3 = in2 + N,
            *in4 = in3 + N,
            *in5 = in4 + N,
            *in6 = in5 + N,
            *in7 = in6 + N;

        int16_t *out0 = O,
            *out1 = out0 + 2 * M,
            *out2 = out1 + 2 * M,
            *out3 = out2 + 2 * M,
            *out4 = out3 + 2 * M,
            *out5 = out4 + 2 * M,
            *out6 = out5 + 2 * M,
            *out7 = out6 + 2 * M;

        __m256i a = _mm256_lddqu_si256((__m256i*) in0);
        __m256i b = _mm256_lddqu_si256((__m256i*) in1);
        __m256i c = _mm256_lddqu_si256((__m256i*) in2);
        __m256i d = _mm256_lddqu_si256((__m256i*) in3);
        __m256i e = _mm256_lddqu_si256((__m256i*) in4);
        __m256i f = _mm256_lddqu_si256((__m256i*) in5);
        __m256i g = _mm256_lddqu_si256((__m256i*) in6);
        __m256i h = _mm256_lddqu_si256((__m256i*) in7);

        __m256i ab_lo, ab_hi, cd_lo, cd_hi, ef_lo, ef_hi, gh_lo, gh_hi;
        __m256i abcd_lo, abcd_lohi, abcd_hi, abcd_hilo, efgh_lo, efgh_lohi, efgh_hi, efgh_hilo;
        __m256i pack1, pack2, pack3, pack4, pack5, pack6, pack7, pack8;

        in_end = in0 + N_VEC;
        for (; in0 < in_end;)
        {
            in0 += VEC_16CAP;
            in1 += VEC_16CAP;
            in2 += VEC_16CAP;
            in3 += VEC_16CAP;
            in4 += VEC_16CAP;
            in5 += VEC_16CAP;
            in6 += VEC_16CAP;
            in7 += VEC_16CAP;

            ab_lo = _mm256_unpacklo_epi16(a, b);
            ab_hi = _mm256_unpackhi_epi16(a, b);

            cd_lo = _mm256_unpacklo_epi16(c, d);
            cd_hi = _mm256_unpackhi_epi16(c, d);

            ef_lo = _mm256_unpacklo_epi16(e, f);
            ef_hi = _mm256_unpackhi_epi16(e, f);

            gh_lo = _mm256_unpacklo_epi16(g, h);
            gh_hi = _mm256_unpackhi_epi16(g, h);

            abcd_lo = _mm256_unpacklo_epi32(ab_lo, cd_lo);
            abcd_lohi = _mm256_unpackhi_epi32(ab_lo, cd_lo);

            abcd_hi = _mm256_unpacklo_epi32(ab_hi, cd_hi);
            abcd_hilo = _mm256_unpackhi_epi32(ab_hi, cd_hi);

            efgh_lo = _mm256_unpacklo_epi32(ef_lo, gh_lo);
            efgh_lohi = _mm256_unpackhi_epi32(ef_lo, gh_lo);

            efgh_hi = _mm256_unpacklo_epi32(ef_hi, gh_hi);
            efgh_hilo = _mm256_unpackhi_epi32(ef_hi, gh_hi);

            pack1 = _mm256_unpacklo_epi64(abcd_lo, efgh_lo);
            pack2 = _mm256_unpackhi_epi64(abcd_lo, efgh_lo);
            pack3 = _mm256_unpacklo_epi64(abcd_hi, efgh_hi);
            pack4 = _mm256_unpackhi_epi64(abcd_hi, efgh_hi);
            pack5 = _mm256_unpacklo_epi64(abcd_lohi, efgh_lohi);
            pack6 = _mm256_unpackhi_epi64(abcd_lohi, efgh_lohi);
            pack7 = _mm256_unpacklo_epi64(abcd_hilo, efgh_hilo);
            pack8 = _mm256_unpackhi_epi64(abcd_hilo, efgh_hilo);

            _mm_stream_si128((__m128i*) out0, _mm256_castsi256_si128(pack1));
            _mm_stream_si128((__m128i*) (out0 + SSE_16CAP), _mm256_castsi256_si128(pack2));
            _mm_stream_si128((__m128i*) out1, _mm256_castsi256_si128(pack5));
            _mm_stream_si128((__m128i*) (out1 + SSE_16CAP), _mm256_castsi256_si128(pack6));
            _mm_stream_si128((__m128i*) out2, _mm256_castsi256_si128(pack3));
            _mm_stream_si128((__m128i*) (out2 + SSE_16CAP), _mm256_castsi256_si128(pack4));
            _mm_stream_si128((__m128i*) out3, _mm256_castsi256_si128(pack7));
            _mm_stream_si128((__m128i*) (out3 + SSE_16CAP), _mm256_castsi256_si128(pack8));

            _mm_stream_si128((__m128i*) out4, _mm256_extracti128_si256(pack1, 1));
            _mm_stream_si128((__m128i*) (out4 + SSE_16CAP), _mm256_extracti128_si256(pack2, 1));
            _mm_stream_si128((__m128i*) out5, _mm256_extracti128_si256(pack5, 1));
            _mm_stream_si128((__m128i*) (out5 + SSE_16CAP), _mm256_extracti128_si256(pack6, 1));
            _mm_stream_si128((__m128i*) out6, _mm256_extracti128_si256(pack3, 1));
            _mm_stream_si128((__m128i*) (out6 + SSE_16CAP), _mm256_extracti128_si256(pack4, 1));
            _mm_stream_si128((__m128i*) out7, _mm256_extracti128_si256(pack7, 1));
            _mm_stream_si128((__m128i*) (out7 + SSE_16CAP), _mm256_extracti128_si256(pack8, 1));

            out0 += M * VEC_16CAP;
            out1 += M * VEC_16CAP;
            out2 += M * VEC_16CAP;
            out3 += M * VEC_16CAP;
            out4 += M * VEC_16CAP;
            out5 += M * VEC_16CAP;
            out6 += M * VEC_16CAP;
            out7 += M * VEC_16CAP;

            a = _mm256_lddqu_si256((__m256i*) in0);
            b = _mm256_lddqu_si256((__m256i*) in1);
            c = _mm256_lddqu_si256((__m256i*) in2);
            d = _mm256_lddqu_si256((__m256i*) in3);
            e = _mm256_lddqu_si256((__m256i*) in4);
            f = _mm256_lddqu_si256((__m256i*) in5);
            g = _mm256_lddqu_si256((__m256i*) in6);
            h = _mm256_lddqu_si256((__m256i*) in7);
        }

        M_VEC = M - M % 8;
        for (i = M_VEC; i < M; i++)
        {
            for (j = 0; j < N_VEC; j++)
            {
                O[j * M + i] = I[i * N + j];
            }
        }

        for (i = N_VEC; i < N; i++)
        {
            for (j = 0; j < M; j++)
            {
                O[i * M + j] = I[j * N + i];
            }
        }

        return;
    }

    // DEINTERLAVE 
    // N is 8 max
    if (N == 8)
    {
        M_VEC = M - M % SSE_16CAP;
        N_VEC = N - N % SSE_16CAP;

        __m256i ab, cd, ef, gh, ac, bd, eg, fh;
        __m256i abcd_lo, abcd_hi, efgh_lo, efgh_hi,
            full1, full2, full3, full4;
        __m256i mask1, mask2;
        mask1 = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
        mask2 = _mm256_setr_epi8(0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15,
            0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15);

        int16_t *in1 = in0 + VEC_16CAP,
                *in2 = in1 + VEC_16CAP,
                *in3 = in2 + VEC_16CAP;

        int16_t *out0, *out1, *out2, *out3, *out4, *out5, *out6, *out7;

        ab = _mm256_stream_load_si256((__m256i*)in0);
        cd = _mm256_stream_load_si256((__m256i*)in1);
        ef = _mm256_stream_load_si256((__m256i*)in2);
        gh = _mm256_stream_load_si256((__m256i*)in3);

        out0 = O;
        out1 = out0 + M;
        out2 = out1 + M;
        out3 = out2 + M;
        out4 = out3 + M;
        out5 = out4 + M;
        out6 = out5 + M;
        out7 = out6 + M;

        for (; in0 < in_end;)
        {
            in0 += 4 * VEC_16CAP;
            in1 += 4 * VEC_16CAP;
            in2 += 4 * VEC_16CAP;
            in3 += 4 * VEC_16CAP;

            ab = _mm256_permutevar8x32_epi32(ab, mask1);
            ab = _mm256_shuffle_epi8(ab, mask2);

            cd = _mm256_permutevar8x32_epi32(cd, mask1);
            cd = _mm256_shuffle_epi8(cd, mask2);

            ef = _mm256_permutevar8x32_epi32(ef, mask1);
            ef = _mm256_shuffle_epi8(ef, mask2);

            gh = _mm256_permutevar8x32_epi32(gh, mask1);
            gh = _mm256_shuffle_epi8(gh, mask2);

            abcd_lo = _mm256_unpacklo_epi32(ab, cd);
            abcd_hi = _mm256_unpackhi_epi32(ab, cd);

            efgh_lo = _mm256_unpacklo_epi32(ef, gh);
            efgh_hi = _mm256_unpackhi_epi32(ef, gh);

            full1 = _mm256_unpacklo_epi64(abcd_lo, efgh_lo);
            full2 = _mm256_unpackhi_epi64(abcd_lo, efgh_lo);
            full3 = _mm256_unpacklo_epi64(abcd_hi, efgh_hi);
            full4 = _mm256_unpackhi_epi64(abcd_hi, efgh_hi);

            _mm_storeu_si128((__m128i*) out0, _mm256_castsi256_si128(full1));
            _mm_storeu_si128((__m128i*) out1, _mm256_castsi256_si128(full2));
            _mm_storeu_si128((__m128i*) out2, _mm256_castsi256_si128(full3));
            _mm_storeu_si128((__m128i*) out3, _mm256_castsi256_si128(full4));
            _mm_storeu_si128((__m128i*) out4, _mm256_extracti128_si256(full1, 1));
            _mm_storeu_si128((__m128i*) out5, _mm256_extracti128_si256(full2, 1));
            _mm_storeu_si128((__m128i*) out6, _mm256_extracti128_si256(full3, 1));
            _mm_storeu_si128((__m128i*) out7, _mm256_extracti128_si256(full4, 1));

            out0 += SSE_16CAP;
            out1 = out0 + M;
            out2 = out1 + M;
            out3 = out2 + M;
            out4 = out3 + M;
            out5 = out4 + M;
            out6 = out5 + M;
            out7 = out6 + M;

            ab = _mm256_stream_load_si256((__m256i*)in0);
            cd = _mm256_stream_load_si256((__m256i*)in1);
            ef = _mm256_stream_load_si256((__m256i*)in2);
            gh = _mm256_stream_load_si256((__m256i*)in3);
        }

        for (i = M_VEC; i < M; i++)
        {
            for (j = 0; j < N_VEC; j++)
            {
                O[j * M + i] = I[i * N + j];
            }
        }

        return;
    }

    if (N == 7)
    {
        __m256i mix1, mix2, mix3, mix4, mix5, mix6, mix7;
        __m256i v1, v2, v3, v4, v5, v6, v7, v8;
        const int blend1 = 129; 
        const int blend2 = 64;
        const int blend3 = 32;
        const int blend4 = 16;
        const int blend5 = 8;
        const int blend6 = 4;
        const int blend7 = 2;

        __m256i m1, m2, m3, m4, m5, m6, m7;

        m1 = _mm256_setr_epi8(0, 1, 14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3,
            0, 1, 14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3);
        m2 = _mm256_setr_epi8(2, 3, 0, 1, 14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5,
            2, 3, 0, 1, 14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5);
        m3 = _mm256_setr_epi8(4, 5, 2, 3, 0, 1, 14, 15, 12, 13, 10, 11, 8, 9, 6, 7,
            4, 5, 2, 3, 0, 1, 14, 15, 12, 13, 10, 11, 8, 9, 6, 7);
        m4 = _mm256_setr_epi8(6, 7, 4, 5, 2, 3, 0, 1, 14, 15, 12, 13, 10, 11, 8, 9,
            6, 7, 4, 5, 2, 3, 0, 1, 14, 15, 12, 13, 10, 11, 8, 9);
        m5 = _mm256_setr_epi8(8, 9, 6, 7, 4, 5, 2, 3, 0, 1, 14, 15, 12, 13, 10, 11,
            8, 9, 6, 7, 4, 5, 2, 3, 0, 1, 14, 15, 12, 13, 10, 11);
        m6 = _mm256_setr_epi8(10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1, 14, 15, 12, 13,
            10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1, 14, 15, 12, 13);
        m7 = _mm256_setr_epi8(12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1, 14, 15,
            12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1, 14, 15);

        M_VEC = M - M % VEC_16CAP;

        int16_t *in1 = in0 + VEC_16CAP,
            *in2 = in1 + VEC_16CAP,
            *in3 = in2 + VEC_16CAP,
            *in4 = in3 + VEC_16CAP,
            *in5 = in4 + VEC_16CAP,
            *in6 = in5 + VEC_16CAP;

        int16_t *out0 = O,
            *out1 = out0 + M,
            *out2 = out1 + M,
            *out3 = out2 + M,
            *out4 = out3 + M,
            *out5 = out4 + M,
            *out6 = out5 + M;

        mix1 = _mm256_stream_load_si256((__m256i*) in0);
        mix2 = _mm256_stream_load_si256((__m256i*) in1);
        mix3 = _mm256_stream_load_si256((__m256i*) in2);
        mix4 = _mm256_stream_load_si256((__m256i*) in3);
        mix5 = _mm256_stream_load_si256((__m256i*) in4);
        mix6 = _mm256_stream_load_si256((__m256i*) in5);
        mix7 = _mm256_stream_load_si256((__m256i*) in6);

        in_end = in0 + M_VEC * N;
        for (; in0 < in_end;)
        {
            in0 += N * VEC_16CAP;
            in1 += N * VEC_16CAP;
            in2 += N * VEC_16CAP;
            in3 += N * VEC_16CAP;
            in4 += N * VEC_16CAP;
            in5 += N * VEC_16CAP;
            in6 += N * VEC_16CAP;

            v1 = _mm256_permute2x128_si256(mix1, mix4, 48);
            v2 = _mm256_permute2x128_si256(mix1, mix5, 33);
            v3 = _mm256_permute2x128_si256(mix2, mix5, 48);
            v4 = _mm256_permute2x128_si256(mix2, mix6, 33);
            v5 = _mm256_permute2x128_si256(mix3, mix6, 48);
            v6 = _mm256_permute2x128_si256(mix3, mix7, 33);
            v7 = _mm256_permute2x128_si256(mix4, mix7, 48);

            mix1 = _mm256_blend_epi16(v1, v2, blend2);
            mix1 = _mm256_blend_epi16(mix1, v3, blend3);
            mix1 = _mm256_blend_epi16(mix1, v4, blend4);
            mix1 = _mm256_blend_epi16(mix1, v5, blend5);
            mix1 = _mm256_blend_epi16(mix1, v6, blend6);
            mix1 = _mm256_blend_epi16(mix1, v7, blend7);
            mix1 = _mm256_shuffle_epi8(mix1, m1);

            mix2 = _mm256_blend_epi16(v1, v2, blend1);
            mix2 = _mm256_blend_epi16(mix2, v3, blend2);
            mix2 = _mm256_blend_epi16(mix2, v4, blend3);
            mix2 = _mm256_blend_epi16(mix2, v5, blend4);
            mix2 = _mm256_blend_epi16(mix2, v6, blend5);
            mix2 = _mm256_blend_epi16(mix2, v7, blend6);
            mix2 = _mm256_shuffle_epi8(mix2, m2);

            mix3 = _mm256_blend_epi16(v1, v2, blend7);
            mix3 = _mm256_blend_epi16(mix3, v3, blend1);
            mix3 = _mm256_blend_epi16(mix3, v4, blend2);
            mix3 = _mm256_blend_epi16(mix3, v5, blend3);
            mix3 = _mm256_blend_epi16(mix3, v6, blend4);
            mix3 = _mm256_blend_epi16(mix3, v7, blend5);
            mix3 = _mm256_shuffle_epi8(mix3, m3);

            mix4 = _mm256_blend_epi16(v1, v2, blend6);
            mix4 = _mm256_blend_epi16(mix4, v3, blend7);
            mix4 = _mm256_blend_epi16(mix4, v4, blend1);
            mix4 = _mm256_blend_epi16(mix4, v5, blend2);
            mix4 = _mm256_blend_epi16(mix4, v6, blend3);
            mix4 = _mm256_blend_epi16(mix4, v7, blend4);
            mix4 = _mm256_shuffle_epi8(mix4, m4);

            mix5 = _mm256_blend_epi16(v1, v2, blend5);
            mix5 = _mm256_blend_epi16(mix5, v3, blend6);
            mix5 = _mm256_blend_epi16(mix5, v4, blend7);
            mix5 = _mm256_blend_epi16(mix5, v5, blend1);
            mix5 = _mm256_blend_epi16(mix5, v6, blend2);
            mix5 = _mm256_blend_epi16(mix5, v7, blend3);
            mix5 = _mm256_shuffle_epi8(mix5, m5);

            mix6 = _mm256_blend_epi16(v1, v2, blend4);
            mix6 = _mm256_blend_epi16(mix6, v3, blend5);
            mix6 = _mm256_blend_epi16(mix6, v4, blend6);
            mix6 = _mm256_blend_epi16(mix6, v5, blend7);
            mix6 = _mm256_blend_epi16(mix6, v6, blend1);
            mix6 = _mm256_blend_epi16(mix6, v7, blend2);
            mix6 = _mm256_shuffle_epi8(mix6, m6);

            mix7 = _mm256_blend_epi16(v1, v2, blend3);
            mix7 = _mm256_blend_epi16(mix7, v3, blend4);
            mix7 = _mm256_blend_epi16(mix7, v4, blend5);
            mix7 = _mm256_blend_epi16(mix7, v5, blend6);
            mix7 = _mm256_blend_epi16(mix7, v6, blend7);
            mix7 = _mm256_blend_epi16(mix7, v7, blend1);
            mix7 = _mm256_shuffle_epi8(mix7, m7);

            _mm256_storeu_si256((__m256i*) out0, mix1);
            _mm256_storeu_si256((__m256i*) out1, mix2);
            _mm256_storeu_si256((__m256i*) out2, mix3);
            _mm256_storeu_si256((__m256i*) out3, mix4);
            _mm256_storeu_si256((__m256i*) out4, mix5);
            _mm256_storeu_si256((__m256i*) out5, mix6);
            _mm256_storeu_si256((__m256i*) out6, mix7);

            out0 += VEC_16CAP;
            out1 += VEC_16CAP;
            out2 += VEC_16CAP;
            out3 += VEC_16CAP;
            out4 += VEC_16CAP;
            out5 += VEC_16CAP;
            out6 += VEC_16CAP;

            mix1 = _mm256_stream_load_si256((__m256i*) in0);
            mix2 = _mm256_stream_load_si256((__m256i*) in1);
            mix3 = _mm256_stream_load_si256((__m256i*) in2);
            mix4 = _mm256_stream_load_si256((__m256i*) in3);
            mix5 = _mm256_stream_load_si256((__m256i*) in4);
            mix6 = _mm256_stream_load_si256((__m256i*) in5);
            mix7 = _mm256_stream_load_si256((__m256i*) in6);
        }

        M_VEC = M - M % VEC_16CAP;
        for (i = M_VEC; i < M; i++)
        {
            out0 = O + i;
            for (j = 0; j < N; j++)
            {
                *out0 = *in0++;
                out0 += M;
            }
        }

        return;
    }

    if (N == 6)
    {
        M_VEC = M - M % SSE_16CAP;

        __m256i mix1, mix2, mix3, mix4, mix5, mix6, mix7;
        __m256i v1, v2, v3, v4, v5, v6, v7, v8;
        const int blend1 = 146;
        const int blend2 = 36;
        const int blend3 = 73;

        __m256i m1, m2, m3;

        m1 = _mm256_setr_epi8(0, 1, 12, 13, 8, 9, 4, 5, 2, 3, 14, 15, 10, 11, 6, 7,
            0, 1, 12, 13, 8, 9, 4, 5, 2, 3, 14, 15, 10, 11, 6, 7);
        m2 = _mm256_setr_epi8(4, 5, 0, 1, 12, 13, 8, 9, 6, 7, 2, 3, 14, 15, 10, 11,
            4, 5, 0, 1, 12, 13, 8, 9, 6, 7, 2, 3, 14, 15, 10, 11);
        m3 = _mm256_setr_epi8(8, 9, 4, 5, 0, 1, 12, 13, 10, 11, 6, 7, 2, 3, 14, 15,
            8, 9, 4, 5, 0, 1, 12, 13, 10, 11, 6, 7, 2, 3, 14, 15);

        int16_t *in1 = in0 + VEC_16CAP,
            *in2 = in1 + VEC_16CAP;

        int16_t *out0 = O,
            *out1 = out0 + M,
            *out2 = out1 + M,
            *out3 = out2 + M,
            *out4 = out3 + M,
            *out5 = out4 + M;

        mix1 = _mm256_stream_load_si256((__m256i*) in0);
        mix2 = _mm256_stream_load_si256((__m256i*) in1);
        mix3 = _mm256_stream_load_si256((__m256i*) in2);

        in_end = in0 + M_VEC * N;
        for (; in0 < in_end;)
        {
            in0 += N * SSE_16CAP;
            in1 += N * SSE_16CAP;
            in2 += N * SSE_16CAP;

            v1 = _mm256_permute2x128_si256(mix1, mix3, 32);
            v2 = mix2;
            v3 = _mm256_permute2x128_si256(mix1, mix3, 49);

            mix1 = _mm256_blend_epi32(v1, v2, blend1);
            mix1 = _mm256_blend_epi32(mix1, v3, blend2);
            mix1 = _mm256_shuffle_epi8(mix1, m1);
            mix1 = _mm256_permute4x64_epi64(mix1, 216);

            mix2 = _mm256_blend_epi32(v1, v2, blend2);
            mix2 = _mm256_blend_epi32(mix2, v3, blend3);
            mix2 = _mm256_shuffle_epi8(mix2, m2);
            mix2 = _mm256_permute4x64_epi64(mix2, 216);

            mix3 = _mm256_blend_epi32(v1, v2, blend3);
            mix3 = _mm256_blend_epi32(mix3, v3, blend1);
            mix3 = _mm256_shuffle_epi8(mix3, m3);
            mix3 = _mm256_permute4x64_epi64(mix3, 216);

            _mm_storeu_si128((__m128i*) out0, _mm256_castsi256_si128(mix1));
            _mm_storeu_si128((__m128i*) out1, _mm256_extracti128_si256(mix1, 1));
            _mm_storeu_si128((__m128i*) out2, _mm256_castsi256_si128(mix2));
            _mm_storeu_si128((__m128i*) out3, _mm256_extracti128_si256(mix2, 1));
            _mm_storeu_si128((__m128i*) out4, _mm256_castsi256_si128(mix3));
            _mm_storeu_si128((__m128i*) out5, _mm256_extracti128_si256(mix3, 1));

            out0 += SSE_16CAP;
            out1 += SSE_16CAP;
            out2 += SSE_16CAP;
            out3 += SSE_16CAP;
            out4 += SSE_16CAP;
            out5 += SSE_16CAP;

            mix1 = _mm256_stream_load_si256((__m256i*) in0);
            mix2 = _mm256_stream_load_si256((__m256i*) in1);
            mix3 = _mm256_stream_load_si256((__m256i*) in2);
        }

        for (i = M_VEC; i < M; i++)
        {
            out0 = O + i;
            for (j = 0; j < N; j++)
            {
                *out0 = *in0++;
                out0 += M;
            }
        }

        return;
    }

    if (N == 5)
    {
        M_VEC = M - M % 4;

        __m256i ad1234, ad5___;
        __m128i ad12, ad34, ad5;
        __m128i gather_n5 = _mm_setr_epi32(0, 5, 10, 15);
        __m256i shuffle8 = _mm256_setr_epi8(0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15,
            0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15);

        __m256i permute32 = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);

        const int32_t scale_32 = 2;
        const int32_t scale_64 = 2;

        int16_t *out0, *out1, *out2, *out3, *out4;

        ad1234 = _mm256_i32gather_epi64((int64_t*)in0, gather_n5, scale_64);
        ad5___ = _mm256_i32gather_epi64((int64_t*)(in0 + 4), gather_n5, scale_32);

        for (i = 0; i < M_VEC; i += 4)
        {
            in0 += N * 4;
            out0 = O + i,
                out1 = out0 + M,
                out2 = out1 + M,
                out3 = out2 + M,
                out4 = out3 + M;

            ad1234 = _mm256_shuffle_epi8(ad1234, shuffle8);
            ad1234 = _mm256_permutevar8x32_epi32(ad1234, permute32);

            ad5___ = _mm256_shuffle_epi8(ad5___, shuffle8);
            ad5___ = _mm256_permutevar8x32_epi32(ad5___, permute32);

            ad12 = _mm256_castsi256_si128(ad1234);
            ad34 = _mm256_extracti128_si256(ad1234, 1);
            ad5 = _mm256_castsi256_si128(ad5___);

            _mm_storeu_si64((long long*)out0, ad12);
            _mm_storeu_si64((long long*)out1, _mm_bsrli_si128(ad12, 8));
            _mm_storeu_si64((long long*)out2, ad34);
            _mm_storeu_si64((long long*)out3, _mm_bsrli_si128(ad34, 8));
            _mm_storeu_si64((long long*)out4, ad5);

            ad1234 = _mm256_i32gather_epi64((int64_t*)in0, gather_n5, scale_64);
            ad5___ = _mm256_i32gather_epi64((int64_t*)(in0 + 4), gather_n5, scale_32);
        }
        for (; i < M; i++)
        {
            out0 = O + i;
            for (j = 0; j < N; j++)
            {
                *out0 = *in0++;
                out0 += M;
            }
        }

        return;
    }

    if (N == 4)
    {
        M_VEC = M - M % SSE_16CAP;

        __m256i ad, eh, abcd, efgh;
        __m256i shuffle8 = _mm256_setr_epi8(0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15,
            0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15);
        __m256i permute32 = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);

        int16_t *in1 = in0 + VEC_16CAP;

        int16_t *out0, *out1, *out2, *out3, *out4, *out5, *out6, *out7;

        ad = _mm256_lddqu_si256((__m256i*)in0);
        eh = _mm256_lddqu_si256((__m256i*)in1);

        __m128i ad12, ad34, eh12, eh34;

        for (i = 0; i < M_VEC; i += 8)
        {
            in0 += 8 * N;
            in1 += 8 * N;

            ad = _mm256_shuffle_epi8(ad, shuffle8);
            eh = _mm256_shuffle_epi8(eh, shuffle8);

            abcd = _mm256_permutevar8x32_epi32(ad, permute32);
            efgh = _mm256_permutevar8x32_epi32(eh, permute32);

            ad12 = _mm256_castsi256_si128(abcd);
            ad34 = _mm256_extracti128_si256(abcd, 1);
            eh12 = _mm256_castsi256_si128(efgh);
            eh34 = _mm256_extracti128_si256(efgh, 1);

            out0 = O + i,
            out1 = out0 + 4,
            out2 = out0 + M,
            out3 = out2 + 4,
            out4 = out2 + M,
            out5 = out4 + 4;
            out6 = out4 + M;
            out7 = out6 + 4;

            _mm_storeu_si64(out0, ad12);
            _mm_storeu_si64(out1, eh12);
            _mm_storeu_si64(out2, _mm_bsrli_si128(ad12, 8));
            _mm_storeu_si64(out3, _mm_bsrli_si128(eh12, 8));
            _mm_storeu_si64(out4, ad34);
            _mm_storeu_si64(out5, eh34);
            _mm_storeu_si64(out6, _mm_bsrli_si128(ad34, 8));
            _mm_storeu_si64(out7, _mm_bsrli_si128(eh34, 8));

            ad = _mm256_lddqu_si256((__m256i*)in0);
            eh = _mm256_lddqu_si256((__m256i*)in1);
        }
        for (; i < M; i++)
        {
            out0 = O + i;
            for (j = 0; j < N; j++)
            {
                *out0 = *in0++;
                out0 += M;
            }
        }

        return;
    }

    if (N == 3)
    {
        M_VEC = M - M % SSE_16CAP;

        __m256i ah12, ah3_;
        __m128i ah1, ah2, ah3;
        __m256i shuffle8 = _mm256_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15,
            0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15);
        __m256i gather_n3 = _mm256_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21);
        const int32_t scale_32 = 2;

        int16_t *out0, *out1, *out2;

        ah12 = _mm256_i32gather_epi32((int32_t*)in0, gather_n3, scale_32);
        ah3_ = _mm256_i32gather_epi32((int32_t*)(in0 + 2), gather_n3, scale_32);

        for (i = 0; i < M_VEC; i += 8)
        {
            in0 += 8 * N;
            out0 = O + i,
                out1 = out0 + M,
                out2 = out1 + M;

            ah12 = _mm256_shuffle_epi8(ah12, shuffle8);
            ah12 = _mm256_permute4x64_epi64(ah12, 216);

            ah3_ = _mm256_shuffle_epi8(ah3_, shuffle8);
            ah3_ = _mm256_permute4x64_epi64(ah3_, 216);

            ah1 = _mm256_castsi256_si128(ah12);
            ah2 = _mm256_extracti128_si256(ah12, 1);
            ah3 = _mm256_castsi256_si128(ah3_);

            _mm_storeu_si128((__m128i*) out0, ah1);
            _mm_storeu_si128((__m128i*) out1, ah2);
            _mm_storeu_si128((__m128i*) out2, ah3);

            ah12 = _mm256_i32gather_epi32((int32_t*)in0, gather_n3, scale_32);
            ah3_ = _mm256_i32gather_epi32((int32_t*)(in0 + 2), gather_n3, scale_32);
        }
        for (; i < M; i++)
        {
            out0 = O + i;
            for (j = 0; j < N; j++)
            {
                *out0 = *in0++;
                out0 += M;
            }
        }

        return;
    }

    if (N == 2)
    {
        M_VEC = M - M % SSE_16CAP;

        __m128i ah1, ah2;
        __m256i ah;
        __m256i shuffle8 = _mm256_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15,
            0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15);

        int16_t *out0, *out1;

        ah = _mm256_load_si256((__m256i*)in0);

        for (i = 0; i < M_VEC; i += SSE_16CAP)
        {
            in0 += 16;

            ah = _mm256_shuffle_epi8(ah, shuffle8);
            ah = _mm256_permute4x64_epi64(ah, 216);

            ah1 = _mm256_castsi256_si128(ah);
            ah2 = _mm256_extracti128_si256(ah, 1);

            out0 = O + i,
                out1 = out0 + M;

            _mm_storeu_si128((__m128i*)out0, ah1);
            _mm_storeu_si128((__m128i*)out1, ah2);

            ah = _mm256_lddqu_si256((__m256i*)in0);
        }
        for (; i < M; i++)
        {
            out0 = O + i;
            for (j = 0; j < N; j++)
            {
                *out0 = *in0++;
                out0 += M;
            }
        }

        return;
    }
}
