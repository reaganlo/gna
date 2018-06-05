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
    if (M * N < VEC_16CAP * VEC_16CAP)
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

    uint32_t M_VEC = M - M % VEC_16CAP;
    uint32_t N_VEC = N - N % VEC_16CAP;

    int16_t *in0 = const_cast<int16_t*>(I);
    int16_t *in_end = in0 + M * N;

    // INTERLEAVE
    // MAX M is 8, MAX N is UINT16_MAX
    if (M == 2 && N >= VEC_16CAP)
    {
        in_end = in0 + N_VEC;
        int16_t *in1 = in0 + N;
        int16_t *out0 = O,
            *out1 = out0 + VEC_16CAP;

        __m128i a, b, ab_lo, ab_hi;

        a = _mm_lddqu_si128((__m128i*)in0);
        b = _mm_lddqu_si128((__m128i*)in1);

        for (; in0 < in_end;)
        {
            in0 += VEC_16CAP;
            in1 += VEC_16CAP;

            ab_lo = _mm_unpacklo_epi16(a, b);
            ab_hi = _mm_unpackhi_epi16(a, b);

            _mm_stream_si128((__m128i*) out0, ab_lo);
            _mm_stream_si128((__m128i*) out1, ab_hi);

            out0 += M * VEC_16CAP;
            out1 += M * VEC_16CAP;

            a = _mm_lddqu_si128((__m128i*)in0);
            b = _mm_lddqu_si128((__m128i*)in1);
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

    if (M == 3 && N >= VEC_16CAP)
    {
        in_end = in0 + N_VEC;
        int16_t *in1 = in0 + N,
            *in2 = in1 + N;
        int16_t *out0 = O,
            *out1 = out0 + VEC_16CAP,
            *out2 = out1 + VEC_16CAP;

        __m128i a, b, c, ab_lo, ab_hi, ab1, ab2, ab3, c1, c2, c3, ab2a, ab2b,
            mask1, mask2a, mask2b, mask3, cmask1, cmask2, cmask3, mix1, mix2, mix3;

        mask1 = _mm_setr_epi8(0, 1, 2, 3, 0, 1, 4, 5, 6, 7, 0, 1, 8, 9, 10, 11);
        mask2a = _mm_setr_epi8(0, 1, 12, 13, 14, 15, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1);
        mask2b = _mm_setr_epi8(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 0, 1, 4, 5);
        mask3 = _mm_setr_epi8(6, 7, 0, 1, 8, 9, 10, 11, 0, 1, 12, 13, 14, 15, 0, 1);
        cmask1 = _mm_setr_epi8(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 0, 1, 0, 1);
        cmask2 = _mm_setr_epi8(4, 5, 0, 1, 0, 1, 6, 7, 0, 1, 0, 1, 8, 9, 0, 1);
        cmask3 = _mm_setr_epi8(0, 1, 10, 11, 0, 1, 0, 1, 12, 13, 0, 1, 0, 1, 14, 15);

        const int blend1 = 36;
        const int blend2 = 73;
        const int blend3 = 146;

        a = _mm_lddqu_si128((__m128i*)in0);
        b = _mm_lddqu_si128((__m128i*)in1);
        c = _mm_lddqu_si128((__m128i*)in2);

        for (; in0 < in_end;)
        {
            in0 += VEC_16CAP;
            in1 += VEC_16CAP;
            in2 += VEC_16CAP;

            ab_lo = _mm_unpacklo_epi16(a, b);
            ab_hi = _mm_unpackhi_epi16(a, b);

            ab1 = _mm_shuffle_epi8(ab_lo, mask1);
            ab3 = _mm_shuffle_epi8(ab_hi, mask3);

            ab2a = _mm_shuffle_epi8(ab_lo, mask2a);
            ab2b = _mm_shuffle_epi8(ab_hi, mask2b);
            ab2 = _mm_blend_epi16(ab2a, ab2b, 240);

            c1 = _mm_shuffle_epi8(c, cmask1);
            c2 = _mm_shuffle_epi8(c, cmask2);
            c3 = _mm_shuffle_epi8(c, cmask3);

            mix1 = _mm_blend_epi16(ab1, c1, blend1);
            mix2 = _mm_blend_epi16(ab2, c2, blend2);
            mix3 = _mm_blend_epi16(ab3, c3, blend3);

            _mm_stream_si128((__m128i*) out0, mix1);
            _mm_stream_si128((__m128i*) out1, mix2);
            _mm_stream_si128((__m128i*) out2, mix3);

            out0 += M * VEC_16CAP;
            out1 += M * VEC_16CAP;
            out2 += M * VEC_16CAP;

            a = _mm_lddqu_si128((__m128i*)in0);
            b = _mm_lddqu_si128((__m128i*)in1);
            c = _mm_lddqu_si128((__m128i*)in2);
        }

        out1 = out0 + 1;
        out2 = out1 + 1;
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

    if (M == 4 && N >= VEC_16CAP)
    {
        in_end = in0 + N_VEC;
        int16_t *in1 = in0 + N,
            *in2 = in1 + N,
            *in3 = in2 + N;

        int16_t *out0 = O,
            *out1 = out0 + VEC_16CAP,
            *out2 = out1 + VEC_16CAP,
            *out3 = out2 + VEC_16CAP;

        __m128i a, b, c, d, ab_lo, ab_hi, cd_lo, cd_hi,
            abcd_lo, abcd_lohi, abcd_hi, abcd_hilo;

        a = _mm_lddqu_si128((__m128i*)in0);
        b = _mm_lddqu_si128((__m128i*)in1);
        c = _mm_lddqu_si128((__m128i*)in2);
        d = _mm_lddqu_si128((__m128i*)in3);

        for (; in0 < in_end;)
        {
            in0 += VEC_16CAP;
            in1 += VEC_16CAP;
            in2 += VEC_16CAP;
            in3 += VEC_16CAP;

            ab_lo = _mm_unpacklo_epi16(a, b);
            ab_hi = _mm_unpackhi_epi16(a, b);
            cd_lo = _mm_unpacklo_epi16(c, d);
            cd_hi = _mm_unpackhi_epi16(c, d);

            abcd_lo = _mm_unpacklo_epi32(ab_lo, cd_lo);
            abcd_lohi = _mm_unpackhi_epi32(ab_lo, cd_lo);
            abcd_hi = _mm_unpacklo_epi32(ab_hi, cd_hi);
            abcd_hilo = _mm_unpackhi_epi32(ab_hi, cd_hi);

            _mm_stream_si128((__m128i*) out0, abcd_lo);
            _mm_stream_si128((__m128i*) out1, abcd_lohi);
            _mm_stream_si128((__m128i*) out2, abcd_hi);
            _mm_stream_si128((__m128i*) out3, abcd_hilo);

            out0 += M * VEC_16CAP;
            out1 += M * VEC_16CAP;
            out2 += M * VEC_16CAP;
            out3 += M * VEC_16CAP;

            a = _mm_lddqu_si128((__m128i*)in0);
            b = _mm_lddqu_si128((__m128i*)in1);
            c = _mm_lddqu_si128((__m128i*)in2);
            d = _mm_lddqu_si128((__m128i*)in3);
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

    if (M == 5 && N >= VEC_16CAP)
    {
        in_end = in0 + N_VEC;
        int16_t *in1 = in0 + N,
            *in2 = in1 + N,
            *in3 = in2 + N,
            *in4 = in3 + N;

        int16_t *out0 = O;

        __m128i a, b, c, d, e, ab_lo, ab_hi, cd_lo, cd_hi,
            abcd_lo, abcd_lohi, abcd_hi, abcd_hilo;

        a = _mm_lddqu_si128((__m128i*)in0);
        b = _mm_lddqu_si128((__m128i*)in1);
        c = _mm_lddqu_si128((__m128i*)in2);
        d = _mm_lddqu_si128((__m128i*)in3);
        e = _mm_lddqu_si128((__m128i*)in4);

        for (; in0 < in_end;)
        {
            in0 += VEC_16CAP;
            in1 += VEC_16CAP;
            in2 += VEC_16CAP;
            in3 += VEC_16CAP;
            in4 += VEC_16CAP;

            ab_lo = _mm_unpacklo_epi16(a, b);
            ab_hi = _mm_unpackhi_epi16(a, b);
            cd_lo = _mm_unpacklo_epi16(c, d);
            cd_hi = _mm_unpackhi_epi16(c, d);

            abcd_lo = _mm_unpacklo_epi32(ab_lo, cd_lo);
            abcd_lohi = _mm_unpackhi_epi32(ab_lo, cd_lo);
            abcd_hi = _mm_unpacklo_epi32(ab_hi, cd_hi);
            abcd_hilo = _mm_unpackhi_epi32(ab_hi, cd_hi);

            _mm_storel_epi64((__m128i*)out0, abcd_lo);
            *(out0 + 4) = _mm_extract_epi16(e, 0);

            _mm_storel_epi64((__m128i*)(out0 + 5), _mm_srli_si128(abcd_lo, 8));
            *(out0 + 9) = _mm_extract_epi16(e, 1);

            _mm_storel_epi64((__m128i*)(out0 + 10), abcd_lohi);
            *(out0 + 14) = _mm_extract_epi16(e, 2);

            _mm_storel_epi64((__m128i*)(out0 + 15), _mm_srli_si128(abcd_lohi, 8));
            *(out0 + 19) = _mm_extract_epi16(e, 3);

            _mm_storel_epi64((__m128i*)(out0 + 20), abcd_hi);
            *(out0 + 24) = _mm_extract_epi16(e, 4);

            _mm_storel_epi64((__m128i*)(out0 + 25), _mm_srli_si128(abcd_hi, 8));
            *(out0 + 29) = _mm_extract_epi16(e, 5);

            _mm_storel_epi64((__m128i*)(out0 + 30), abcd_hilo);
            *(out0 + 34) = _mm_extract_epi16(e, 6);

            _mm_storel_epi64((__m128i*)(out0 + 35), _mm_srli_si128(abcd_hilo, 8));
            *(out0 + 39) = _mm_extract_epi16(e, 7);

            out0 += 40;

            a = _mm_lddqu_si128((__m128i*)in0);
            b = _mm_lddqu_si128((__m128i*)in1);
            c = _mm_lddqu_si128((__m128i*)in2);
            d = _mm_lddqu_si128((__m128i*)in3);
            e = _mm_lddqu_si128((__m128i*)in4);
        }

        int16_t *out1 = out0 + 1;
        int16_t *out2 = out1 + 1;
        int16_t *out3 = out2 + 1;
        int16_t *out4 = out3 + 1;
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

    if (M == 6 && N >= VEC_16CAP)
    {
        in_end = in0 + N_VEC;
        int16_t *in1 = in0 + N,
            *in2 = in1 + N,
            *in3 = in2 + N,
            *in4 = in3 + N,
            *in5 = in4 + N;

        int16_t *out0 = O;

        __m128i a, b, c, d, e, f, ab_lo, ab_hi, cd_lo, cd_hi, ef_lo, ef_hi,
            abcd_lo, abcd_lohi, abcd_hi, abcd_hilo;

        a = _mm_lddqu_si128((__m128i*)in0);
        b = _mm_lddqu_si128((__m128i*)in1);
        c = _mm_lddqu_si128((__m128i*)in2);
        d = _mm_lddqu_si128((__m128i*)in3);
        e = _mm_lddqu_si128((__m128i*)in4);
        f = _mm_lddqu_si128((__m128i*)in5);

        for (; in0 < in_end;)
        {
            in0 += VEC_16CAP;
            in1 += VEC_16CAP;
            in2 += VEC_16CAP;
            in3 += VEC_16CAP;
            in4 += VEC_16CAP;
            in5 += VEC_16CAP;

            ab_lo = _mm_unpacklo_epi16(a, b);
            ab_hi = _mm_unpackhi_epi16(a, b);
            cd_lo = _mm_unpacklo_epi16(c, d);
            cd_hi = _mm_unpackhi_epi16(c, d);
            ef_lo = _mm_unpacklo_epi16(e, f);
            ef_hi = _mm_unpackhi_epi16(e, f);

            abcd_lo = _mm_unpacklo_epi32(ab_lo, cd_lo);
            abcd_lohi = _mm_unpackhi_epi32(ab_lo, cd_lo);
            abcd_hi = _mm_unpacklo_epi32(ab_hi, cd_hi);
            abcd_hilo = _mm_unpackhi_epi32(ab_hi, cd_hi);

            _mm_storel_epi64((__m128i*)out0, abcd_lo);
            *(int32_t*)(out0 + 4) = _mm_extract_epi32(ef_lo, 0);

            _mm_storel_epi64((__m128i*)(out0 + 6), _mm_srli_si128(abcd_lo, 8));
            *(int32_t*)(out0 + 10) = _mm_extract_epi32(ef_lo, 1);

            _mm_storel_epi64((__m128i*)(out0 + 12), abcd_lohi);
            *(int32_t*)(out0 + 16) = _mm_extract_epi32(ef_lo, 2);

            _mm_storel_epi64((__m128i*)(out0 + 18), _mm_srli_si128(abcd_lohi, 8));
            *(int32_t*)(out0 + 22) = _mm_extract_epi32(ef_lo, 3);

            _mm_storel_epi64((__m128i*)(out0 + 24), abcd_hi);
            *(int32_t*)(out0 + 28) = _mm_extract_epi32(ef_hi, 0);

            _mm_storel_epi64((__m128i*)(out0 + 30), _mm_srli_si128(abcd_hi, 8));
            *(int32_t*)(out0 + 34) = _mm_extract_epi32(ef_hi, 1);

            _mm_storel_epi64((__m128i*)(out0 + 36), abcd_hilo);
            *(int32_t*)(out0 + 40) = _mm_extract_epi32(ef_hi, 2);

            _mm_storel_epi64((__m128i*)(out0 + 42), _mm_srli_si128(abcd_hilo, 8));
            *(int32_t*)(out0 + 46) = _mm_extract_epi32(ef_hi, 3);

            out0 += 48;

            a = _mm_lddqu_si128((__m128i*)in0);
            b = _mm_lddqu_si128((__m128i*)in1);
            c = _mm_lddqu_si128((__m128i*)in2);
            d = _mm_lddqu_si128((__m128i*)in3);
            e = _mm_lddqu_si128((__m128i*)in4);
            f = _mm_lddqu_si128((__m128i*)in5);
        }

        int16_t *out1 = out0 + 1;
        int16_t *out2 = out1 + 1;
        int16_t *out3 = out2 + 1;
        int16_t *out4 = out3 + 1;
        int16_t *out5 = out4 + 1;
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

    if (M == 7 && N >= VEC_16CAP)
    {
        in_end = in0 + N_VEC;
        int16_t *in1 = in0 + N,
            *in2 = in1 + N,
            *in3 = in2 + N,
            *in4 = in3 + N,
            *in5 = in4 + N,
            *in6 = in5 + N;

        int16_t *out0 = O;

        __m128i a, b, c, d, e, f, g, ab_lo, ab_hi, cd_lo, cd_hi, ef_lo, ef_hi,
            abcd_lo, abcd_lohi, abcd_hi, abcd_hilo;

        a = _mm_lddqu_si128((__m128i*)in0);
        b = _mm_lddqu_si128((__m128i*)in1);
        c = _mm_lddqu_si128((__m128i*)in2);
        d = _mm_lddqu_si128((__m128i*)in3);
        e = _mm_lddqu_si128((__m128i*)in4);
        f = _mm_lddqu_si128((__m128i*)in5);
        g = _mm_lddqu_si128((__m128i*)in6);

        for (; in0 < in_end;)
        {
            in0 += VEC_16CAP;
            in1 += VEC_16CAP;
            in2 += VEC_16CAP;
            in3 += VEC_16CAP;
            in4 += VEC_16CAP;
            in5 += VEC_16CAP;
            in6 += VEC_16CAP;

            ab_lo = _mm_unpacklo_epi16(a, b);
            ab_hi = _mm_unpackhi_epi16(a, b);
            cd_lo = _mm_unpacklo_epi16(c, d);
            cd_hi = _mm_unpackhi_epi16(c, d);
            ef_lo = _mm_unpacklo_epi16(e, f);
            ef_hi = _mm_unpackhi_epi16(e, f);

            abcd_lo = _mm_unpacklo_epi32(ab_lo, cd_lo);
            abcd_lohi = _mm_unpackhi_epi32(ab_lo, cd_lo);
            abcd_hi = _mm_unpacklo_epi32(ab_hi, cd_hi);
            abcd_hilo = _mm_unpackhi_epi32(ab_hi, cd_hi);

            _mm_storel_epi64((__m128i*)out0, abcd_lo);
            *(int32_t*)(out0 + 4) = _mm_extract_epi32(ef_lo, 0);
            *(out0 + 6) = _mm_extract_epi16(g, 0);

            _mm_storel_epi64((__m128i*)(out0 + 7), _mm_srli_si128(abcd_lo, 8));
            *(int32_t*)(out0 + 11) = _mm_extract_epi32(ef_lo, 1);
            *(out0 + 13) = _mm_extract_epi16(g, 1);

            _mm_storel_epi64((__m128i*)(out0 + 14), abcd_lohi);
            *(int32_t*)(out0 + 18) = _mm_extract_epi32(ef_lo, 2);
            *(out0 + 20) = _mm_extract_epi16(g, 2);

            _mm_storel_epi64((__m128i*)(out0 + 21), _mm_srli_si128(abcd_lohi, 8));
            *(int32_t*)(out0 + 25) = _mm_extract_epi32(ef_lo, 3);
            *(out0 + 27) = _mm_extract_epi16(g, 3);

            _mm_storel_epi64((__m128i*)(out0 + 28), abcd_hi);
            *(int32_t*)(out0 + 32) = _mm_extract_epi32(ef_hi, 0);
            *(out0 + 34) = _mm_extract_epi16(g, 4);

            _mm_storel_epi64((__m128i*)(out0 + 35), _mm_srli_si128(abcd_hi, 8));
            *(int32_t*)(out0 + 39) = _mm_extract_epi32(ef_hi, 1);
            *(out0 + 41) = _mm_extract_epi16(g, 5);

            _mm_storel_epi64((__m128i*)(out0 + 42), abcd_hilo);
            *(int32_t*)(out0 + 46) = _mm_extract_epi32(ef_hi, 2);
            *(out0 + 48) = _mm_extract_epi16(g, 6);

            _mm_storel_epi64((__m128i*)(out0 + 49), _mm_srli_si128(abcd_hilo, 8));
            *(int32_t*)(out0 + 53) = _mm_extract_epi32(ef_hi, 3);
            *(out0 + 55) = _mm_extract_epi16(g, 7);

            out0 += 56;

            a = _mm_lddqu_si128((__m128i*)in0);
            b = _mm_lddqu_si128((__m128i*)in1);
            c = _mm_lddqu_si128((__m128i*)in2);
            d = _mm_lddqu_si128((__m128i*)in3);
            e = _mm_lddqu_si128((__m128i*)in4);
            f = _mm_lddqu_si128((__m128i*)in5);
            g = _mm_lddqu_si128((__m128i*)in6);
        }

        int16_t *out1 = out0 + 1;
        int16_t *out2 = out1 + 1;
        int16_t *out3 = out2 + 1;
        int16_t *out4 = out3 + 1;
        int16_t *out5 = out4 + 1;
        int16_t *out6 = out5 + 1;
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

    if (M == 8 && N >= VEC_16CAP)
    {
        in_end = in0 + N_VEC;
        int16_t *in1 = in0 + N,
            *in2 = in1 + N,
            *in3 = in2 + N,
            *in4 = in3 + N,
            *in5 = in4 + N,
            *in6 = in5 + N,
            *in7 = in6 + N;

        int16_t *out0 = O,
            *out1 = out0 + VEC_16CAP,
            *out2 = out1 + VEC_16CAP,
            *out3 = out2 + VEC_16CAP,
            *out4 = out3 + VEC_16CAP,
            *out5 = out4 + VEC_16CAP,
            *out6 = out5 + VEC_16CAP,
            *out7 = out6 + VEC_16CAP;

        __m128i a, b, c, d, e, f, g, h,
            ab_lo, ab_hi, cd_lo, cd_hi, ef_lo, ef_hi, gh_lo, gh_hi,
            abcd_lo, abcd_lohi, abcd_hi, abcd_hilo,
            efgh_lo, efgh_lohi, efgh_hi, efgh_hilo,
            pack1, pack2, pack3, pack4, pack5, pack6, pack7, pack8;

        a = _mm_lddqu_si128((__m128i*)in0);
        b = _mm_lddqu_si128((__m128i*)in1);
        c = _mm_lddqu_si128((__m128i*)in2);
        d = _mm_lddqu_si128((__m128i*)in3);
        e = _mm_lddqu_si128((__m128i*)in4);
        f = _mm_lddqu_si128((__m128i*)in5);
        g = _mm_lddqu_si128((__m128i*)in6);
        h = _mm_lddqu_si128((__m128i*)in7);

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

            ab_lo = _mm_unpacklo_epi16(a, b);
            ab_hi = _mm_unpackhi_epi16(a, b);
            cd_lo = _mm_unpacklo_epi16(c, d);
            cd_hi = _mm_unpackhi_epi16(c, d);
            ef_lo = _mm_unpacklo_epi16(e, f);
            ef_hi = _mm_unpackhi_epi16(e, f);
            gh_lo = _mm_unpacklo_epi16(g, h);
            gh_hi = _mm_unpackhi_epi16(g, h);

            abcd_lo = _mm_unpacklo_epi32(ab_lo, cd_lo);
            abcd_lohi = _mm_unpackhi_epi32(ab_lo, cd_lo);
            abcd_hi = _mm_unpacklo_epi32(ab_hi, cd_hi);
            abcd_hilo = _mm_unpackhi_epi32(ab_hi, cd_hi);

            efgh_lo = _mm_unpacklo_epi32(ef_lo, gh_lo);
            efgh_lohi = _mm_unpackhi_epi32(ef_lo, gh_lo);
            efgh_hi = _mm_unpacklo_epi32(ef_hi, gh_hi);
            efgh_hilo = _mm_unpackhi_epi32(ef_hi, gh_hi);

            pack1 = _mm_unpacklo_epi64(abcd_lo, efgh_lo);
            pack2 = _mm_unpackhi_epi64(abcd_lo, efgh_lo);
            pack3 = _mm_unpacklo_epi64(abcd_lohi, efgh_lohi);
            pack4 = _mm_unpackhi_epi64(abcd_lohi, efgh_lohi);

            pack5 = _mm_unpacklo_epi64(abcd_hi, efgh_hi);
            pack6 = _mm_unpackhi_epi64(abcd_hi, efgh_hi);
            pack7 = _mm_unpacklo_epi64(abcd_hilo, efgh_hilo);
            pack8 = _mm_unpackhi_epi64(abcd_hilo, efgh_hilo);

            _mm_stream_si128((__m128i*) out0, pack1);
            _mm_stream_si128((__m128i*) out1, pack2);
            _mm_stream_si128((__m128i*) out2, pack3);
            _mm_stream_si128((__m128i*) out3, pack4);
            _mm_stream_si128((__m128i*) out4, pack5);
            _mm_stream_si128((__m128i*) out5, pack6);
            _mm_stream_si128((__m128i*) out6, pack7);
            _mm_stream_si128((__m128i*) out7, pack8);

            out0 += M * VEC_16CAP;
            out1 += M * VEC_16CAP;
            out2 += M * VEC_16CAP;
            out3 += M * VEC_16CAP;
            out4 += M * VEC_16CAP;
            out5 += M * VEC_16CAP;
            out6 += M * VEC_16CAP;
            out7 += M * VEC_16CAP;

            a = _mm_lddqu_si128((__m128i*)in0);
            b = _mm_lddqu_si128((__m128i*)in1);
            c = _mm_lddqu_si128((__m128i*)in2);
            d = _mm_lddqu_si128((__m128i*)in3);
            e = _mm_lddqu_si128((__m128i*)in4);
            f = _mm_lddqu_si128((__m128i*)in5);
            g = _mm_lddqu_si128((__m128i*)in6);
            h = _mm_lddqu_si128((__m128i*)in7);
        }

        out1 = out0 + 1;
        out2 = out1 + 1;
        out3 = out2 + 1;
        out4 = out3 + 1;
        out5 = out4 + 1;
        out6 = out5 + 1;
        out7 = out6 + 1;
        for (i = N_VEC; i < N; i++)
        {
            *out0 = *in0++;
            *out1 = *in1++;
            *out2 = *in2++;
            *out3 = *in3++;
            *out4 = *in4++;
            *out5 = *in5++;
            *out6 = *in6++;
            *out7 = *in7++;

            out0 += M;
            out1 += M;
            out2 += M;
            out3 += M;
            out4 += M;
            out5 += M;
            out6 += M;
            out7 += M;
        }

        return;
    }

    if (N == 2 && M >= 8)
    {
        int16_t *in1 = in0 + VEC_16CAP;
        uint32_t end = M - M % VEC_16CAP;
        int16_t *out0 = O;
        int16_t *out1 = out0 + M;

        __m128i ad, eh, ah1, ah2, mask;

        ad = _mm_lddqu_si128((__m128i*)in0);
        eh = _mm_lddqu_si128((__m128i*)in1);
        mask = _mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15);

        for (i = 0; i < end; i += VEC_16CAP)
        {
            in0 += N * VEC_16CAP;
            in1 += N * VEC_16CAP;

            ad = _mm_shuffle_epi8(ad, mask);
            eh = _mm_shuffle_epi8(eh, mask);

            ah1 = _mm_unpacklo_epi64(ad, eh);
            ah2 = _mm_unpackhi_epi64(ad, eh);

            _mm_storeu_si128((__m128i*) out0, ah1);
            _mm_storeu_si128((__m128i*) out1, ah2);

            out0 += VEC_16CAP;
            out1 += VEC_16CAP;

            ad = _mm_lddqu_si128((__m128i*)in0);
            eh = _mm_lddqu_si128((__m128i*)in1);
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
        int16_t *in1 = in0 + VEC_16CAP,
            *in2 = in1 + VEC_16CAP;

        int16_t *out0 = O,
            *out1 = out0 + M,
            *out2 = out1 + M;

        __m128i mix1, mix2, mix3, t1, t2, t3, mask1, mask2, mask3;
        mask1 = _mm_setr_epi8(0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15, 4, 5, 10, 11);
        mask2 = _mm_setr_epi8(2, 3, 8, 9, 14, 15, 4, 5, 10, 11, 0, 1, 6, 7, 12, 13);
        mask3 = _mm_setr_epi8(4, 5, 10, 11, 0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15);

        mix1 = _mm_lddqu_si128((__m128i*)in0);
        mix2 = _mm_lddqu_si128((__m128i*)in1);
        mix3 = _mm_lddqu_si128((__m128i*)in2);

        for (i = 0; i < M_VEC; i += VEC_16CAP)
        {
            in0 += N * VEC_16CAP;
            in1 += N * VEC_16CAP;
            in2 += N * VEC_16CAP;

            t1 = _mm_blend_epi16(mix1, mix2, 146);
            t1 = _mm_blend_epi16(t1, mix3, 36);
            t1 = _mm_shuffle_epi8(t1, mask1);

            t2 = _mm_blend_epi16(mix1, mix2, 36);
            t2 = _mm_blend_epi16(t2, mix3, 73);
            t2 = _mm_shuffle_epi8(t2, mask2);

            t3 = _mm_blend_epi16(mix1, mix2, 73);
            t3 = _mm_blend_epi16(t3, mix3, 146);
            t3 = _mm_shuffle_epi8(t3, mask3);

            _mm_storeu_si128((__m128i*) out0, t1);
            _mm_storeu_si128((__m128i*) out1, t2);
            _mm_storeu_si128((__m128i*) out2, t3);

            out0 += VEC_16CAP;
            out1 += VEC_16CAP;
            out2 += VEC_16CAP;

            mix1 = _mm_lddqu_si128((__m128i*)in0);
            mix2 = _mm_lddqu_si128((__m128i*)in1);
            mix3 = _mm_lddqu_si128((__m128i*)in2);
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

    if (N == 4 && M >= 8)
    {
        uint32_t end = M - M % VEC_16CAP;
        int16_t *in1 = in0 + VEC_16CAP,
            *in2 = in1 + VEC_16CAP,
            *in3 = in2 + VEC_16CAP;

        int16_t *out0 = O,
            *out1 = out0 + M,
            *out2 = out1 + M,
            *out3 = out2 + M;

        __m128i ab, cd, ef, gh, ac, bd, eg, fh;
        __m128i abcd_lo, abcd_hi, efgh_lo, efgh_hi;
        __m128i mask = _mm_setr_epi8(0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15);
        __m128i full1, full2, full3, full4;

        ab = _mm_lddqu_si128((__m128i*)in0);
        cd = _mm_lddqu_si128((__m128i*)in1);
        ef = _mm_lddqu_si128((__m128i*)in2);
        gh = _mm_lddqu_si128((__m128i*)in3);

        for (i = 0; i < end; i += VEC_16CAP)
        {
            in0 += N * VEC_16CAP;
            in1 += N * VEC_16CAP;
            in2 += N * VEC_16CAP;
            in3 += N * VEC_16CAP;

            ac = _mm_unpacklo_epi16(ab, cd);
            bd = _mm_unpackhi_epi16(ab, cd);
            eg = _mm_unpacklo_epi16(ef, gh);
            fh = _mm_unpackhi_epi16(ef, gh);

            abcd_lo = _mm_unpacklo_epi32(ac, bd);
            abcd_hi = _mm_unpackhi_epi32(ac, bd);

            efgh_lo = _mm_unpacklo_epi32(eg, fh);
            efgh_hi = _mm_unpackhi_epi32(eg, fh);

            full1 = _mm_unpacklo_epi64(abcd_lo, efgh_lo);
            full2 = _mm_unpackhi_epi64(abcd_lo, efgh_lo);
            full3 = _mm_unpacklo_epi64(abcd_hi, efgh_hi);
            full4 = _mm_unpackhi_epi64(abcd_hi, efgh_hi);

            full1 = _mm_shuffle_epi8(full1, mask);
            full2 = _mm_shuffle_epi8(full2, mask);
            full3 = _mm_shuffle_epi8(full3, mask);
            full4 = _mm_shuffle_epi8(full4, mask);

            _mm_storeu_si128((__m128i*) out0, full1);
            _mm_storeu_si128((__m128i*) out1, full2);
            _mm_storeu_si128((__m128i*) out2, full3);
            _mm_storeu_si128((__m128i*) out3, full4);

            out0 += VEC_16CAP;
            out1 += VEC_16CAP;
            out2 += VEC_16CAP;
            out3 += VEC_16CAP;

            ab = _mm_lddqu_si128((__m128i*)in0);
            cd = _mm_lddqu_si128((__m128i*)in1);
            ef = _mm_lddqu_si128((__m128i*)in2);
            gh = _mm_lddqu_si128((__m128i*)in3);
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

    if (N == 5)
    {
        uint32_t end = M - M % SSE_16CAP;
        int16_t *in1 = in0 + SSE_16CAP,
            *in2 = in1 + SSE_16CAP,
            *in3 = in2 + SSE_16CAP,
            *in4 = in3 + SSE_16CAP;

        int16_t *out0 = O,
            *out1 = out0 + M,
            *out2 = out1 + M,
            *out3 = out2 + M,
            *out4 = out3 + M;

        __m128i a, b, c, d, e, f, g, h;
        __m128i v1, v2, v3, v4, v5;
        __m128i mask1, mask2, mask3, mask4, mask5;
        mask1 = _mm_setr_epi8(0, 1, 10, 11, 4, 5, 14, 15, 8, 9, 2, 3, 12, 13, 6, 7);
        mask2 = _mm_setr_epi8(2, 3, 12, 13, 6, 7, 0, 1, 10, 11, 4, 5, 14, 15, 8, 9);
        mask3 = _mm_setr_epi8(4, 5, 14, 15, 8, 9, 2, 3, 12, 13, 6, 7, 0, 1, 10, 11);
        mask4 = _mm_setr_epi8(6, 7, 0, 1, 10, 11, 4, 5, 14, 15, 8, 9, 2, 3, 12, 13);
        mask5 = _mm_setr_epi8(8, 9, 2, 3, 12, 13, 6, 7, 0, 1, 10, 11, 4, 5, 14, 15);
        const int blend1 = 132,
            blend2 = 16,
            blend3 = 66,
            blend4 = 8,
            blend5 = 33;

        a = _mm_load_si128((__m128i*)in0);
        b = _mm_load_si128((__m128i*)in1);
        c = _mm_load_si128((__m128i*)in2);
        d = _mm_load_si128((__m128i*)in3);
        e = _mm_load_si128((__m128i*)in4);

        for (i = 0; i < end; i += SSE_16CAP)
        {
            in0 += N * SSE_16CAP;
            in1 += N * SSE_16CAP;
            in2 += N * SSE_16CAP;
            in3 += N * SSE_16CAP;
            in4 += N * SSE_16CAP;

            v1 = _mm_blend_epi16(a, b, blend1);
            v1 = _mm_blend_epi16(v1, c, blend2);
            v1 = _mm_blend_epi16(v1, d, blend3);
            v1 = _mm_blend_epi16(v1, e, blend4);
            v1 = _mm_shuffle_epi8(v1, mask1);

            v2 = _mm_blend_epi16(a, b, blend4);
            v2 = _mm_blend_epi16(v2, c, blend5);
            v2 = _mm_blend_epi16(v2, d, blend1);
            v2 = _mm_blend_epi16(v2, e, blend2);
            v2 = _mm_shuffle_epi8(v2, mask2);

            v3 = _mm_blend_epi16(a, b, blend2);
            v3 = _mm_blend_epi16(v3, c, blend3);
            v3 = _mm_blend_epi16(v3, d, blend4);
            v3 = _mm_blend_epi16(v3, e, blend5);
            v3 = _mm_shuffle_epi8(v3, mask3);

            v4 = _mm_blend_epi16(a, b, blend5);
            v4 = _mm_blend_epi16(v4, c, blend1);
            v4 = _mm_blend_epi16(v4, d, blend2);
            v4 = _mm_blend_epi16(v4, e, blend3);
            v4 = _mm_shuffle_epi8(v4, mask4);

            v5 = _mm_blend_epi16(a, b, blend3);
            v5 = _mm_blend_epi16(v5, c, blend4);
            v5 = _mm_blend_epi16(v5, d, blend5);
            v5 = _mm_blend_epi16(v5, e, blend1);
            v5 = _mm_shuffle_epi8(v5, mask5);

            _mm_storeu_si128((__m128i*) out0, v1);
            _mm_storeu_si128((__m128i*) out1, v2);
            _mm_storeu_si128((__m128i*) out2, v3);
            _mm_storeu_si128((__m128i*) out3, v4);
            _mm_storeu_si128((__m128i*) out4, v5);

            out0 += SSE_16CAP;
            out1 += SSE_16CAP;
            out2 += SSE_16CAP;
            out3 += SSE_16CAP;
            out4 += SSE_16CAP;

            a = _mm_lddqu_si128((__m128i*)in0);
            b = _mm_lddqu_si128((__m128i*)in1);
            c = _mm_lddqu_si128((__m128i*)in2);
            d = _mm_lddqu_si128((__m128i*)in3);
            e = _mm_lddqu_si128((__m128i*)in4);
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

    if (N == 6)
    {
        uint32_t end = M - M % 4;
        int16_t *in1 = in0 + VEC_16CAP,
                *in2 = in1 + VEC_16CAP;

        int16_t *out0 = O,
            *out1 = out0 + M,
            *out2 = out1 + M,
            *out3 = out2 + M,
            *out4 = out3 + M,
            *out5 = out4 + M;

        __m128i ab, bc, cd;
        __m128i mix1, mix2, mix3;
        __m128i mask1, mask2, mask3;

        ab = _mm_lddqu_si128((__m128i*)in0);
        bc = _mm_lddqu_si128((__m128i*)in1);
        cd = _mm_lddqu_si128((__m128i*)in2);

        mask1 = _mm_setr_epi8(0, 1, 12, 13, 8, 9, 4, 5, 2, 3, 14, 15, 10, 11, 6, 7);
        mask2 = _mm_setr_epi8(4, 5, 0, 1, 12, 13, 8, 9, 6, 7, 2, 3, 14, 15, 10, 11);
        mask3 = _mm_setr_epi8(8, 9, 4, 5, 0, 1, 12, 13, 10, 11, 6, 7, 2, 3, 14, 15);

        in_end = in0 + end * N;
        for (; in0 < in_end;)
        {
            in0 += N * 4;
            in1 += N * 4;
            in2 += N * 4;

            mix1 = _mm_blend_epi16(ab, bc, 48);
            mix1 = _mm_blend_epi16(mix1, cd, 12);
            mix1 = _mm_shuffle_epi8(mix1, mask1);

            mix2 = _mm_blend_epi16(ab, bc, 195);
            mix2 = _mm_blend_epi16(mix2, cd, 48);
            mix2 = _mm_shuffle_epi8(mix2, mask2);

            mix3 = _mm_blend_epi16(ab, bc, 12);
            mix3 = _mm_blend_epi16(mix3, cd, 195);
            mix3 = _mm_shuffle_epi8(mix3, mask3);

            _mm_storel_epi64((__m128i*)out0, mix1);
            _mm_storel_epi64((__m128i*)out1, _mm_srli_si128(mix1, 8));
            _mm_storel_epi64((__m128i*)out2, mix2);
            _mm_storel_epi64((__m128i*)out3, _mm_srli_si128(mix2, 8));
            _mm_storel_epi64((__m128i*)out4, mix3);
            _mm_storel_epi64((__m128i*)out5, _mm_srli_si128(mix3, 8));

            out0 += 4;
            out1 += 4;
            out2 += 4;
            out3 += 4;
            out4 += 4;
            out5 += 4;

            ab = _mm_lddqu_si128((__m128i*)in0);
            bc = _mm_lddqu_si128((__m128i*)in1);
            cd = _mm_lddqu_si128((__m128i*)in2);
        }

        for (i = end; i < M; i++)
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

    if (N == 7)
    {
        uint32_t end = M - M % VEC_16CAP;
        int16_t *in1 = in0 + N,
            *in2 = in1 + N,
            *in3 = in2 + N,
            *in4 = in3 + N,
            *in5 = in4 + N,
            *in6 = in5 + N,
            *in7 = in6 + N;

        int16_t *out0 = O,
            *out1 = out0 + M,
            *out2 = out1 + M,
            *out3 = out2 + M,
            *out4 = out3 + M,
            *out5 = out4 + M,
            *out6 = out5 + M;

        __m128i a, b, c, d, e, f, g, h;
        __m128i ab_lo, ab_hi, cd_lo, cd_hi, ef_lo, ef_hi, gh_lo, gh_hi;
        __m128i abcd_lo, abcd_hi, abcd_lohi, abcd_hilo, efgh_lo, efgh_lohi, efgh_hi, efgh_hilo;
        __m128i full1, full2, full3, full4, full5, full6, full7, full8;

        a = _mm_lddqu_si128((__m128i*)in0);
        b = _mm_lddqu_si128((__m128i*)in1);
        c = _mm_lddqu_si128((__m128i*)in2);
        d = _mm_lddqu_si128((__m128i*)in3);
        e = _mm_lddqu_si128((__m128i*)in4);
        f = _mm_lddqu_si128((__m128i*)in5);
        g = _mm_lddqu_si128((__m128i*)in6);
        h = _mm_lddqu_si128((__m128i*)in7);

        in_end = in0 + end * N;
        for (; in0 < in_end;)
        {
            in0 += N * VEC_16CAP;
            in1 += N * VEC_16CAP;
            in2 += N * VEC_16CAP;
            in3 += N * VEC_16CAP;
            in4 += N * VEC_16CAP;
            in5 += N * VEC_16CAP;
            in6 += N * VEC_16CAP;
            in7 += N * VEC_16CAP;

            ab_lo = _mm_unpacklo_epi16(a, b);
            ab_hi = _mm_unpackhi_epi16(a, b);
            cd_lo = _mm_unpacklo_epi16(c, d);
            cd_hi = _mm_unpackhi_epi16(c, d);
            ef_lo = _mm_unpacklo_epi16(e, f);
            ef_hi = _mm_unpackhi_epi16(e, f);
            gh_lo = _mm_unpacklo_epi16(g, h);
            gh_hi = _mm_unpackhi_epi16(g, h);

            abcd_lo = _mm_unpacklo_epi32(ab_lo, cd_lo);
            abcd_lohi = _mm_unpackhi_epi32(ab_lo, cd_lo);
            abcd_hi = _mm_unpacklo_epi32(ab_hi, cd_hi);
            abcd_hilo = _mm_unpackhi_epi32(ab_hi, cd_hi);

            efgh_lo = _mm_unpacklo_epi32(ef_lo, gh_lo);
            efgh_lohi = _mm_unpackhi_epi32(ef_lo, gh_lo);
            efgh_hi = _mm_unpacklo_epi32(ef_hi, gh_hi);
            efgh_hilo = _mm_unpackhi_epi32(ef_hi, gh_hi);

            full1 = _mm_unpacklo_epi64(abcd_lo, efgh_lo);
            full2 = _mm_unpackhi_epi64(abcd_lo, efgh_lo);
            full3 = _mm_unpacklo_epi64(abcd_lohi, efgh_lohi);
            full4 = _mm_unpackhi_epi64(abcd_lohi, efgh_lohi);
            full5 = _mm_unpacklo_epi64(abcd_hi, efgh_hi);
            full6 = _mm_unpackhi_epi64(abcd_hi, efgh_hi);
            full7 = _mm_unpacklo_epi64(abcd_hilo, efgh_hilo);

            _mm_storeu_si128((__m128i*)out0, full1);
            _mm_storeu_si128((__m128i*)out1, full2);
            _mm_storeu_si128((__m128i*)out2, full3);
            _mm_storeu_si128((__m128i*)out3, full4);
            _mm_storeu_si128((__m128i*)out4, full5);
            _mm_storeu_si128((__m128i*)out5, full6);
            _mm_storeu_si128((__m128i*)out6, full7);

            out0 += VEC_16CAP;
            out1 += VEC_16CAP;
            out2 += VEC_16CAP;
            out3 += VEC_16CAP;
            out4 += VEC_16CAP;
            out5 += VEC_16CAP;
            out6 += VEC_16CAP;

            a = _mm_lddqu_si128((__m128i*)in0);
            b = _mm_lddqu_si128((__m128i*)in1);
            c = _mm_lddqu_si128((__m128i*)in2);
            d = _mm_lddqu_si128((__m128i*)in3);
            e = _mm_lddqu_si128((__m128i*)in4);
            f = _mm_lddqu_si128((__m128i*)in5);
            g = _mm_lddqu_si128((__m128i*)in6);
            h = _mm_lddqu_si128((__m128i*)in7);
        }
        for (i = end; i < M; i++)
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

    // for grouping 8 and large matrices
    if (N >= 8 && M >= 8)
    {
        __m128i a, b, c, d, e, f, g, h;
        __m128i ab_lo, cd_lo, ef_lo, gh_lo;
        __m128i ab_hi, cd_hi, ef_hi, gh_hi;
        __m128i abcd12, abcd34, abcd56, abcd78,
            efgh12, efgh34, efgh56, efgh78,
            abcdefgh1, abcdefgh2, abcdefgh3, abcdefgh4,
            abcdefgh5, abcdefgh6, abcdefgh7, abcdefgh8;

        int16_t *in1,
            *in2,
            *in3,
            *in4,
            *in5,
            *in6,
            *in7;

        int16_t *out0,
            *out1,
            *out2,
            *out3,
            *out4,
            *out5,
            *out6,
            *out7;

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
                in7 = in6 + N;

            a = _mm_lddqu_si128((__m128i*)in0);
            b = _mm_lddqu_si128((__m128i*)in1);
            c = _mm_lddqu_si128((__m128i*)in2);
            d = _mm_lddqu_si128((__m128i*)in3);
            e = _mm_lddqu_si128((__m128i*)in4);
            f = _mm_lddqu_si128((__m128i*)in5);
            g = _mm_lddqu_si128((__m128i*)in6);
            h = _mm_lddqu_si128((__m128i*)in7);

            out1 = out0 + M,
                out2 = out1 + M,
                out3 = out2 + M,
                out4 = out3 + M,
                out5 = out4 + M,
                out6 = out5 + M,
                out7 = out6 + M;

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

                ab_lo = _mm_unpacklo_epi16(a, b);
                cd_lo = _mm_unpacklo_epi16(c, d);
                ef_lo = _mm_unpacklo_epi16(e, f);
                gh_lo = _mm_unpacklo_epi16(g, h);

                ab_hi = _mm_unpackhi_epi16(a, b);
                cd_hi = _mm_unpackhi_epi16(c, d);
                ef_hi = _mm_unpackhi_epi16(e, f);
                gh_hi = _mm_unpackhi_epi16(g, h);

                abcd12 = _mm_unpacklo_epi32(ab_lo, cd_lo);
                abcd34 = _mm_unpackhi_epi32(ab_lo, cd_lo);
                abcd56 = _mm_unpacklo_epi32(ab_hi, cd_hi);
                abcd78 = _mm_unpackhi_epi32(ab_hi, cd_hi);

                efgh12 = _mm_unpacklo_epi32(ef_lo, gh_lo);
                efgh34 = _mm_unpackhi_epi32(ef_lo, gh_lo);
                efgh56 = _mm_unpacklo_epi32(ef_hi, gh_hi);
                efgh78 = _mm_unpackhi_epi32(ef_hi, gh_hi);

                abcdefgh1 = _mm_unpacklo_epi64(abcd12, efgh12);
                abcdefgh2 = _mm_unpackhi_epi64(abcd12, efgh12);
                abcdefgh3 = _mm_unpacklo_epi64(abcd34, efgh34);
                abcdefgh4 = _mm_unpackhi_epi64(abcd34, efgh34);
                abcdefgh5 = _mm_unpacklo_epi64(abcd56, efgh56);
                abcdefgh6 = _mm_unpackhi_epi64(abcd56, efgh56);
                abcdefgh7 = _mm_unpacklo_epi64(abcd78, efgh78);
                abcdefgh8 = _mm_unpackhi_epi64(abcd78, efgh78);

                _mm_storeu_si128((__m128i*)out0, abcdefgh1);
                _mm_storeu_si128((__m128i*)out1, abcdefgh2);
                _mm_storeu_si128((__m128i*)out2, abcdefgh3);
                _mm_storeu_si128((__m128i*)out3, abcdefgh4);
                _mm_storeu_si128((__m128i*)out4, abcdefgh5);
                _mm_storeu_si128((__m128i*)out5, abcdefgh6);
                _mm_storeu_si128((__m128i*)out6, abcdefgh7);
                _mm_storeu_si128((__m128i*)out7, abcdefgh8);

                out0 += M * 8;
                out1 += M * 8;
                out2 += M * 8;
                out3 += M * 8;
                out4 += M * 8;
                out5 += M * 8;
                out6 += M * 8;
                out7 += M * 8;

                a = _mm_lddqu_si128((__m128i*)in0);
                b = _mm_lddqu_si128((__m128i*)in1);
                c = _mm_lddqu_si128((__m128i*)in2);
                d = _mm_lddqu_si128((__m128i*)in3);
                e = _mm_lddqu_si128((__m128i*)in4);
                f = _mm_lddqu_si128((__m128i*)in5);
                g = _mm_lddqu_si128((__m128i*)in6);
                h = _mm_lddqu_si128((__m128i*)in7);
            }

            in0 = in0 - N_VEC + VEC_16CAP * N;
            out0 = out0 - N_VEC * M + VEC_16CAP;
        }

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
}
