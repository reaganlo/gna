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

#include "pwl.h"
#include "igemv.h"

/**
 * Mask for resetting xBase buffer address to beginning
 */
static const uint64_t XBASE_ADDRESS_RESET = 0xFFFFFFFFFFFFF000;

/**
* PWL segment bit shift size
*/
static const uint64_t BIT_SHIFT_SIZE = 3;

//#if 1 == GNA_SAT
/**
 * Maximum value of 2B output, used for saturation handling
 */
static const int64_t OUTPUT_2B_MAX = 32767;

/**
 * Minimum value of 2B output, used for saturation handling
 */
static const int64_t OUTPUT_2B_MIN = -32768;
//#endif // GNA_SAT

__forceinline
static const void PwlSaturateStoreOutSingle(
    int64_t         sum,
    int16_t*        O,
    pwl_params*     params)
{
//#if 1 == GNA_SAT
    if (sum >= OUTPUT_2B_MIN && sum <= OUTPUT_2B_MAX)
//#endif
    {
        *O = (int16_t)sum;
    }
//#if 1 == GNA_SAT
    else if (sum > OUTPUT_2B_MAX)
    {
        *O = (int16_t)OUTPUT_2B_MAX;
        (*params->nSaturated)++;
    }
    else
    {
        *O = (int16_t)OUTPUT_2B_MIN;
        (*params->nSaturated)++;
    }
//#endif
}

__forceinline
static const void PwlSaturateStoreOutAll(
    int64_t         sum,
    int32_t         j,
    int16_t*        output,
    pwl_params*     params)
{
//#if 1 == GNA_SAT
    if (sum >= OUTPUT_2B_MIN && sum <= OUTPUT_2B_MAX)
//#endif
    {
        output[j] = (int16_t)sum;
    }
//#if 1 == GNA_SAT
    else if (sum > OUTPUT_2B_MAX)
    {
        output[j] = (int16_t)OUTPUT_2B_MAX;
        (*params->nSaturated)++;
    }
    else
    {
        output[j] = (int16_t)OUTPUT_2B_MIN;
        (*params->nSaturated)++;
    }
//#endif
}

void
PwlApplySingleBinary(
    pwl_params*     params,
    int32_t         I,
    int16_t*        O)
{
    int64_t     sum;
    pwl_x_t*    xBase;
    pwl_y_t*    seg;
    uint32_t    k;
    uint32_t    k_upper;
    uint32_t    k_lower;

    if (I > params->xBase0Bi)
    {
        k_upper = params->NS;
        k = k_upper >> 1;
        k_lower = 0;
        xBase = params->xBase + k;
        sum = (int64_t)I + *xBase;
        do
        {
            if (sum < 0)
            {
                k_upper = k;
                k = (k + k_lower) >> 1;
                xBase = (pwl_x_t*)((int64_t)xBase & XBASE_ADDRESS_RESET);
                xBase += k;
                sum = (int64_t)I + *xBase;
            }
            else
            {
                k_lower = k;
                k = (k_upper + k) >> 1;
                xBase = (pwl_x_t*)((int64_t)xBase & XBASE_ADDRESS_RESET);
                xBase += k;
                sum = (int64_t)I + *xBase;
            }
        } while (k_upper > k_lower + 1);
        seg = (pwl_y_t*)((int8_t*)xBase + PWL_X_BUFFER_SIZE);
        sum *= seg->slope; // prod = diff * slope
        sum = sum >> seg->shift; // prod_shift = prod >> slope_shift
        sum += seg->yBase;                   // sum = prod_shift + ybase;
        PwlSaturateStoreOutSingle(sum, O, params);
    }
    else
    {
        *O = params->yBase0Bi;
    }
}

void
PwlApplyAllBinary(
    pwl_params*     params)
{
    int32_t     i;                      // row iterator
    int32_t     j;                      // column iterator
    int64_t     sum;                    // tmp sum
    int32_t*    input;                  // input row
    int16_t*    output;                 // output row
    pwl_x_t*    xBase;
    pwl_y_t*    seg;
    const int32_t nOutCols = params->nOutCols;
    uint32_t    k;
    uint32_t    k_upper;
    uint32_t    k_lower;

    // input and sum prefetch
    params->xBase += (params->NS >> 1);
    i = params->nRowBegin;
    j = params->nColBegin;
    input = params->I + i * nOutCols;
    output = params->O + i * nOutCols;
    do
    {
        do
        {
            if (input[j] > params->xBase0Bi)
            {
                k_upper = params->NS;
                k = k_upper >> 1;
                k_lower = 0;
                xBase = params->xBase;
                sum = (int64_t)input[j] + *xBase;
                do
                {
                    if (sum < 0)
                    {
                        k_upper = k;
                        k = (k + k_lower) >> 1;
                        xBase = (pwl_x_t*)((int64_t)xBase & XBASE_ADDRESS_RESET);
                        xBase += k;
                        sum = (int64_t)input[j] + *xBase;
                    }
                    else
                    {
                        k_lower = k;
                        k = (k_upper + k) >> 1;
                        xBase = (pwl_x_t*)((int64_t)xBase & XBASE_ADDRESS_RESET);
                        xBase += k;
                        sum = (int64_t)input[j] + *xBase;
                    }
                } while (k_upper > k_lower + 1);
                seg = (pwl_y_t*)((int8_t*)xBase + PWL_X_BUFFER_SIZE);
                sum *= seg->slope; // prod = diff * slope
                sum = sum >> seg->shift; // prod_shift = prod >> slope_shift
                sum += seg->yBase;                   // sum = prod_shift + ybase;
                PwlSaturateStoreOutAll(sum, j, output, params);
            }
            else
            {
                output[j] = params->yBase0Bi;
            }
             // go to next column input and prefetch input
            j++;
        } while (j <= params->nColEnd);
        // go to next row and prefetch input
        input += nOutCols;
        output += nOutCols;
        j = params->nColBegin;
        i++;
    } while (i <= params->nRowEnd);

    params->xBase -= (params->NS >> 1);
}

void
PwlApplySingleLookup(
    pwl_params*     params,
    int32_t         I,
    int16_t*        O)
{
    int64_t     k;                      // lookup table iterator and helper
    int64_t     sum;                    // tmp sum
    pwl_u_t*    lookup = params->lookup;// lookup table

    sum = I + (int64_t)params->xBase0Neg;
    if (sum > 0)
    {
        k = sum + params->xBase1diff;
        if (k >= 0)
        {
            sum = k;
            k = k >> params->width;
            if (k > params->count)
            {
                k = params->count;
            }
            lookup += k;

            sum += lookup->xBaseB;
            if (sum >= 0)
            {
                sum *= lookup->slopeB; // prod = diff * slope
                sum = sum >> lookup->shiftB; // prod_shift = prod >> slope_shift
                sum += lookup->yBaseB;                   // sum = prod_shift + ybase;
            }
            else
            {
                sum += lookup->xBaseA;
                sum *= lookup->slopeA; // prod = diff * slope
                sum = sum >> lookup->shiftA; // prod_shift = prod >> slope_shift
                sum += lookup->yBaseA;
            }
        }
        else
        {
            sum *= params->slope0; // prod = diff * slope
            sum = sum >> params->shift0; // prod_shift = prod >> slope_shift
            sum += params->yBase0Lu;                   // sum = prod_shift + ybase;
        }
        PwlSaturateStoreOutSingle(sum, O, params);
    }
    else
    {
        *O = params->yBase0Lu;
    }
}

void
PwlApplyAllLookup(
    pwl_params*     params)
{
    int32_t     i;                      // row iterator
    int32_t     j;                      // column iterator
    int64_t     k;                      // lookup table iterator and helper
    int64_t     sum;                    // tmp sum
    int32_t*    input;                  // input row
    int16_t*    output;                 // output row
    pwl_u_t*    lookup = params->lookup;// lookup table
    pwl_x_t     xBase0Lu = params->xBase0Neg;
    pwl_x_t     xBase1diff = params->xBase1diff;
    int32_t     count = params->count;
    int32_t     nOutCols = params->nOutCols;

    // input and k prefetch
    i = params->nRowBegin;
    j = params->nColBegin;
    input = params->I + i * nOutCols;
    output = params->O + i * nOutCols;
    do
    {
        do
        {       
            lookup = params->lookup;
            sum = input[j] + (int64_t)xBase0Lu;
            if (sum > 0)
            {   
                k = sum + xBase1diff;
                if (k >= 0)
                {
                    sum = k;
                    k = k >> params->width;
                    if (k > count)
                    {
                        k = count;
                    }
                    lookup += k;

                    sum += lookup->xBaseB;
                    if (sum >= 0)
                    {
                        sum *= lookup->slopeB; // prod = diff * slope
                        sum = sum >> lookup->shiftB; // prod_shift = prod >> slope_shift
                        sum += lookup->yBaseB;                   // sum = prod_shift + ybase;
                    }
                    else
                    {
                        sum += lookup->xBaseA;
                        sum *= lookup->slopeA; // prod = diff * slope
                        sum = sum >> lookup->shiftA; // prod_shift = prod >> slope_shift
                        sum += lookup->yBaseA;
                    }
                }
                else
                {
                    sum *= params->slope0;      // prod = diff * slope
                    sum = sum >> params->shift0;// prod_shift = prod >> slope_shift
                    sum += params->yBase0Lu;    // sum = prod_shift + ybase;
                }
                PwlSaturateStoreOutAll(sum, j, output, params);
            }
            else
            {
                output[j] = params->yBase0Lu;
            }
            j++;
        } while (j <= params->nColEnd);
        // pointer pre-calculation
        input += nOutCols;
        output += nOutCols;
        j = params->nColBegin;
        i++;
    } while (i <= params->nRowEnd);
}

void
PwlPrepareAuxBuffers(
    pwl_params*     params,
    nn_pwl_seg*   segments
)
{
    pwl_u_t*    lookup;                 // lookup table
    int32_t     s;                      // PWL segment iterator
    int32_t     i;                      // lookup element offset iterator (beginning) 
    int64_t     j;                      // lookup element offset iterator (end) 
    pwl_x_t     xBaseA;                 // left segment xBase value (extracted)
    pwl_x_t     xBaseB;                 // right segment x Base value (extracted)
    int64_t     width = UINT32_MAX;     // lookup segment width - minimum distance between segments' xbases
    int64_t     count;                  // lookup segment count (active)
    pwl_s_t     useg;
    bool        useLookup = false;

    params->pwlAll = NULL;
    params->pwlSingle = NULL;

    if (params->NS > PWL_SIZE_ALGORITHM_TRESHOLD && params->prevLu == segments)
    {
        params->pwlAll = PwlApplyAllLookup;
        params->pwlSingle = PwlApplySingleLookup;
        return;
    }
    if (params->NS <= PWL_SIZE_ALGORITHM_TRESHOLD && params->prevBi == segments)
    {
        params->pwlAll = PwlApplyAllBinary;
        params->pwlSingle = PwlApplySingleBinary;
        return;
    }

    if (params->NS > PWL_SIZE_ALGORITHM_TRESHOLD)
    {
        // first PWL pass - analyze PWL
        xBaseB = segments[1].xBase & XBASEMASK;
        for (i = 2; i < params->NS; i++)
        {
            xBaseA = xBaseB;
            xBaseB = segments[i].xBase & XBASEMASK;
            j = xBaseB - xBaseA; // min xbase diff > abs(XBASEMASK) = 4
            if (j < width)
            {
                width = j;
            }
        }
        if (width >= 1 && width <= INT32_MAX)
        {
#ifndef _WIN64
            // scan 32 MSB
            _BitScanReverse((unsigned long*)&s, (unsigned long)(width >> (sizeof(s) * CHAR_BIT)));
            if (0 == s)
            {
                // scan 32 LSB
                _BitScanReverse((unsigned long*)&s, (unsigned long)(width & UINT32_MAX));
            }
            else
            {
                s += sizeof(s) * CHAR_BIT;
            }
#else
            _BitScanReverse64((unsigned long*)&s, width);
#endif
            width = (uint64_t)1 << (uint64_t)s;
            j = PADD((int64_t)(
                segments[params->NS - 1].xBase & XBASEMASK) - (segments[1].xBase & XBASEMASK),
                width);
            count = j / width + 1;
            if (0 < count && count <= PWL_LOOKUP_COUNT)
            {
                useLookup = true;
            }
        }
    }
    // second pass - PWL lookup build
    if (useLookup)
    {
        params->xBase0Lu = segments[0].xBase & XBASEMASK;
        params->xBase0Neg = -1 * params->xBase0Lu;
        params->shift0 = ((segments[0].xBase & ~XBASEMASK) + 1) << BIT_SHIFT_SIZE;
        params->slope0 = segments[0].slope;
        params->yBase0Lu = segments[0].yBase;
        params->xBase1diff = params->xBase0Lu - (segments[1].xBase & XBASEMASK);
        params->width = s;
        params->count = (uint16_t)count - 1;
        params->pwlAll = PwlApplyAllLookup;
        params->pwlSingle = PwlApplySingleLookup;
        lookup = params->lookup;   
        width = s - 1;
        i = 0;
        j = 0;
        s = 2;
        xBaseA = (segments[1].xBase & XBASEMASK);
        xBaseB = segments[s].xBase & XBASEMASK;
        while (s < params->NS)
        {
            useg.xBase = params->xBase0Lu - params->xBase1diff - (pwl_x_t)(segments[s - 1].xBase & XBASEMASK);
            useg.shift = ((segments[s - 1].xBase & ~XBASEMASK) + 1) << BIT_SHIFT_SIZE;
            useg.slope = segments[s - 1].slope;
            useg.resvd = 0;
            useg.yBase = segments[s - 1].yBase;

            j = (xBaseB - xBaseA) >> width;
            // take care of case when PWL segment is ending at Even lookup entry
            if ((0 == (j & 1)) && (xBaseB != xBaseA + (j << width)))
            {
                j++;
            }
            for (; i < j; i++)
            {
                if (i & 1)
                {
                    lookup[(i & ~1) / 2].xBaseB = useg.xBase;
                    lookup[(i & ~1) / 2].slopeB = useg.slope;
                    lookup[(i & ~1) / 2].shiftB = useg.shift;
                    lookup[(i & ~1) / 2].yBaseB = useg.yBase;
                }
                else
                {
                    lookup[i / 2].xBaseA = useg.xBase;
                    lookup[i / 2].slopeA = useg.slope;
                    lookup[i / 2].shiftA = useg.shift;
                    lookup[i / 2].yBaseA = useg.yBase;
                }
            }
            s++;
            xBaseB = segments[s].xBase & XBASEMASK;
        }
        useg.xBase = params->xBase0Lu - params->xBase1diff  - (pwl_x_t)(segments[s - 1].xBase & XBASEMASK);
        useg.shift = ((segments[s - 1].xBase & ~XBASEMASK) + 1) << BIT_SHIFT_SIZE;
        useg.slope = segments[s - 1].slope;
        useg.resvd = 0;
        useg.yBase = segments[s - 1].yBase;
        for (; i < count * PWL_LOOKUP_SEG_SCOUNT; i++)
        {
            if (i & 1)
            {
                lookup[(i & ~1) / 2].xBaseB = useg.xBase;
                lookup[(i & ~1) / 2].slopeB = useg.slope;
                lookup[(i & ~1) / 2].shiftB = useg.shift;
                lookup[(i & ~1) / 2].yBaseB = useg.yBase;
            }
            else
            {
                lookup[i / 2].xBaseA = useg.xBase;
                lookup[i / 2].slopeA = useg.slope;
                lookup[i / 2].shiftA = useg.shift;
                lookup[i / 2].yBaseA = useg.yBase;
            }
        }
        for (i = 0; i < count; i++)
        {
            lookup[i].xBaseA = lookup[i].xBaseA - lookup[i].xBaseB;
        }
        params->prevLu = segments;
    }
    else
    {
        params->pwlAll = PwlApplyAllBinary;
        params->pwlSingle = PwlApplySingleBinary;
        params->xBase0Bi = segments[0].xBase & XBASEMASK;
        params->yBase0Bi = segments[0].yBase;
        i = 0;
        for (; i < params->NS; i++)
        {
            params->xBase[i]      = -1 * (pwl_x_t)(segments[i].xBase & XBASEMASK);
            params->ySeg[i].shift = ((segments[i].xBase & ~XBASEMASK) + 1) << BIT_SHIFT_SIZE;
            params->ySeg[i].slope = segments[i].slope;
            params->ySeg[i].resvd = 0;
            params->ySeg[i].yBase = segments[i].yBase;
        }
        params->prevBi = segments;
    }
}
