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

#include "GnaException.h"
#include "KernelMacros.h"
#include "pwl.h"

using namespace GNA;

// Mask for resetting xBase buffer address to beginning
static const uint64_t XBASE_ADDRESS_RESET = 0xFFFFFFFFFFFFF000;

// PWL segment bit shift size
static const uint64_t BIT_SHIFT_SIZE = 3;

// Mask for retrieving PWL segment xBase value
const int32_t XBASEMASK = 0xFFFFFFFC;

// Number of segments above which lookup algorithm is used when possible
// otherwise binary search is used
const int32_t PWL_SIZE_ALGORITHM_TRESHOLD = 3;

// Kernel-names macros for PWL functions
#define pwlKernelImplSingleLookup KERNEL(pwlKernelImplSingleLookup)
#define pwlKernelImplSingleBinary KERNEL(pwlKernelImplSingleBinary)
#define pwlKernelImplAllLookup KERNEL(pwlKernelImplAllLookup)
#define pwlKernelImplAllBinary KERNEL(pwlKernelImplAllBinary)

#define PADD(value, pad)   ((((value) + pad -1) / pad) * pad)

#if 1 == GNA_SAT
/**
 * Maximum value of 2B output, used for saturation handling
 */
static const int64_t OUTPUT_2B_MAX = 32767;

/**
 * Minimum value of 2B output, used for saturation handling
 */
static const int64_t OUTPUT_2B_MIN = -32768;
#endif // GNA_SAT

__forceinline static const void pwlSaturateStoreOut(int64_t sum, int16_t* O, uint32_t * const saturationCount)
{
#if 1 == GNA_SAT
    if (sum >= OUTPUT_2B_MIN && sum <= OUTPUT_2B_MAX)
#endif
    {
        *O = (int16_t)sum;
    }
#if 1 == GNA_SAT
    else if (sum > OUTPUT_2B_MAX)
    {
        *O = (int16_t)OUTPUT_2B_MAX;
        (*saturationCount)++;
    }
    else
    {
        *O = (int16_t)OUTPUT_2B_MIN;
        (*saturationCount)++;
    }
#endif
}

void pwlKernelImplSingleBinary(PwlCachedConfig const * const pwl, int32_t I, int16_t * const O,
    uint32_t * const saturationCount)
{
    int64_t sum;
    pwl_x_t* xBase;
    pwl_y_t* seg;
    uint32_t k;
    uint32_t k_upper;
    uint32_t k_lower;

    if (I > pwl->Binary.xBase0)
    {
        k_upper = pwl->segmentCount;
        k = k_upper >> 1;
        k_lower = 0;
        xBase = pwl->Binary.xBase + k;
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
        seg = (pwl_y_t*)(xBase + pwl->segmentCount);
        sum *= seg->slope; // prod = diff * slope
        sum = sum >> seg->shift; // prod_shift = prod >> slope_shift
        sum += seg->yBase;                   // sum = prod_shift + ybase;
        pwlSaturateStoreOut(sum, O, saturationCount);
    }
    else
    {
        *O = pwl->Binary.yBase0;
    }
}

void pwlKernelImplAllBinary(PwlCachedConfig const * const pwl, PwlOutputConfig const * const outputConfig)
{
    int64_t sum;                    // tmp sum
    const int32_t* input;           // input row
    const int32_t* inputEnd;           // input row
    int16_t* output;                // output row
    pwl_x_t* xBase;
    pwl_x_t * const xBaseReset = pwl->Binary.xBase + (pwl->segmentCount >> 1);;
    pwl_y_t* seg;
    uint32_t k;
    uint32_t k_upper;
    uint32_t k_lower;

    // input and sum prefetch
    input = outputConfig->input;
    inputEnd = input + outputConfig->elementCount;
    output = outputConfig->output;
    do
    {
        if (*input > pwl->Binary.xBase0)
        {
            k_upper = pwl->segmentCount;
            k = k_upper >> 1;
            k_lower = 0;
            xBase = xBaseReset;
            sum = (int64_t)*input + *xBase;
            do
            {
                if (sum < 0)
                {
                    k_upper = k;
                    k = (k + k_lower) >> 1;
                    xBase = (pwl_x_t*)((int64_t)xBase & XBASE_ADDRESS_RESET);
                    xBase += k;
                    sum = (int64_t)*input + *xBase;
                }
                else
                {
                    k_lower = k;
                    k = (k_upper + k) >> 1;
                    xBase = (pwl_x_t*)((int64_t)xBase & XBASE_ADDRESS_RESET);
                    xBase += k;
                    sum = (int64_t)*input + *xBase;
                }
            } while (k_upper > k_lower + 1);
            seg = (pwl_y_t*)(xBase + pwl->segmentCount);
            sum *= seg->slope; // prod = diff * slope
            sum = sum >> seg->shift; // prod_shift = prod >> slope_shift
            sum += seg->yBase;                   // sum = prod_shift + ybase;
            pwlSaturateStoreOut(sum, output, outputConfig->saturationCount);
        }
        else
        {
            *output = pwl->Binary.yBase0;
        }
        output++;
        input++;
    } while (input < inputEnd);
}

void pwlKernelImplSingleLookup(PwlCachedConfig const * const pwl, int32_t I, int16_t * const O, 
    uint32_t * const saturationCount)
{
    int64_t k;                      // lookup table iterator and helper
    int64_t sum;                    // tmp sum
    pwl_u_t* lookup = pwl->Lookup.table;  // lookup table

    sum = I + (int64_t)pwl->Lookup.xBase0Neg;
    if (sum > 0)
    {
        k = sum + pwl->Lookup.xBase1diff;
        if (k >= 0)
        {
            sum = k;
            k = k >> pwl->Lookup.width;
            if (k > pwl->Lookup.count)
            {
                k = pwl->Lookup.count;
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
            sum *= pwl->Lookup.slope0; // prod = diff * slope
            sum = sum >> pwl->Lookup.shift0; // prod_shift = prod >> slope_shift
            sum += pwl->Lookup.yBase0;                   // sum = prod_shift + ybase;
        }
        pwlSaturateStoreOut(sum, O, saturationCount);
    }
    else
    {
        *O = pwl->Lookup.yBase0;
    }
}

void pwlKernelImplAllLookup(PwlCachedConfig const * const pwl, PwlOutputConfig const * const outputConfig)
{
    int64_t k;                      // lookup table iterator and helper
    int64_t sum;                    // tmp sum
    const int32_t* input;           // input row
    const int32_t* inputEnd;           // input row
    int16_t* output;                // output row
    pwl_u_t* lookup = pwl->Lookup.table;  // lookup table
    pwl_x_t xBase0 = pwl->Lookup.xBase0Neg;
    pwl_x_t xBase1diff = pwl->Lookup.xBase1diff;
    int32_t count = pwl->Lookup.count;

    // input and k prefetch
    input = outputConfig->input;
    inputEnd = input + outputConfig->elementCount;
    output = outputConfig->output;
    do
    {
        lookup = pwl->Lookup.table;
        sum = *input + (int64_t)xBase0;
        if (sum > 0)
        {
            k = sum + xBase1diff;
            if (k >= 0)
            {
                sum = k;
                k = k >> pwl->Lookup.width;
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
                sum *= pwl->Lookup.slope0;      // prod = diff * slope
                sum = sum >> pwl->Lookup.shift0;// prod_shift = prod >> slope_shift
                sum += pwl->Lookup.yBase0;    // sum = prod_shift + ybase;
            }
            pwlSaturateStoreOut(sum, output, outputConfig->saturationCount);
        }
        else
        {
            *output = pwl->Lookup.yBase0;
        }
        output++;
        input++;
    } while (input < inputEnd);
}

PwlCached::PwlCached(int32_t const * const inputIn, nn_pwl_seg const * const segments, uint32_t segmentCountIn)
{
    int32_t s;                      // PWL segment iterator
    int32_t i;                      // pwl.lookup element offset iterator (beginning) 
    int64_t j;                      // pwl.lookup element offset iterator (end) 
    pwl_x_t xBaseAtmp;                 // left segment xBase value (extracted)
    pwl_x_t xBaseBtmp;                 // right segment x Base value (extracted)
    int64_t widthTmp = UINT32_MAX;     // pwl.lookup segment widthTmp - minimum distance between pwl.segments' xbases
    int64_t countTmp;                  // pwl.lookup segment countTmp (active)
    pwl_s_t usegTmp;
    bool useLookup = false;
    ActivateAll = NULL;
    ActivateSingle = NULL;
    pwl.input = inputIn;
    pwl.segmentCount = segmentCountIn;

    if (pwl.segmentCount > PWL_SIZE_ALGORITHM_TRESHOLD)
    {
        ActivateAll = pwlKernelImplAllLookup;
        ActivateSingle = pwlKernelImplSingleLookup;
    }
    if (pwl.segmentCount <= PWL_SIZE_ALGORITHM_TRESHOLD)
    {
        ActivateAll = pwlKernelImplAllBinary;
        ActivateSingle = pwlKernelImplSingleBinary;
    }

    if (pwl.segmentCount > PWL_SIZE_ALGORITHM_TRESHOLD)
    {
        // first PWL pass - analyze PWL
        xBaseBtmp = segments[1].xBase & XBASEMASK;
        for (i = 2; i < pwl.segmentCount; i++)
        {
            xBaseAtmp = xBaseBtmp;
            xBaseBtmp = segments[i].xBase & XBASEMASK;
            j = xBaseBtmp - xBaseAtmp; // min xbase diff > abs(XBASEMASK) = 4
            if (j < widthTmp)
            {
                widthTmp = j;
            }
        }
        if (widthTmp >= 1 && widthTmp <= INT32_MAX)
        {
#ifndef _WIN64
            // scan 32 MSB
            _BitScanReverse((unsigned long*)&s, (unsigned long)(widthTmp >> (sizeof(s) * CHAR_BIT)));
            if (0 == s)
            {
                // scan 32 LSB
                _BitScanReverse((unsigned long*)&s, (unsigned long)(widthTmp & UINT32_MAX));
            }
            else
            {
                s += sizeof(s) * CHAR_BIT;
            }
#else
            _BitScanReverse64((unsigned long*)&s, widthTmp);
#endif
            widthTmp = (uint64_t)1 << (uint64_t)s;
            j = PADD((int64_t)(
                segments[pwl.segmentCount - 1].xBase & XBASEMASK) - (segments[1].xBase & XBASEMASK),
                widthTmp);
            countTmp = j / widthTmp + 1;
            if (0 < countTmp && countTmp <= PWL_LOOKUP_COUNT)
            {
                useLookup = true;
            }
        }
    }
    // second pass - PWL pwl.lookup build
    if (useLookup)
    {
        allocateLookupCaches();
        pwl.Lookup.xBase0 = segments[0].xBase & XBASEMASK;
        pwl.Lookup.xBase0Neg = -1 * pwl.Lookup.xBase0;
        pwl.Lookup.shift0 = ((segments[0].xBase & ~XBASEMASK) + 1) << BIT_SHIFT_SIZE;
        pwl.Lookup.slope0 = segments[0].slope;
        pwl.Lookup.yBase0 = segments[0].yBase;
        pwl.Lookup.xBase1diff = pwl.Lookup.xBase0 - (segments[1].xBase & XBASEMASK);
        pwl.Lookup.width = s;
        pwl.Lookup.count = (uint16_t)countTmp - 1;
        ActivateAll = pwlKernelImplAllLookup;
        ActivateSingle = pwlKernelImplSingleLookup;
        widthTmp = s - 1;
        i = 0;
        j = 0;
        s = 2;
        xBaseAtmp = (segments[1].xBase & XBASEMASK);
        xBaseBtmp = segments[s].xBase & XBASEMASK;
        while (s < pwl.segmentCount)
        {
            usegTmp.xBase = pwl.Lookup.xBase0 - pwl.Lookup.xBase1diff - (pwl_x_t)(segments[s - 1].xBase & XBASEMASK);
            usegTmp.shift = ((segments[s - 1].xBase & ~XBASEMASK) + 1) << BIT_SHIFT_SIZE;
            usegTmp.slope = segments[s - 1].slope;
            usegTmp.resvd = 0;
            usegTmp.yBase = segments[s - 1].yBase;

            j = (xBaseBtmp - xBaseAtmp) >> widthTmp;
            // take care of case when PWL segment is ending at Even pwl.lookup entry
            if ((0 == (j & 1)) && (xBaseBtmp != xBaseAtmp + (j << widthTmp)))
            {
                j++;
            }
            for (; i < j; i++)
            {
                if (i & 1)
                {
                    pwl.Lookup.table[(i & ~1) / 2].xBaseB = usegTmp.xBase;
                    pwl.Lookup.table[(i & ~1) / 2].slopeB = usegTmp.slope;
                    pwl.Lookup.table[(i & ~1) / 2].shiftB = usegTmp.shift;
                    pwl.Lookup.table[(i & ~1) / 2].yBaseB = usegTmp.yBase;
                }
                else
                {
                    pwl.Lookup.table[i / 2].xBaseA = usegTmp.xBase;
                    pwl.Lookup.table[i / 2].slopeA = usegTmp.slope;
                    pwl.Lookup.table[i / 2].shiftA = usegTmp.shift;
                    pwl.Lookup.table[i / 2].yBaseA = usegTmp.yBase;
                }
            }
            s++;
            xBaseBtmp = segments[s].xBase & XBASEMASK;
        }
        usegTmp.xBase = pwl.Lookup.xBase0 - pwl.Lookup.xBase1diff  - (pwl_x_t)(segments[s - 1].xBase & XBASEMASK);
        usegTmp.shift = ((segments[s - 1].xBase & ~XBASEMASK) + 1) << BIT_SHIFT_SIZE;
        usegTmp.slope = segments[s - 1].slope;
        usegTmp.resvd = 0;
        usegTmp.yBase = segments[s - 1].yBase;
        for (; i < countTmp * PWL_LOOKUP_SEG_SCOUNT; i++)
        {
            if (i & 1)
            {
                pwl.Lookup.table[(i & ~1) / 2].xBaseB = usegTmp.xBase;
                pwl.Lookup.table[(i & ~1) / 2].slopeB = usegTmp.slope;
                pwl.Lookup.table[(i & ~1) / 2].shiftB = usegTmp.shift;
                pwl.Lookup.table[(i & ~1) / 2].yBaseB = usegTmp.yBase;
            }
            else
            {
                pwl.Lookup.table[i / 2].xBaseA = usegTmp.xBase;
                pwl.Lookup.table[i / 2].slopeA = usegTmp.slope;
                pwl.Lookup.table[i / 2].shiftA = usegTmp.shift;
                pwl.Lookup.table[i / 2].yBaseA = usegTmp.yBase;
            }
        }
        for (i = 0; i < countTmp; i++)
        {
            pwl.Lookup.table[i].xBaseA = pwl.Lookup.table[i].xBaseA - pwl.Lookup.table[i].xBaseB;
        }
    }
    else
    {
        allocateBinaryCaches();
        ActivateAll = pwlKernelImplAllBinary;
        ActivateSingle = pwlKernelImplSingleBinary;
        pwl.Binary.xBase0 = segments[0].xBase & XBASEMASK;
        pwl.Binary.yBase0 = segments[0].yBase;
        i = 0;
        for (; i < pwl.segmentCount; i++)
        {
            pwl.Binary.xBase[i]      = -1 * (pwl_x_t)(segments[i].xBase & XBASEMASK);
            pwl.Binary.ySeg[i].shift = ((segments[i].xBase & ~XBASEMASK) + 1) << BIT_SHIFT_SIZE;
            pwl.Binary.ySeg[i].slope = segments[i].slope;
            pwl.Binary.ySeg[i].resvd = 0;
            pwl.Binary.ySeg[i].yBase = segments[i].yBase;
        }
    }
}

PwlCached::~PwlCached()
{
    if (nullptr != pwl.Lookup.table)
    {
        _gna_free(pwl.Lookup.table);
        memset(&pwl, 0, sizeof(pwl));
    }
    if (nullptr != pwl.Binary.xBase)
    {
        _gna_free(pwl.Binary.xBase);
        memset(&pwl, 0, sizeof(pwl));
    }
}

void PwlCached::allocateBinaryCaches()
{
    auto totalSize = pwl.segmentCount * (sizeof(pwl_x_t) + sizeof(pwl_y_t));
    pwl.Binary.xBase = (pwl_x_t*)_gna_malloc(totalSize);
    if (nullptr == pwl.Binary.xBase)
    {
        throw GnaException(GNA_ERR_RESOURCES);
    }
    memset(pwl.Binary.xBase, 0, totalSize);
    pwl.Binary.ySeg = (pwl_y_t*)(pwl.Binary.xBase + pwl.segmentCount);
}

void PwlCached::allocateLookupCaches()
{
    pwl.Lookup.table = (pwl_u_t*)_gna_malloc(PWL_LOOKUP_SIZE);
    if (nullptr == pwl.Lookup.table)
    {
        throw GnaException(GNA_ERR_RESOURCES);
    }
    memset(pwl.Lookup.table, 0xff, PWL_LOOKUP_SIZE);
}
