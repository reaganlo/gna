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

#if defined(_WIN32)
#pragma warning (disable: 592)
#endif

#if defined(__GNUC__)
#include <limits.h>
#endif

#include "GnaException.h"
#include "KernelArguments.h"
#include "KernelMacros.h"
#include "pwl.h"

#define UNREFERENCED_PARAMETER(P) ((void)(P))

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
const int32_t PWL_SIZE_OPT_ALGORITHM_TRESHOLD = 32;

#if 1 == GNA_SAT
/**
 * Maximum value of 2B output, used for saturation handling
 */
static const int64_t OUTPUT_MAX[5]{0, INT8_MAX, INT16_MAX, 0, INT32_MAX };

/**
 * Minimum value of 2B output, used for saturation handling
 */
static const int64_t OUTPUT_MIN[5]{0, INT8_MIN, INT16_MIN, 0, INT32_MIN };
#endif // GNA_SAT

#define PADD(value, pad)   ((((value) + pad -1) / pad) * pad)

static __forceinline void pwlSaturateStoreOut(int64_t sum, int8_t* O,
    uint32_t * const saturationCount, uint32_t bytesPerOutput)
{
    UNREFERENCED_PARAMETER(saturationCount);
#if 1 == GNA_SAT

    if (sum >= OUTPUT_MIN[bytesPerOutput] && sum <= OUTPUT_MAX[bytesPerOutput])
#endif
    {
        if (bytesPerOutput == 1)
            *(int8_t*)O = (int8_t)sum;
        else if (bytesPerOutput == 2)
            *(int16_t*)O = (int16_t)sum;
        else if (bytesPerOutput == 4)
            *(int32_t*)O = (int32_t)sum;
    }
#if 1 == GNA_SAT
    else if (sum > OUTPUT_MAX[bytesPerOutput])
    {
        if (bytesPerOutput == 1)
            *(int8_t*)O = (int8_t)OUTPUT_MAX[bytesPerOutput];
        else if (bytesPerOutput == 2)
            *(int16_t*)O = (int16_t)OUTPUT_MAX[bytesPerOutput];
        else if (bytesPerOutput == 4)
            *(int32_t*)O = (int32_t)OUTPUT_MAX[bytesPerOutput];
        (*saturationCount)++;
    }
    else
    {
        if (bytesPerOutput == 1)
            *(int8_t*)O = (int8_t)OUTPUT_MIN[bytesPerOutput];
        else if (bytesPerOutput == 2)
            *(int16_t*)O = (int16_t)OUTPUT_MIN[bytesPerOutput];
        else if (bytesPerOutput == 4)
            *(int32_t*)O = (int32_t)OUTPUT_MIN[bytesPerOutput];
        (*saturationCount)++;
    }
#endif
}

__forceinline int32_t pwlFindFirstBitSet(int64_t bits)
{
    int32_t s = 0;

#if defined(__GNUC__) && defined(__LP64__)
    int32_t leadingZeros = __builtin_clzll(bits);
    s = sizeof(int64_t) * CHAR_BIT - leadingZeros - 1;
#elif defined(__GNUC__)
    int32_t widthHigh = (int32_t)(bits >> sizeof(s) * CHAR_BIT);
    if (widthHigh != 0)
    {
        int32_t leadingZeros = __builtin_clz(widthHigh);
        s = sizeof(int64_t) * CHAR_BIT - leadingZeros - 1;
    }
    else
    {
        int32_t widthLow = (int32_t)bits;
        int32_t leadingZeros = __builtin_clz(widthLow);
        s = sizeof(int32_t) * CHAR_BIT - leadingZeros - 1;
    }
#elif defined(_WIN64)
#if !defined(_MSC_VER)
    _BitScanReverse64((unsigned __int32*)&s, bits);
#else
    _BitScanReverse64((unsigned long*)&s, bits);
#endif
#elif defined(_WIN32)
    // scan 32 MSB
#if !defined(_MSC_VER)
    _BitScanReverse((unsigned __int32*)&s, (unsigned long)(bits >> (sizeof(s) * CHAR_BIT)));
#else
    _BitScanReverse((unsigned long*)&s, (unsigned long)(bits >> (sizeof(s) * CHAR_BIT)));
#endif
    if (0 == s)
    {
        // scan 32 LSB
#if !defined(_MSC_VER)
        _BitScanReverse((unsigned __int32*)&s, (unsigned long)(bits & UINT32_MAX));
#else
        _BitScanReverse((unsigned long*)&s, (unsigned long)(bits & UINT32_MAX));
#endif
    }
    else
    {
        s += sizeof(s) * CHAR_BIT;
    }
#endif
    return s;
}

#define pwlKernelImplSingleBinary KERNEL(pwlKernelImplSingleBinary)
void pwlKernelImplSingleBinary(PwlCachedConfig const * const pwl, int32_t I, int16_t* O,
    uint32_t * const saturationCount)
{
    int64_t     sum;
    nn_pwl_seg* segment;
    uint32_t    k;
    uint32_t    k_upper;
    uint32_t    k_lower;

    segment = pwl->Binary.source;
    if (I > pwl->Binary.xBase0)
    {
        k_upper = pwl->segmentCount;
        k = k_upper >> 1;
        k_lower = 0;
        sum = (int64_t)I - (int64_t)(segment[k].xBase & XBASEMASK);
        do
        {
            if (sum < 0)
            {
                k_upper = k;
                k += k_lower;
            }
            else
            {
                k_lower = k;
                k += k_upper;
            }
            k >>= 1;
            sum = (int64_t)I - (int32_t)(segment[k].xBase & XBASEMASK);
        } while (k_upper > k_lower + 1);
        sum *= segment[k].slope; // prod = diff * slope
        sum >>= (((segment[k].xBase & ~XBASEMASK) + 1) << BIT_SHIFT_SIZE); // prod_shift = prod >> slope_shift
        sum += segment[k].yBase;                   // sum = prod_shift + ybase;
        pwlSaturateStoreOut(sum, (int8_t*)O, saturationCount, pwl->bytesPerOutput);
    }
    else
    {
        *O = pwl->Binary.yBase0;
    }
}

#define pwlKernelImplAllBinary KERNEL(pwlKernelImplAllBinary)
void pwlKernelImplAllBinary(ExecutionKernelConfig<ActivationConfig> const * const config)
{
    int64_t     sum;
    int32_t*    input;
    int32_t*    inputEnd;
    int8_t*    output;
    nn_pwl_seg* segment;
    uint32_t    k;
    uint32_t    k_upper;
    uint32_t    k_lower;
    auto pwl = &config->RequestConfig->Transform.Kernel->pwl;

    // input and sum prefetch
    segment = pwl->Binary.source;
    input = (int32_t*)config->RequestConfig->Inputs;
    inputEnd = input + config->RequestConfig->Transform.ElementCount;
    output = config->RequestConfig->Outputs;
    do
    {
        if (*input > pwl->Binary.xBase0)
        {
            k_upper = pwl->segmentCount;
            k = k_upper >> 1;
            k_lower = 0;
            sum = (int64_t)*input - (int64_t)(segment[k].xBase & XBASEMASK);
            do
            {
                if (sum < 0)
                {
                    k_upper = k;
                    k += k_lower;
                }
                else
                {
                    k_lower = k;
                    k += k_upper;
                }
                k >>= 1;
                sum = (int64_t)*input - (int32_t)(segment[k].xBase & XBASEMASK);
            } while (k_upper > k_lower + 1);
            sum *= segment[k].slope; // prod = diff * slope
            sum >>= (((segment[k].xBase & ~XBASEMASK) + 1) << BIT_SHIFT_SIZE); // prod_shift = prod >> slope_shift
            sum += segment[k].yBase;                   // sum = prod_shift + ybase;
            pwlSaturateStoreOut(sum, output, config->SaturationCount, pwl->bytesPerOutput);
        }
        else
        {
            pwlSaturateStoreOut(pwl->Binary.yBase0, output, config->SaturationCount, pwl->bytesPerOutput);
        }

        input++;
        output += pwl->bytesPerOutput;
    } while (input < inputEnd);
}

#define pwlKernelImplAllLinear KERNEL(pwlKernelImplAllLinear)
void pwlKernelImplAllLinear(ExecutionKernelConfig<ActivationConfig> const * const config)
{
    auto pwl = &config->RequestConfig->Transform.Kernel->pwl;
    int32_t const * input = (int32_t*)config->RequestConfig->Inputs;
    int32_t const * const inputEnd = input + config->RequestConfig->Transform.ElementCount;
    int8_t * output = config->RequestConfig->Outputs;

    int64_t sum;
    nn_pwl_seg * end = pwl->Binary.source - 1;
    nn_pwl_seg * segment;

    do
    {
        sum = (int64_t)*input - pwl->Binary.xBase0;
        if (sum <= 0)
        {
            pwlSaturateStoreOut(pwl->Binary.yBase0, output, config->SaturationCount, pwl->bytesPerOutput);
        }
        else
        {
            segment = end + pwl->segmentCount;
            sum = (int64_t)*input - (int64_t)(segment->xBase & XBASEMASK);
            while (sum < 0 && --segment > end)
            {
                sum = (int64_t)*input - (int64_t)(segment->xBase & XBASEMASK);
            }
            sum *= segment->slope; // prod = diff * slope
            sum >>= (((segment->xBase & ~XBASEMASK) + 1) << BIT_SHIFT_SIZE); // prod_shift = prod >> slope_shift
            sum += segment->yBase;                   // sum = prod_shift + ybase;
            pwlSaturateStoreOut(sum, output, config->SaturationCount, pwl->bytesPerOutput);
        }

        input++;
        output += pwl->bytesPerOutput;
    } while (input < inputEnd);
}

#define pwlKernelImplSingleLinear KERNEL(pwlKernelImplSingleLinear)
void pwlKernelImplSingleLinear(PwlCachedConfig const * const pwl, int32_t I, int16_t* O,
    uint32_t * const saturationCount)
{
    int64_t     sum;
    nn_pwl_seg const * segment;
    nn_pwl_seg* end = pwl->Binary.source - 1;

    sum = (int64_t)I - pwl->Binary.xBase0;
    if (sum <= 0)
    {
        pwlSaturateStoreOut(pwl->Binary.yBase0, (int8_t*)O, saturationCount, pwl->bytesPerOutput);
    }
    else
    {
        segment = end + pwl->segmentCount;
        sum = (int64_t)I - (int64_t)(segment->xBase & XBASEMASK);
        while (sum < 0 && --segment > end)
        {
            sum = (int64_t)I - (int64_t)(segment->xBase & XBASEMASK);
        }

        sum *= segment->slope; // prod = diff * slope
        sum >>= (((segment->xBase & ~XBASEMASK) + 1) << BIT_SHIFT_SIZE); // prod_shift = prod >> slope_shift
        sum += segment->yBase;                   // sum = prod_shift + ybase;
        pwlSaturateStoreOut(sum, (int8_t*)O, saturationCount, pwl->bytesPerOutput);
    }
}

#define pwlKernelImplSingleBinaryOpt KERNEL(pwlKernelImplSingleBinaryOpt)
void pwlKernelImplSingleBinaryOpt(PwlCachedConfig const * const pwl, int32_t I, int16_t * const O,
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
        xBase = (pwl_x_t*)pwl->data + k;
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
        pwlSaturateStoreOut(sum, (int8_t*)O, saturationCount, pwl->bytesPerOutput);
    }
    else
    {
        pwlSaturateStoreOut(pwl->Binary.yBase0, (int8_t*)O, saturationCount, pwl->bytesPerOutput);
    }
}

#define pwlKernelImplAllBinaryOpt KERNEL(pwlKernelImplAllBinaryOpt)
void pwlKernelImplAllBinaryOpt(ExecutionKernelConfig<ActivationConfig> const * const config)
{
    int64_t sum;                    // tmp sum
    const int32_t* input;           // input row
    const int32_t* inputEnd;           // input row
    int8_t* output;                // output row
    auto pwl = &config->RequestConfig->Transform.Kernel->pwl;
    pwl_x_t* xBase;
    pwl_x_t * const xBaseReset = (pwl_x_t*)pwl->data + (pwl->segmentCount >> 1);;
    pwl_y_t* seg;
    uint32_t k;
    uint32_t k_upper;
    uint32_t k_lower;

    // input and sum prefetch
    input = (int32_t*)config->RequestConfig->Inputs;
    inputEnd = input + config->RequestConfig->Transform.ElementCount;
    output = config->RequestConfig->Outputs;
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
                }
                else
                {
                    k_lower = k;
                    k = (k_upper + k) >> 1;
                }
                xBase = (pwl_x_t*)((int64_t)xBase & XBASE_ADDRESS_RESET);
                xBase += k;
                sum = (int64_t)*input + *xBase;
            } while (k_upper > k_lower + 1);
            seg = (pwl_y_t*)(xBase + pwl->segmentCount);
            sum *= seg->slope; // prod = diff * slope
            sum = sum >> seg->shift; // prod_shift = prod >> slope_shift
            sum += seg->yBase;                   // sum = prod_shift + ybase;
            pwlSaturateStoreOut(sum, output, config->SaturationCount, pwl->bytesPerOutput);
        }
        else
        {
            pwlSaturateStoreOut(pwl->Binary.yBase0, output, config->SaturationCount, pwl->bytesPerOutput);
        } output += pwl->bytesPerOutput;
        input++;
    } while (input < inputEnd);
}

#define pwlKernelImplSingleLookup KERNEL(pwlKernelImplSingleLookup)
void pwlKernelImplSingleLookup(PwlCachedConfig const * const pwl, int32_t I, int16_t * const O,
    uint32_t * const saturationCount)
{
    int64_t k;                      // lookup table iterator and helper
    int64_t sum;                    // tmp sum
    pwl_u_t* lookup = (pwl_u_t*)pwl->data;  // lookup table

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
        pwlSaturateStoreOut(sum, (int8_t*)O, saturationCount, pwl->bytesPerOutput);
    }
    else
    {
        pwlSaturateStoreOut(pwl->Lookup.yBase0, (int8_t*)O, saturationCount, pwl->bytesPerOutput);
    }
}

#define pwlKernelImplAllLookup KERNEL(pwlKernelImplAllLookup)
void pwlKernelImplAllLookup(ExecutionKernelConfig<ActivationConfig> const * const config)
{
    int64_t k;                      // lookup table iterator and helper
    int64_t sum;                    // tmp sum
    const int32_t* input;           // input row
    const int32_t* inputEnd;           // input row
    int8_t* output;                // output row
    auto pwl = &config->RequestConfig->Transform.Kernel->pwl;
    pwl_u_t* lookup = (pwl_u_t*)pwl->data;  // lookup table
    pwl_x_t xBase0 = pwl->Lookup.xBase0Neg;
    pwl_x_t xBase1diff = pwl->Lookup.xBase1diff;
    int32_t count = pwl->Lookup.count;

    // input and k prefetch
    input = (int32_t*)config->RequestConfig->Inputs;
    inputEnd = input + config->RequestConfig->Transform.ElementCount;
    output = (int8_t*)config->RequestConfig->Outputs;
    do
    {
        lookup = (pwl_u_t*)pwl->data;
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
            pwlSaturateStoreOut(sum, output, config->SaturationCount, pwl->bytesPerOutput);
        }
        else
        {
            pwlSaturateStoreOut(pwl->Lookup.yBase0, (int8_t*)output, config->SaturationCount, pwl->bytesPerOutput);
        }
        output += pwl->bytesPerOutput;
        input++;
    } while (input < inputEnd);
}

void PwlCached::KERNEL(InitializeActivationFunctions)() const
{
    if (useLookup)
    {
        ActivateAll = pwlKernelImplAllLookup;
        ActivateSingle = pwlKernelImplSingleLookup;
    }
    else
    {
        if (pwl.segmentCount > PWL_SIZE_OPT_ALGORITHM_TRESHOLD)
        {
            ActivateAll = pwlKernelImplAllBinaryOpt;
            ActivateSingle = pwlKernelImplSingleBinaryOpt;
        }
        else if (pwl.segmentCount > PWL_SIZE_ALGORITHM_TRESHOLD)
        {
            ActivateAll = pwlKernelImplAllBinary;
            ActivateSingle = pwlKernelImplSingleBinary;
        }
        else
        {
            ActivateAll = pwlKernelImplAllLinear;
            ActivateSingle = pwlKernelImplSingleLinear;
        }
    }

}

#if OPT_LEVEL == 0
PwlCached::PwlCached(const gna_data_mode mode, nn_pwl_seg const * const segments, uint32_t segmentCountIn)
{
    // TODO:3: enable different modes
    uint32_t s = 0;                    // PWL segment iterator
    uint32_t i;                        // pwl.lookup element offset iterator (beginning)
    int64_t j;                         // pwl.lookup element offset iterator (end)
    pwl_x_t xBaseAtmp;                 // left segment xBase value (extracted)
    pwl_x_t xBaseBtmp;                 // right segment x Base value (extracted)
    int64_t widthTmp = UINT32_MAX;     // pwl.lookup segment widthTmp - minimum distance between pwl.segments' xbases
    int64_t countTmp = 0;              // pwl.lookup segment countTmp (active)
    pwl_s_t usegTmp;
    ActivateAll = NULL;
    ActivateSingle = NULL;
    pwl.segmentCount = segmentCountIn;
    pwl.bytesPerOutput = mode;

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
            s = pwlFindFirstBitSet(widthTmp);
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
        pwl.Lookup.width = (int8_t)s;
        pwl.Lookup.count = (uint16_t)countTmp - 1;
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

            pwl_u_t* LookupSegment;
            for (; i < j; i++)
            {
                if (i & 1)
                {
                    LookupSegment = &((pwl_u_t*)pwl.data)[(i & ~1) / 2];
                    LookupSegment->xBaseB = usegTmp.xBase;
                    LookupSegment->slopeB = usegTmp.slope;
                    LookupSegment->shiftB = usegTmp.shift;
                    LookupSegment->yBaseB = usegTmp.yBase;
                }
                else
                {
                    LookupSegment = &((pwl_u_t*)pwl.data)[i / 2];
                    LookupSegment->xBaseA = usegTmp.xBase;
                    LookupSegment->slopeA = usegTmp.slope;
                    LookupSegment->shiftA = usegTmp.shift;
                    LookupSegment->yBaseA = usegTmp.yBase;
                }
            }
            s++;
            xBaseBtmp = segments[s].xBase & XBASEMASK;
        }
        usegTmp.xBase = pwl.Lookup.xBase0 - pwl.Lookup.xBase1diff - (pwl_x_t)(segments[s - 1].xBase & XBASEMASK);
        usegTmp.shift = ((segments[s - 1].xBase & ~XBASEMASK) + 1) << BIT_SHIFT_SIZE;
        usegTmp.slope = segments[s - 1].slope;
        usegTmp.resvd = 0;
        usegTmp.yBase = segments[s - 1].yBase;
        pwl_u_t* LookupSegment;
        for (; i < countTmp * PWL_LOOKUP_SEG_SCOUNT; i++)
        {
            if (i & 1)
            {
                LookupSegment = &((pwl_u_t*)pwl.data)[(i & ~1) / 2];
                LookupSegment->xBaseB = usegTmp.xBase;
                LookupSegment->slopeB = usegTmp.slope;
                LookupSegment->shiftB = usegTmp.shift;
                LookupSegment->yBaseB = usegTmp.yBase;
            }
            else
            {
                LookupSegment = &((pwl_u_t*)pwl.data)[i / 2];
                LookupSegment->xBaseA = usegTmp.xBase;
                LookupSegment->slopeA = usegTmp.slope;
                LookupSegment->shiftA = usegTmp.shift;
                LookupSegment->yBaseA = usegTmp.yBase;
            }
        }
        for (i = 0; i < countTmp; i++)
        {
            ((pwl_u_t*)pwl.data)[i].xBaseA = ((pwl_u_t*)pwl.data)[i].xBaseA - ((pwl_u_t*)pwl.data)[i].xBaseB;
        }
    }
    else
    {
        pwl.Binary.source = (nn_pwl_seg*)segments;
        pwl.Binary.xBase0 = segments[0].xBase & XBASEMASK;
        pwl.Binary.yBase0 = segments[0].yBase;

        if (pwl.segmentCount > 32)
        {
            allocateBinaryCaches();
            i = 0;
            for (; i < pwl.segmentCount; i++)
            {
                ((pwl_x_t*)pwl.data)[i] = -1 * (pwl_x_t)(segments[i].xBase & XBASEMASK);
                pwl.Binary.ySeg[i].shift = ((segments[i].xBase & ~XBASEMASK) + 1) << BIT_SHIFT_SIZE;
                pwl.Binary.ySeg[i].slope = segments[i].slope;
                pwl.Binary.ySeg[i].resvd = 0;
                pwl.Binary.ySeg[i].yBase = segments[i].yBase;
            }
        }
        else
            pwl.data = nullptr;
    }
}

PwlCached::~PwlCached()
{
    if (nullptr != pwl.data)
    {
        _gna_free(pwl.data);
        memset(&pwl, 0, sizeof(pwl));
    }
}

void PwlCached::allocateBinaryCaches()
{
    auto totalSize = pwl.segmentCount * (sizeof(pwl_x_t) + sizeof(pwl_y_t));
    pwl.data = _gna_malloc(totalSize);
    if (nullptr == pwl.data)
    {
        throw GnaException(GNA_ERR_RESOURCES);
    }
    memset(pwl.data, 0, totalSize);
    pwl.Binary.ySeg = (pwl_y_t*)((pwl_x_t*)pwl.data + pwl.segmentCount);
}

void PwlCached::allocateLookupCaches()
{
    pwl.data = _gna_malloc(PWL_LOOKUP_SIZE);
    if (nullptr == pwl.data)
    {
        throw GnaException(GNA_ERR_RESOURCES);
    }
    memset(pwl.data, 0xff, PWL_LOOKUP_SIZE);
}
#endif
