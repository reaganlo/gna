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

#pragma once

#include <immintrin.h>
#include <stdint.h>

#if defined(__GNUC__) && !defined(__INTEL_COMPILER)
#define __forceinline inline
#endif

// buffer 24K
const uint32_t hw_buf_size[8] =
{
    12288, 12288, 12096, 12288, 12000, 12096, 12096, 12288
};

// buffer 12K
//const uint32_t hw_buf_size[8] =
//{
//    6144, 6144, 6048, 6144, 5760, 6048, 6048, 6144
//};

// buffer 6K
//const uint32_t hw_buf_size[8] =
//{
//    3072, 3072, 2880, 3072, 2880, 2880, 3024, 3072
//};

__forceinline void saturate(int64_t* const sum, uint32_t * const saturationCount)
{
    if (*sum > INT32_MAX)
    {
        *sum = INT32_MAX;
        (*saturationCount)++;
    }
    else if (*sum < INT32_MIN)
    {
        *sum = INT32_MIN;
        (*saturationCount)++;
    }
}

__forceinline void saturate_store_out(int64_t const * const sum, int32_t * const out, uint32_t * const saturationCount)
{
    if (*sum > INT32_MAX)
    {
        *out = INT32_MAX;
        (*saturationCount)++;
    }
    else if (*sum < INT32_MIN)
    {
        *out = INT32_MIN;
        (*saturationCount)++;
    }
    else
    {
        *out = (int32_t)*sum;
    }
}
