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

#include "common.h"

#include "KernelArguments.h"

// PWL Segment x base type
typedef int64_t pwl_x_t;

// Unpacked double-segment for lookup table entry
typedef struct
{
    pwl_x_t     xBaseA;
    pwl_x_t     xBaseB;
    int16_t     slopeA;
    int16_t     shiftA;
    int16_t     reservedA;
    int16_t     yBaseA;
    int16_t     slopeB;
    int16_t     shiftB;
    int16_t     reservedB;
    int16_t     yBaseB;
} pwl_u_t;

static_assert(32 == sizeof(pwl_u_t), "Invalid size of pwl_u_t");

// PWL Unpacked single-segment, auxiliary for lookup preparation
typedef struct
{
    pwl_x_t     xBase;
    int16_t     slope;
    int16_t     shift;
    int16_t     resvd;
    int16_t     yBase;
} pwl_s_t;

static_assert(16 == sizeof(pwl_s_t), "Invalid size of pwl_s_t");

// PWL Unpacked segment values, for split PWL segment and binary search
typedef struct __pwl_y
{
    int16_t     slope;
    int16_t     shift;
    int16_t     resvd;
    int16_t     yBase;
} pwl_y_t;

static_assert(8 == sizeof(pwl_y_t), "Invalid size of pwl_y_t");

namespace GNA
{

    // PWL cached config (constant for given layer)
    struct PwlCachedConfig
    {
        int32_t const * input;
        uint32_t segmentCount;
        void * data;

        union
        {
            struct __lookup
            {
                pwl_x_t xBase0;             // first segment xBase value (Lookup algorithm)
                pwl_x_t xBase0Neg;          // first segment xBase value x -1 for addition only  (Lookup algorithm)
                pwl_x_t xBase1diff;         // difference between first and second PWL segment xBases, for lookup
                int16_t slope0;             // first segment slope value (Lookup algorithm)
                int16_t shift0;             // first segment extracted shift value (Lookup algorithm)
                int16_t yBase0;             // first segment yBase value (Lookup algorithm)
                uint16_t count;             // number of lookup segments (active)
                uint8_t width;
                uint8_t _reserved[7];       // padding
            } Lookup;
            struct __binary
            {
                nn_pwl_seg* source;         // unpacked segments
                pwl_y_t*  ySeg;             // extracted PWL segments value data
                pwl_x_t xBase0;             // first segment xBase value (binary search algorithm)
                int16_t yBase0;             // first segment yBase value (binary search algorithm)        
                uint8_t _reserved[6];       // padding
            } Binary;
        };
    };
    // Function pointer for apply PWL for single input-output
    typedef void(*PwlApplySingle)(PwlCachedConfig const * const pwl, int32_t I, int16_t * const output,
        uint32_t * const saturationCount);

    // Function pointer for apply PWL for all inputs-outputs
    typedef void(*PwlApplyAll)(PwlCachedConfig const * const pwl, PwlOutputConfig const * const outputConfig);

    // PWL cache and config (constant for given layer)
    struct PwlCached
    {
        // Prepares PWL parameters and auxiliary buffers
        PwlCached(int32_t const * const inputIn, uint32_t elementsCount, nn_pwl_seg const * const segmentsIn, uint32_t segmentCountIn);
        virtual ~PwlCached();

        // PWL LOOKUP table number of elements
        static const int32_t PWL_LOOKUP_COUNT = 1024;

        // PWL LOOKUP table element size in B
        static const int32_t PWL_LOOKUP_SEG_SIZE = sizeof(pwl_u_t);

        // PWL LOOKUP table - number of segments per element
        static const int32_t PWL_LOOKUP_SEG_SCOUNT = 2;

        // PWL LOOKUP table size in B
        static const int32_t PWL_LOOKUP_SIZE = (PWL_LOOKUP_COUNT)* PWL_LOOKUP_SEG_SIZE;

        PwlCachedConfig pwl;
        PwlApplySingle  ActivateSingle;              // algorithm used for PWL for single in-out
        PwlApplyAll     ActivateAll;                 // algorithm used for PWL for all in-outs

    private:
        void allocateLookupCaches();
        void allocateBinaryCaches();
    };

}
