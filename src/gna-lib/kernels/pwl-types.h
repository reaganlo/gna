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

#include "KernelArguments.h"

/**
* PWL Segment x base type
*/
typedef int64_t pwl_x_t;

/**
* PWL Unpacked double-segment for lookup table entry
*/
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

/**
* PWL Unpacked single-segment, auxiliary for lookup preparation
*/
typedef struct
{
    pwl_x_t     xBase;
    int16_t     slope;
    int16_t     shift;
    int16_t     resvd;
    int16_t     yBase;
} pwl_s_t;

static_assert(16 == sizeof(pwl_s_t), "Invalid size of pwl_s_t");

/**
* PWL Unpacked segment values, for split PWL segment and binary search
*/
typedef struct __pwl_y
{
    int16_t     slope;
    int16_t     shift;
    int16_t     resvd;
    int16_t     yBase;
} pwl_y_t;

// PWL cache and config (constant for given layer)
struct __PwlCached
{
    pwl_x_t         xBase0Lu;               // first segment xBase value (Lookup algorithm)
    pwl_x_t         xBase0Neg;              // first segment xBase value x -1 for addition only  (Lookup algorithm)
    pwl_x_t         xBase1diff;             // difference between first and second PWL segment xBases, for lookup
    pwl_x_t         xBase0Bi;               // first segment xBase value (binary search algorithm)

    int16_t         slope0;                 // first segment slope value (Lookup algorithm)
    int16_t         shift0;                 // first segment extracted shift value (Lookup algorithm)
    int16_t         yBase0Lu;               // first segment yBase value (Lookup algorithm)
    int16_t         yBase0Bi;               // first segment yBase value (binary search algorithm)
    PwlBaseConfig   config;
    uint32_t        _reserved1;             // padding
    uint16_t        count;                  // number of lookup segments (active)
    uint8_t         width;                  // padding
    uint8_t         _reserved2;
    pwl_u_t*        lookup;                 // lookup table data
    pwl_x_t*        xBase;                  // extracted PWL segments xBase data
    pwl_y_t*        ySeg;                   // extracted PWL segments value data
    const nn_pwl_seg* prevLu;                 // previous PWL segments data for lookup table
    const nn_pwl_seg* prevBi;                 // previous PWL segments data for binary search
    PwlApplySingle  pwlSingle;              // algorithm used for PWL for single in-out
    PwlApplyAll     pwlAll;                 // algorithm used for PWL for all in-outs
};

static_assert(8 == sizeof(pwl_y_t), "Invalid size of pwl_y_t");

/**
* PWL LOOKUP table number of elements
*/
const int32_t PWL_LOOKUP_COUNT = 1024;

/**
* PWL LOOKUP table element size in B
*/
const int32_t PWL_LOOKUP_SEG_SIZE = sizeof(pwl_u_t);

/**
* PWL LOOKUP table - number of segments per element
*/
const int32_t PWL_LOOKUP_SEG_SCOUNT = 2;

/**
* PWL LOOKUP table size in B
*/
const int32_t PWL_LOOKUP_SIZE = (PWL_LOOKUP_COUNT) * PWL_LOOKUP_SEG_SIZE;

/**
* Size of additional buffer for unpacked PWL cache
*/
const int32_t PWL_PARAMS_BUFFER_SIZE = ALIGN64(sizeof(PwlCached));

/**
* PWL xBase buffer size in bytes
*/
const int32_t PWL_X_BUFFER_SIZE = sizeof(pwl_x_t) * XNN_N_PWL_SEGS_MAX;

/**
* PWL Unpacked segment values
*/
const int32_t PWL_Y_BUFFER_SIZE = sizeof(pwl_y_t) * XNN_N_PWL_SEGS_MAX;
