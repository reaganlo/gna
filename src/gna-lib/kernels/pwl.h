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

#include "KernelMacros.h"
#include "common.h"
#include "pwl-types.h"

/**
 * Mask for retrieving PWL segment xBase value
 */
const int32_t XBASEMASK = 0xFFFFFFFC;

/**
 * Number of segments above which lookup algorithm is used when possible
 * otherwise binary search is used
 */
const int32_t PWL_SIZE_ALGORITHM_TRESHOLD = 3;

/**
 * Pads value
 */
#define PADD(value, pad)   ((((value) + pad -1) / pad) * pad)

/**
 * Kernel-names macros for PWL functions
 */
#define PwlApplySingleLookup    KERNEL(PwlApplySingleLookup)
#define PwlApplyAllLookup       KERNEL(PwlApplyAllLookup)
#define PwlApplySingleBinary    KERNEL(PwlApplySingleBinary)
#define PwlApplyAllBinary       KERNEL(PwlApplyAllBinary)
#define PwlPrepareAuxBuffers    KERNEL(PwlPrepareAuxBuffers)

#ifdef __cplusplus
extern "C" {  // API uses C linkage so that it can be used by C and C++ applications
#endif

/**
 * Calculates ApplyPicewiseLinearSegment using lookup table
 *
 * @params  Parameters for PWL function
 * @I       input value
 * @O       pointer to output value
 */
void
PwlApplySingleLookup(
    pwl_params*     params,
    int32_t         I,
    int16_t*        O);

/**
 * Calculates ApplyPicewiseLinearSegment using binary search
 *
 * @params  Parameters for PWL function
 * @I       input value
 * @O       pointer to output value
 */
void
PwlApplySingleBinary(
    pwl_params*     params,
    int32_t         I,
    int16_t*        O);

/**
 * Calculates ApplyPicewiseLinearSegment for all outputs using lookup table
 *
 * @params  Parameters for PWL function
 */
void
PwlApplyAllLookup(
    pwl_params*     params);

/**
 * Calculates ApplyPicewiseLinearSegment for all outputs using binary search
 *
 * @params  Parameters for PWL function
 */
void
PwlApplyAllBinary(
    pwl_params*     params);

/**
 * Calculates PWL parameters and auxiliary buffers
 *
 * @params      PWL parameters output and auxiliary buffers memory
 * @segments    original PWL segments data
 */
void
PwlPrepareAuxBuffers(
    pwl_params*     params,
    nn_pwl_seg*   segments);

/**
 * Prepares PWL parameters and auxiliary buffers
 *
 * @params      PWL parameters output and auxiliary buffers memory
 * @nRowBegin   output row start
 * @nRowEnd     output row end
 * @nColBegin   output column start
 * @nColEnd     output column end
 * @nOutCols    number of output columns
 * @nSaturated  number of saturation
 * @I           input values
 * @O           output values
 * @segments    original PWL segments data
 */
__forceinline
void
PwlSetup(
    pwl_params*     params,
    uint32_t        nRowBegin,
    uint32_t        nRowEnd,
    uint32_t        nColBegin,
    uint32_t        nColEnd,
    uint32_t        nOutCols,
    uint32_t*       nSaturated,
    int32_t*        I,
    int16_t*        O,
    nn_pwl_seg*   segments
)
{
    // store basic parameters
    params->nRowBegin = nRowBegin;
    params->nRowEnd = nRowEnd;
    params->nColBegin = nColBegin;
    params->nColEnd = nColEnd;
    params->nOutCols = nOutCols;
    params->nSaturated = nSaturated;
    params->I = I;
    params->O = O;
    PwlPrepareAuxBuffers(params, segments);
}

#ifdef __cplusplus
}
#endif
