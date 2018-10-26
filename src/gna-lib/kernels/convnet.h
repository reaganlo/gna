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
#include "pwl.h"

#define ConvolutionKernelImpl KERNEL(ConvolutionKernelImpl)
#define ConvolutionPoolingKernelImpl KERNEL(ConvolutionPoolingKernelImpl)
#define MaxPartialPoolingFunction KERNEL(MaxPartialPoolingFunction)
#define SumPartialPoolingFunction KERNEL(SumPartialPoolingFunction)

using GNA::PwlCached;

#ifdef __cplusplus
extern "C" {
#endif

void ConvolutionKernelImpl(ConvolutionConfig const * const filterConfig);

void ConvolutionPoolingKernelImpl(ConvolutionConfig const * const filterConfig,
    PoolingConfig const * const poolConfig, PwlCached const * const pwl);

/* Calculates MaxPartialPoolingFunction
* @PS   number of pool size
* @PNE  number of pool entries
* @PSI  number of pool start index
* @P    pointer to pool array
* @V    pointer to value
*/
void MaxPartialPoolingFunction(const uint32_t PS, const uint32_t PNE, const uint32_t PSI, const int64_t* P, int64_t* V);


/* Calculates SumPartialPoolingFunction
* @PS   number of pool size
* @PNE  number of pool entries
* @PSI  number of pool start index
* @P    pointer to pool array
* @V    pointer to value
*/
void SumPartialPoolingFunction(const uint32_t PS, const uint32_t PNE, const uint32_t PSI, const int64_t* P, int64_t* V);

#ifdef __cplusplus
}
#endif
