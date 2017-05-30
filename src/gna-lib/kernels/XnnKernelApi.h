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
#include "pwl.h"

namespace GNA
{

typedef void (*AffineKernel)(AffineConfig const * const config);

typedef void (*AffineActiveListKernel)(AffineConfig const * const config, AffineConfigAl const * const al);

typedef void (*PwlKernel)(PwlCached const * const pwl, PwlOutputConfig const * const outputConfig);

typedef void (*RecurrentKernel)(RecurrentConfig const * const config, PwlCached const * const pwl);

typedef void (*ConvolutionKernel)(ConvolutionConfig const * const config);

typedef void (*ConvolutionPoolingKernel)(ConvolutionConfig const * const filterConfig, 
    PoolingConfig const * const poolConfig, PwlCached const * const pwl);

typedef void (*TransposeKernel)(TransposeConfig const * const config);

typedef void (*CopyKernel)(CopyConfig const * const config);

// Xnn kernel provider
// Contains XNN kernel function pointers for selected acceleration
typedef struct _XnnKernel
{
    AffineKernel affineSingle1Bfull;
    AffineKernel affineSingle2Bfull;
    AffineActiveListKernel affineSingle1Bal;
    AffineActiveListKernel affineSingle2Bal;

    AffineKernel affineMulti1B;
    AffineKernel affineMulti2B;

    AffineKernel diagonal1B;
    AffineKernel diagonal2B;
    
    RecurrentKernel recurrent1B;
    RecurrentKernel recurrent2B;
    
    ConvolutionKernel convolution;
    ConvolutionPoolingKernel convolutionPooling; // TODO: split pwl and pooling from conv. kernel in next phase

    PwlKernel pwl;

    TransposeKernel transpose;

    CopyKernel copy;

} XnnKernel;

// Export list of available Xnn kernels providers

// FAST VERSIONS
// generic Xnn kernel provider
extern XnnKernel xnnKernel_generic;

// sse4.2 accelerated Xnn kernel provider
extern XnnKernel xnnKernel_sse4;

// avx1 accelerated Xnn kernel provider
extern XnnKernel xnnKernel_avx1;

// avx2 accelerated Xnn kernel provider
extern XnnKernel xnnKernel_avx2;

// SATURATED VERSIONS
// generic Xnn kernel provider
extern XnnKernel xnnKernel_generic_sat;

// sse4.2 accelerated Xnn kernel provider
extern XnnKernel xnnKernel_sse4_sat;

// avx1 accelerated Xnn kernel provider
extern XnnKernel xnnKernel_avx1_sat;

// avx2 accelerated Xnn kernel provider
extern XnnKernel xnnKernel_avx2_sat;

}
