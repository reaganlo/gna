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
#include "ConvolutionKernelArguments.h"
#include "PoolingKernelArguments.h"
#include "pwl.h"

#include "../gna-api/gna2-inference-impl.h"

#include <map>

struct ActivationConfig;
struct AffineConfig;
struct AffineConfigAl;
struct ConvolutionConfig2D;
struct ConvolutionConfig;
struct CopyConfig;
struct PoolingConfig2D;
struct PoolingConfig;
struct RecurrentConfig;
struct TransposeConfig;
template <typename TransformConfig> struct ExecutionKernelConfig;

namespace GNA
{
struct PwlCached;

// TODO:3: move to layer/functions base header
template<typename KernelType>
using KernelMap = std::map<AccelerationMode, KernelType>;

typedef void (*VoidKernel)();

typedef void (*AffineKernel)(AffineConfig const * const config);

typedef void (*AffineActiveListKernel)(AffineConfig const * const config, AffineConfigAl const * const al);

typedef void (*ActivationKernel)(ExecutionKernelConfig<ActivationConfig> const * const config);

typedef void (*RecurrentKernel)(RecurrentConfig const * const config);

typedef void (*ConvolutionKernel)(ConvolutionConfig const * const config);

typedef void (*ConvolutionKernel2D)(ExecutionKernelConfig<ConvolutionConfig2D> const * const config);

typedef void (*PoolingKernel2D)(ExecutionKernelConfig<PoolingConfig2D> const * const config);

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

    ActivationKernel pwl;

    TransposeKernel transpose;

    CopyKernel copy;

    AffineKernel affineSingle1B1Bfull;
    AffineKernel affineSingle2B1Bfull;
    AffineKernel affineSingle1B2Bfull;
    AffineKernel affineSingle2B2Bfull;
    AffineActiveListKernel affineSingle1B1Bal;
    AffineActiveListKernel affineSingle2B1Bal;
    AffineActiveListKernel affineSingle1B2Bal;
    AffineActiveListKernel affineSingle2B2Bal;
    AffineKernel affineMulti1B1B;
    AffineKernel affineMulti2B1B;
    AffineKernel affineMulti1B2B;
    AffineKernel affineMulti2B2B;
    AffineKernel diagonal1B1B;
    AffineKernel diagonal2B1B;
    AffineKernel diagonal1B2B;
    AffineKernel diagonal2B2B;
    RecurrentKernel recurrent1B1B;
    RecurrentKernel recurrent2B1B;
    RecurrentKernel recurrent1B2B;
    RecurrentKernel recurrent2B2B;
    ConvolutionKernel convolution1B;
    ConvolutionPoolingKernel convolutionPooling1B; // TODO: split pwl and pooling from conv. kernel in next phase
    ConvolutionKernel convolution2B;
    ConvolutionPoolingKernel convolutionPooling2B; // TODO: split pwl and pooling from conv. kernel in next phase
    TransposeKernel transpose1B;
    TransposeKernel transpose2B;
    CopyKernel copy1B;
    CopyKernel copy2B;

    ConvolutionKernel2D convolution2D1B1B;
    ConvolutionKernel2D convolution2D1B2B;
    ConvolutionKernel2D convolution2D2B1B;
    ConvolutionKernel2D convolution2D2B2B;

    PoolingKernel2D convolutionPooling2D1B;
    PoolingKernel2D convolutionPooling2D2B;
    PoolingKernel2D convolutionPooling2D4B;
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

void setHwCompatibilityMode_generic(uint32_t bufferElementCounts[2][XNN_N_GROUP_MAX]);
void setHwCompatibilityMode_sse4(uint32_t bufferElementCounts[2][XNN_N_GROUP_MAX]);
void setHwCompatibilityMode_avx1(uint32_t bufferElementCounts[2][XNN_N_GROUP_MAX]);
void setHwCompatibilityMode_avx2(uint32_t bufferElementCounts[2][XNN_N_GROUP_MAX]);
void setHwCompatibilityMode_generic_sat(uint32_t bufferElementCounts[2][XNN_N_GROUP_MAX]);
void setHwCompatibilityMode_sse4_sat(uint32_t bufferElementCounts[2][XNN_N_GROUP_MAX]);
void setHwCompatibilityMode_avx1_sat(uint32_t bufferElementCounts[2][XNN_N_GROUP_MAX]);
void setHwCompatibilityMode_avx2_sat(uint32_t bufferElementCounts[2][XNN_N_GROUP_MAX]);

}
