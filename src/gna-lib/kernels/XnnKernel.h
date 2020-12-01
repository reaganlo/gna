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

typedef void (*AffineKernel)(ExecutionKernelConfig<AffineConfig> const * const config);

typedef void (*AffineActiveListKernel)(
        ExecutionKernelConfig<AffineConfig> const * const config, AffineConfigAl al);

typedef void (*ActivationKernel)(ExecutionKernelConfig<ActivationConfig> const * const config);

typedef void (*RecurrentKernel)(ExecutionKernelConfig<RecurrentConfig> const * const config);

typedef void (*ConvolutionKernel)(ConvolutionConfig const * const config);

typedef void (*ConvolutionKernel2D)(ExecutionKernelConfig<ConvolutionConfig2D> const * const config);

typedef void (*PoolingKernel2D)(ExecutionKernelConfig<PoolingConfig2D> const * const config);

typedef void (*ConvolutionPoolingKernel)(ConvolutionConfig const * const filterConfig,
    PoolingConfig const * const poolConfig, PwlCached const * const pwl);

typedef void (*TransposeKernel)(TransposeConfig const * const config);

typedef void (*CopyKernel)(CopyConfig const * const config);

enum KernelType
{
    affineSingle1Bfull,
    affineSingle2Bfull,
    affineSingle1Bal,
    affineSingle2Bal,
    affineMulti1B,
    affineMulti2B,
    diagonal1B,
    diagonal2B,
    recurrent1B,
    recurrent2B,
    convolution,
    convolutionPooling,
    pwl,
    transpose,
    copy,
    affineSingle1B1Bfull,
    affineSingle2B1Bfull,
    affineSingle1B2Bfull,
    affineSingle2B2Bfull,
    affineSingle1B1Bal,
    affineSingle2B1Bal,
    affineSingle1B2Bal,
    affineSingle2B2Bal,
    affineMulti1B1B,
    affineMulti2B1B,
    affineMulti1B2B,
    affineMulti2B2B,
    diagonal1B1B,
    diagonal2B1B,
    diagonal1B2B,
    diagonal2B2B,
    recurrent1B1B,
    recurrent2B1B,
    recurrent1B2B,
    recurrent2B2B,
    convolution1B,
    convolutionPooling1B,
    convolution2B,
    convolutionPooling2B,
    transpose1B,
    transpose2B,
    copy1B,
    copy2B,
    convolution2D1B1B,
    convolution2D1B2B,
    convolution2D2B1B,
    convolution2D2B2B,
    convolutionPooling2D1B,
    convolutionPooling2D2B,
    convolutionPooling2D4B,
};

template<Gna2AccelerationMode accelerationMode, bool hardwareConsistencyEnabled>
VoidKernel GetXnnKernel(KernelType type);

void setHwCompatibilityMode_generic(uint32_t bufferElementCounts[2][XNN_N_GROUP_MAX]);
void setHwCompatibilityMode_sse4(uint32_t bufferElementCounts[2][XNN_N_GROUP_MAX]);
void setHwCompatibilityMode_avx1(uint32_t bufferElementCounts[2][XNN_N_GROUP_MAX]);
void setHwCompatibilityMode_avx2(uint32_t bufferElementCounts[2][XNN_N_GROUP_MAX]);
void setHwCompatibilityMode_generic_sat(uint32_t bufferElementCounts[2][XNN_N_GROUP_MAX]);
void setHwCompatibilityMode_sse4_sat(uint32_t bufferElementCounts[2][XNN_N_GROUP_MAX]);
void setHwCompatibilityMode_avx1_sat(uint32_t bufferElementCounts[2][XNN_N_GROUP_MAX]);
void setHwCompatibilityMode_avx2_sat(uint32_t bufferElementCounts[2][XNN_N_GROUP_MAX]);

}
