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

//*****************************************************************************
// AccelerationDetector.h - declarations for acceleration modes dispatcher
//                 (currently CPU instruction set extensions only)
// Note:   CPU instruction set extensions detection code based on
//          "Intel® Architecture Instruction Set Extensions Programming reference"
//

#pragma once

#define ACC_SUPPORTED 1
#define ACC_NOTSUPPORTED 0

#include <array>
#include <map>

#include "common.h"
#include "IoctlSender.h"
#include "gmm.h"
#include "XnnKernelApi.h"
#include "LayerFunctions.h"

namespace GNA
{

enum GnaFeature
{
    BaseFunctionality = 0, // DNN, DNN_AL, DIAGONAL, RNN, COPY, TRANSPOSE, PWL
    CNN,
    LegacyGMM,
    GMMLayer,
    MultiBias,
    L1Distance,
    L2Distance,
    ComputerVision,
    Layer8K,

    GnaFeatureCount
};

/**
 * Manages runtime acceleration modes
 * and configures execution kernels for given acceleration
 */
class AccelerationDetector : protected IoctlSender
{

public:
    AccelerationDetector();
    ~AccelerationDetector() = default;

    acceleration AccelerationDetector::GetFastestAcceleration() const;

    bool IsHardwarePresent() const;

    bool IsLayerSupported(intel_layer_kind_t layerType) const;

    bool HasFeature(GnaFeature feature) const;

    const uint32_t GetHardwareBufferSize() const;

    // TODO: restrict maps access so unavailable accelerations are not selected
    static const std::map<WeightMode, const std::map<acceleration, AffineKernel>> AffineKernels;

    static const std::map<WeightMode, const std::map<acceleration, AffineActiveListKernel>> AffineKernelsAl;

    static const std::map<WeightMode, const std::map<acceleration, AffineKernel>> MultibiasKernels;

    static const std::map<WeightMode, const std::map<acceleration, AffineActiveListKernel>> MultibiasKernelsAl;

    static const std::map<WeightMode, const std::map<acceleration, DiagonalKernel>> DiagonalKernels;

    static const std::map<WeightMode, std::map<acceleration, RecurrentKernel>> RecurrentKernels;

    static const std::map<gna_gmm_mode, std::map<acceleration, GmmMaxMix>> GmmKernels;

    static const std::map<gna_gmm_mode, std::map<acceleration, GmmMaxMixActiveList>> GmmActiveListKernels;

    static const std::map<acceleration, ConvolutionKernel> ConvolutionalFilterKernels;

    static const std::map<acceleration, ConvolutionPoolingKernel> ConvolutionalFilterPoolKernels;

    static const std::map<acceleration, TransposeKernel> TransposeKernels;

    static const std::map<acceleration, CopyKernel> CopyKernels;

    static const std::map<acceleration, PwlKernel> PwlKernels;

protected:
    static const std::map<GnaDeviceType, std::array<bool, GnaFeatureCount>> gnaFeatureMap;

    std::map<acceleration, uint8_t> accelerationModes;

    GNA_CPBLTS deviceCapabilities;

    acceleration fastestAcceleration;

private:
    void discoverHardwareExistence();

    void discoverHardwareCapabilities();
};

}
