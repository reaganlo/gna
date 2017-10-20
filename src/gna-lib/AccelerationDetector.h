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
#include <string>

#include "common.h"
#include "IoctlSender.h"
#if WINDOWS == 1
#include "WindowsIoctlSender.h"
#else // LINUX

#endif
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
    NewPerformanceCounters,

    GnaFeatureCount
};

/**
 * Manages runtime acceleration modes
 * and configures execution kernels for given acceleration
 */
class AccelerationDetector
{

public:
    AccelerationDetector(IoctlSender &senderIn);
    ~AccelerationDetector() = default;

    acceleration AccelerationDetector::GetFastestAcceleration() const;

    static char const * const AccelerationToString(acceleration accel);

    bool IsHardwarePresent() const;

    bool IsLayerSupported(intel_layer_kind_t layerType) const;

    bool HasFeature(GnaFeature feature) const;

    const uint32_t GetHardwareBufferSize() const;

    template<typename T>
    static const std::map<const acceleration, const T>& GetKernelMap(WeightMode weightMode, nn_layer_kind layerKind);

    template<typename T>
    static const std::map<const acceleration, const T>& GetKernelMap(WeightMode weightMode);

    template<typename T>
    static const std::map<const acceleration, const T>& GetKernelMap();

    template<typename T>
    static const std::map<const acceleration, const T>& GetKernelMap(gna_gmm_mode);

    void UpdateKernelsMap();

    static std::map<const WeightMode, std::map<const acceleration, const AffineKernel>> AffineKernels;
    static std::map<const WeightMode, std::map<const acceleration, const AffineActiveListKernel>> AccelerationDetector::AffineKernelsAl;
    static std::map<const WeightMode, std::map<const acceleration, const AffineKernel>> MultibiasKernels;
    static std::map<const WeightMode, std::map<const acceleration, const RecurrentKernel>> RecurrentKernels;
    static std::map<const WeightMode, std::map<const acceleration, const AffineKernel>> AccelerationDetector::DiagonalKernels;

    static std::map<const acceleration, const TransposeKernel> TransposeKernels;
    static std::map<const acceleration, const CopyKernel> AccelerationDetector::CopyKernels;
    static std::map<const acceleration, const ConvolutionKernel> ConvolutionKernels;
    static std::map<const acceleration, const ConvolutionPoolingKernel> AccelerationDetector::PoolingKernels;
    static std::map<const acceleration, const PwlKernel> PwlKernels;

    static std::map<const gna_gmm_mode, std::map<const acceleration, const GmmMaxMix>> AccelerationDetector::GmmKernels;
    static std::map<const gna_gmm_mode, std::map<const acceleration, const GmmMaxMixActiveList>> GmmActiveListKernels;

protected:
    static const std::map<GnaDeviceType, std::array<bool, GnaFeatureCount>> gnaFeatureMap;

    std::map<acceleration, uint8_t> accelerationModes;

    GNA_CPBLTS deviceCapabilities;

    acceleration fastestAcceleration;

private:
    static std::map<acceleration const, std::string const> accelerationNames;

    void discoverHardwareExistence();

    void discoverHardwareCapabilities();

    IoctlSender &ioctlSender;
};

}
