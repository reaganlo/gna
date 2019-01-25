/*
 INTEL CONFIDENTIAL
 Copyright 2018 Intel Corporation.

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
#include <unordered_map>
#include <string>

#include "common.h"
#if defined(_WIN32)
#include "WindowsIoctlSender.h"
#else
#include "LinuxIoctlSender.h"
#endif
#include "gmm.h"
#include "XnnKernelApi.h"

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
    CNN2D,

    GnaFeatureCount
};

typedef enum _nn_bias_mode
{
    GNA_BIAS_MODE_NOT_SUPPORTED = GNA_NOT_SUPPORTED,
    GNA_BIAS_MODE_1_2_4B = GNA_INT8,                         // 1, 2 or 4B per bias, used for kernel selection
    GNA_BIAS_MODE_RICH_FORMAT = GNA_DATA_RICH_FORMAT,           // 8B Rich bias intel_compound_bias_t data is used, only with GNA_INT8 weight mode.
    GNA_BIAS_MODE_CONSTANT_SCALAR = GNA_DATA_CONSTANT_SCALAR,   // Single 4B (GNA_INT32) signed integer scalar is used instead of tensor.
    GNA_BIAS_MODE_DISABLED = GNA_DATA_DISABLED,                 // No data is read
} nn_bias_mode;

struct KernelMode
{
    KernelMode(gna_data_mode input, gna_data_mode weight, nn_bias_mode bias) :
        Input{ input },
        Weight{ weight },
        Bias{ bias }
    {}

    KernelMode(gna_data_mode input, gna_data_mode weight, gna_data_mode bias) :
        Input{ input },
        Weight{ weight },
        Bias{ translateBias(bias) }
    {}

    KernelMode(gna_data_mode input) :
        Input{ input },
        Weight{ GNA_INT8 },
        Bias{ GNA_BIAS_MODE_1_2_4B }
    {}

    KernelMode(gna_gmm_mode gmmMode) :
        KernelMode{GNA_INT16, _data_mode(gmmMode + 1), GNA_BIAS_MODE_1_2_4B}
    {}

    ~KernelMode() = default;
    nn_bias_mode translateBias(gna_data_mode bias)
    {
        switch (bias)
        {
        case GNA_INT8:
        case GNA_INT16:
        case GNA_INT32:
            return GNA_BIAS_MODE_1_2_4B;
        default:
            return static_cast<nn_bias_mode>(bias);
        }
    }
    bool operator==(const KernelMode &mode) const
    {
        return mode.Input == Input && mode.Weight == Weight &&
            mode.Bias == Bias;
    }

    bool operator<(const KernelMode &mode) const
    {
        if (mode.Input != Input)
            return mode.Input < Input;
        if (mode.Weight != Weight)
            return mode.Weight < Weight;
        return mode.Bias < Bias;
    }

    const gna_data_mode Input;
    const gna_data_mode Weight;
    const nn_bias_mode Bias;
};

typedef enum _kernel_op
{
    KERNEL_AFFINE = INTEL_AFFINE,
    KERNEL_AFFINE_DIAGONAL = INTEL_AFFINE_DIAGONAL,
    KERNEL_AFFINE_MULTIBIAS = INTEL_AFFINE_MULTIBIAS,
    KERNEL_CONVOLUTIONAL = INTEL_CONVOLUTIONAL,
    KERNEL_COPY = INTEL_COPY,
    KERNEL_TRANSPOSE = INTEL_DEINTERLEAVE,
    KERNEL_GMM = INTEL_GMM,
    KERNEL_RECURRENT = INTEL_RECURRENT,
    KERNEL_CONVOLUTIONAL_2D = INTEL_CONVOLUTIONAL_2D,
    KERNEL_CNN_2D_ADDITION = GNA_LAYER_CNN_2D_ADDITION,
    KERNEL_CNN_2D_CONVERSION = GNA_LAYER_CNN_2D_CONVERSION,
    KERNEL_POOLING_2D = GNA_LAYER_CNN_2D_POOLING,
    KERNEL_POOLING,
    KERNEL_PWL,
    KERNEL_AFFINE_AL,
    KERNEL_GMM_AL,

} kernel_op;

struct GnaHardwareCapabiities
{
    gna_device_generation Generation;
    // Basic, CNN,   GMM,  GMMLayer, MultiBias, L1Dist, L2Dist, ComputerVision, Layer8K, NewPerformanceCounters
    std::array<bool, GnaFeatureCount> Features;
    uint32_t ComputeEngineCount;
    std::map<const uint32_t /* input precision */, const uint32_t> MacCountPerCE;
    uint32_t BufferSizesPerCEInKB;
    uint32_t PoolingEngineCountPerCE;
    uint32_t ActivationEngineCount;
    std::array<uint32_t, XNN_N_GROUP_MAX> BufferElementCountBackward;
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

    static void GetHardwareConsistencySettings(uint32_t bufferElementCount[2 * XNN_N_GROUP_MAX],
        gna_device_version hwId);

    AccelerationMode GetFastestAcceleration() const;

    static char const * AccelerationToString(AccelerationMode accel);

    static gna_device_version GetDeviceVersion(gna_device_generation generation);

    static uint32_t GetComputeEngineCount(gna_device_version hwId);

    static uint32_t GetBufferSizeInKB(gna_device_version hwId);
 
    static inline uint32_t GetBufferSizeInKB(gna_device_generation generation)
    {
        return GetBufferSizeInKB(GetDeviceVersion(generation));
    }

    // Gets the number of data elements that may be stored in hw buffer
    static uint32_t GetBufferElementCount(gna_device_version hwId,
            uint32_t grouping, uint32_t inputPrecision = GNA_INT16);

    bool IsHardwarePresent() const;

    bool IsLayerSupported(nn_operation operation) const;

    bool HasFeature(GnaFeature feature) const;

    uint32_t GetBufferElementCount(uint32_t grouping, uint32_t inputPrecision = GNA_INT16) const
    {
        return GetBufferElementCount(deviceCapabilities.hwId, grouping, inputPrecision);
    }

    gna_device_version GetDeviceVersion() const;

    template<typename KernelType>
    static const KernelMap<KernelType>&
    GetKernelMap(kernel_op operation, KernelMode dataMode = {GNA_INT16})
    {
        return (KernelMap<KernelType>&)(
            Kernels.at(operation).at(dataMode));
    }

    static std::map<kernel_op, std::map<KernelMode, KernelMap<VoidKernel>>> Kernels;

protected:
    static std::map<gna_device_version, const GnaHardwareCapabiities> gnaCapsMap;

    std::map<AccelerationMode, uint8_t> accelerationModes;

    GnaCapabilities deviceCapabilities;

    AccelerationMode fastestAcceleration;

private:
    static std::map<AccelerationMode const, std::string const> accelerationNames;

    void discoverHardware();

    IoctlSender &ioctlSender;
};

}
