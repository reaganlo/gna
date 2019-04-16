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

#include "gmm.h"
#include "XnnKernel.h"

#include <map>
#include <vector>

namespace GNA
{

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

/**
 * Manages runtime acceleration modes
 * and configures execution kernels for given acceleration
 */
class AccelerationDetector
{

public:
    AccelerationDetector();
    ~AccelerationDetector() = default;

    const std::vector<Gna2AccelerationMode>& GetSupportedCpuAccelerations() const;

    template<typename KernelType>
    static const KernelMap<KernelType>&
    GetKernelMap(kernel_op operation, KernelMode dataMode = {GNA_INT16})
    {
        return (KernelMap<KernelType>&)(
            Kernels.at(operation).at(dataMode));
    }

    void SetHardwareAcceleration(bool isHardwareSupported)
    {
        accelerationModes[AccelerationMode{Gna2AccelerationModeHardware}] = isHardwareSupported;
    }

    static std::map<kernel_op, std::map<KernelMode, KernelMap<VoidKernel>>> Kernels;

protected:
    std::map<AccelerationMode, bool> accelerationModes;

private:
    //sorted from slowest to fastest
    std::vector<Gna2AccelerationMode> supportedCpuAccelerations;
};

}
