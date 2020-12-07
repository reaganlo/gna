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

#include "XnnKernel.h"

#include "DataMode.h"
#include "Expect.h"
#include "ModelError.h"

#include "gna2-inference-api.h"
#include "gna2-inference-impl.h"

#include <map>
#include <vector>

namespace GNA
{
struct KernelMode
{
    constexpr KernelMode(const DataMode & input, const DataMode & weight, const DataMode & bias) :
        KernelMode{ input.Type, weight.Type, bias.Type }
    {}

    constexpr KernelMode(const DataMode & input) :
        KernelMode{ input.Type }
    {}

    constexpr KernelMode(DataType input, DataType weight, DataType bias) :
        Value{ static_cast<DataType>((input << 16) | (weight << 8) | translateBias(bias)) }
    {}

    constexpr KernelMode(DataType input) :
        KernelMode(input, Gna2DataTypeNone, Gna2DataTypeNone)
    {}

    ~KernelMode() = default;

    constexpr bool operator<(const KernelMode &mode) const
    {
        return mode.Value < Value;
    }

protected:

    /** kernels use single mode for bias type 8/16/32b */
    static constexpr DataType translateBias(DataType bias)
    {
        switch (bias)
        {
        case Gna2DataTypeInt16:
        case Gna2DataTypeInt32:
            return Gna2DataTypeInt8;
        default:
            return bias;
        }
    }

    DataType Value = Gna2DataTypeNone;
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
    GetKernelMap(kernel_op operation, KernelMode dataMode = {Gna2DataTypeInt16})
    {
        try
        {
            return reinterpret_cast<const KernelMap<KernelType>&>(
                GetKernels(operation, dataMode));
        }
        catch (std::out_of_range &)
        {
            throw GnaModelErrorException{ Gna2ItemTypeOperandType, Gna2ErrorTypeNotInSet, 0 };
        }
    }

    void SetHardwareAcceleration(bool isHardwareSupported)
    {
        accelerationModes[AccelerationMode{ Gna2AccelerationModeHardware }] = isHardwareSupported;
    }

    void PrintAllAccelerationModes() const;
protected:
    std::map<AccelerationMode, bool> accelerationModes;

private:
    void DetectSoftwareAccelerationModes();
    //sorted from slowest to fastest
    std::vector<Gna2AccelerationMode> supportedCpuAccelerations;

    static const KernelMap<VoidKernel>& GetKernels(kernel_op operation, KernelMode dataMode);
};

}
