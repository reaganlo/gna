/*
 INTEL CONFIDENTIAL
 Copyright 2019 Intel Corporation.

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

#include "Address.h"
#include "DataMode.h"
#include "Tensor.h"

#include "ConvolutionKernelArguments.h"

#include <cstdint>

namespace GNA
{
class FullCapabilitiesMap;
class LayerValidator;
struct Shape;

template<typename T>
struct SetLimits;

struct BiasTensor : public Tensor
{
    BiasTensor(const Shape& dimensions, const uint32_t biasVectorIndex, const DataMode& dataMode,
        void * buffer, const LayerValidator& validator, Gna2BiasMode mode = Gna2BiasModeDefault);

    virtual ~BiasTensor() = default;
    BiasTensor(const Gna2Tensor &apiTensor, uint32_t biasVectorIndex,
                Gna2BiasMode biasMode, const LayerValidator& validatorIn);

    // NOTE: this works only for software mode, HW requires base MB array buffer
    virtual operator const BaseAddress () const override
    {
        return Buffer + (VectorIndex * Mode.Size);
    }

    // NOTE: this works only for software mode, HW requires base MB array buffer
    virtual operator void* () const override
    {
        return Buffer + (VectorIndex * Mode.Size);
    }

    const uint32_t VectorCount;
    const uint32_t VectorIndex;
    const KernelBiasMode BiasMode;

protected:
    static KernelBiasMode ToKernelBiasMode(Gna2BiasMode mode, Gna2TensorMode tensorMode);

    static const FullCapabilitiesMap capabilities;
    static const SetLimits<KernelBiasMode> modeLimits;

private:
    void validate() const;
};


}
