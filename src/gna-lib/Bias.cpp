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

#define NOMINMAX 1

#include "Bias.h"

#include "AffineLayerCapabilities.h"
#include "Capabilities.h"
#include "ConvolutionKernelArguments.h"
#include "ConvolutionalLayer2DCapabilities.h"
#include "Expect.h"
#include "GmmLayerCapabilities.h"
#include "ModelError.h"
#include "ParameterLimits.h"
#include "PoolingFunctions2D.h"
#include "Shape.h"
#include "Validator.h"

#include "gna2-common-api.h"

#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <utility>

using namespace GNA;

const FullCapabilitiesMap BiasTensor::capabilities =
{
    GetOperationCaps<INTEL_AFFINE>(BiasOperandIndex),
    GetOperationCaps<INTEL_AFFINE_DIAGONAL>(BiasOperandIndex),
    GetOperationCaps<INTEL_AFFINE_MULTIBIAS>(BiasOperandIndex),
    GetOperationCaps<INTEL_RECURRENT>(BiasOperandIndex),
    GetOperationCaps<INTEL_CONVOLUTIONAL>(BiasOperandIndex),
    GetOperationCaps<INTEL_CONVOLUTIONAL_2D>(BiasOperandIndex),
    GetOperationCaps<INTEL_CONVOLUTIONAL_1D>(BiasOperandIndex),
    GetOperationCaps<INTEL_GMM>(BiasOperandIndex),
};

const SetLimits<KernelBiasMode> BiasTensor::modeLimits
{
    { KernelBiasModeDisabled, KernelBiasModePerFilter, KernelBiasModePerStride },
    Gna2StatusXnnErrorBiasMode
};

BiasTensor::BiasTensor(const Shape& dimensions, const uint32_t biasVectorIndex, const DataMode& dataMode,
    void * buffer, const LayerValidator& validatorIn, Gna2BiasMode biasMode)
try :
    Tensor{ dimensions, dataMode, buffer, Validator{ validatorIn, capabilities } },
    VectorCount{ biasMode == Gna2BiasModeGrouping ? Dimensions.at('W') : 1 },
    VectorIndex{ biasVectorIndex },
    BiasMode{ ToKernelBiasMode(biasMode, dataMode.Mode) }
{
    validate();
}
catch (GnaException& e)
{
    ModelErrorHelper::SetOperandIndexRethrow(e, BiasOperandIndex);
}

BiasTensor::BiasTensor(const Gna2Tensor &apiTensor, const uint32_t biasVectorIndex,
        Gna2BiasMode biasMode, const LayerValidator& validatorIn)
try :
    Tensor{ apiTensor, capabilities.GetOrder(validatorIn), Validator { validatorIn, capabilities } },
    VectorCount{ biasMode == Gna2BiasModeGrouping ? Dimensions.at('W') : 1 },
    VectorIndex{ biasVectorIndex },
    BiasMode{ ToKernelBiasMode(biasMode, apiTensor.Mode) }
{
    validate();
}
catch (GnaException& e)
{
    ModelErrorHelper::SetOperandIndexRethrow(e, BiasOperandIndex);
}

void BiasTensor::validate() const
{
    const std::function<void()> command = [&]()
    {
        ModelErrorHelper::ExpectAboveEq(VectorIndex, 0u);
        ModelErrorHelper::ExpectBelowEq(VectorIndex, VectorCount - 1);
    };
    ModelErrorHelper::ExecuteForModelItem(command, GNA2_DISABLED, BiasVectorParamIndex);
    Expect::InSet(BiasMode, modeLimits);
}

KernelBiasMode BiasTensor::ToKernelBiasMode(Gna2BiasMode mode, Gna2TensorMode tensorMode)
{
    //TODO:3:Handle constant scalar when enabled in HW
    if (Gna2TensorModeDisabled == tensorMode ||
        Gna2TensorModeConstantScalar == tensorMode)
    {
        return KernelBiasModeDisabled;
    }
    static const std::map<Gna2BiasMode, KernelBiasMode> biasMap
    {
        { Gna2BiasModeDefault, KernelBiasModePerFilter },
        { Gna2BiasModePerStride, KernelBiasModePerStride },
        { Gna2BiasModeGrouping, KernelBiasModePerFilter },
    };
    return biasMap.at(mode);
}
