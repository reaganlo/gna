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

#include "AffineLayers.h"

#include "ActivationHelper.h"
#include "ActiveList.h"
#include "Address.h"
#include "Bias.h"
#include "DataMode.h"
#include "Expect.h"
#include "KernelArguments.h"
#include "LayerConfiguration.h"
#include "LayerInput.h"
#include "LayerOutput.h"
#include "Tensor.h"
#include "Weight.h"

#include "gna2-common-api.h"
#include "gna2-memory-api.h"

#include "common.h"
#include "gna-api-types-xnn.h"
#include "gna-api.h"

#include <algorithm>
#include <map>

using namespace GNA;

//TODO:3:provide better mechanism for scratchpad
void *getGlobal2MBScratchpad()
{
    static void* ptr = nullptr;
    uint32_t sizeGranted;
    if (ptr == nullptr)
    {
        const auto status = Gna2MemoryAlloc(1 << 21, &sizeGranted, &ptr);
        if (status != Gna2StatusSuccess || ptr == nullptr)
        {
            Log->Error("Unsuccessful Scratchpad allocation\n");
        }
    }
    return ptr;
}

AffineBaseLayer::AffineBaseLayer(
    const nn_layer& layer, std::vector<TransformOperation> transforms,
    const BaseValidator& validatorIn) :
        Layer(layer, validatorIn, transforms, layer.pOutputsIntermediate)
{
}

AffineBaseLayer::AffineBaseLayer(
        const Gna2Operation& operation,
        const std::vector<TransformOperation> transforms,
        const BaseValidator& validatorIn) :
    Layer(operation, validatorIn, transforms, BaseAddress(getGlobal2MBScratchpad()))
{
}

DataConfig AffineBaseLayer::GetDataMode() const
{
    auto affineTransform = static_cast<AffineFunction const *>(GetInputTransform());
    return AffineBaseLayer::getDataMode(affineTransform);
}

void AffineLayer::UpdateKernelConfigs(LayerConfiguration& layerConfiguration) const
{
    AffineBaseLayer::UpdateKernelConfigs(layerConfiguration);
    auto activation = Transforms.Get<ActivationFunction>(ActivationTransform);
    if (activation)
    {
        auto const outputCount = layerConfiguration.ActList ?
            layerConfiguration.ActList->IndicesCount : Output.Dimensions.at('H');
        activation->UpdateActiveOutputCount(layerConfiguration.ConfigList,
                                            outputCount * Output.Dimensions.at('W'));
    }
}

AffineLayer::AffineLayer(const nn_layer& layer, const BaseValidator& validatorIn) :
    AffineBaseLayer(layer, { AffineTransform, ActivationTransform }, validatorIn)
{}

AffineLayer::AffineLayer(const Gna2Operation& operation, const BaseValidator& validatorIn) :
    AffineBaseLayer(operation, { AffineTransform, ActivationTransform }, validatorIn)
{
    if(operation.Type == Gna2OperationTypeElementWiseAffine)
    {
        ModelErrorHelper::ExpectEqual(Output.AsModelValue('H'), Input.AsModelValue('H'));
        ModelErrorHelper::ExpectEqual(Output.AsModelValue('W'), Input.AsModelValue('W'));
    }
}

Tensor const & AffineBaseLayer::GetOperand(uint32_t operandIndex) const
{
    // TODO:3:replace with generic solution when all layers are transforms
    switch (operandIndex)
    {
    case ScratchpadOperandIndex:
        return Output.ScratchPad;
    case WeightOperandIndex: //[[fallthrough]]
    case BiasOperandIndex: //[[fallthrough]]
    case WeightScaleFactorOperandIndex:
        return GetInputTransform()->GetOperand(operandIndex);
    case PwlOperandIndex:
        return getTransformOperand(ActivationTransform, 2);// TODO:3:Intentional literal, replace with generic solution when all layers are transforms
    default:
        return Layer::GetOperand(operandIndex);
    }
}
