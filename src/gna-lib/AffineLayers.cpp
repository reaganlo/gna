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

#include "AffineLayers.h"

#include "LayerConfiguration.h"
#include "Validator.h"

using std::make_unique;

using namespace GNA;

AffineBaseLayer::AffineBaseLayer(const nn_layer *layer) :
    Layer(layer),
    Affine(AffineFunction::Create(layer->nLayerKind, layer->pLayerStruct,
        AffineBaseConfig{Output.ElementCount, Input.VectorCount, Input.ElementCount, Input.Buffer, Output.Buffer})),
    Activation(ActivationFunction::Create(&static_cast<const nn_layer_affine*>(layer->pLayerStruct)->pwl, false,
        Output.ScratchPad,
        PwlOutputConfig{0, Output.ElementCount - 1, 0, Input.VectorCount - 1, Output.ElementCount, Output.Buffer}))
{
    Output.SetOutputMode(Activation.operator bool(), layer->nBytesPerOutput);
    if (Activation)
    {
        Layer::ComputeHidden = [this](acceleration accel, KernelBuffers *fvBuffers, uint32_t *saturationCount)
        {this->computeHiddenPwl(accel, fvBuffers, saturationCount); };

        Layer::ComputeConfig = [this](LayerConfiguration &layerConfiguration, acceleration accel, KernelBuffers *fvBuffers, uint32_t *saturationCount)
        {this->computeConfigPwl(layerConfiguration, accel, fvBuffers, saturationCount); };
    }
    else
    {
        Layer::ComputeHidden = [this](acceleration accel, KernelBuffers *fvBuffers, uint32_t *saturationCount)
        {this->computeHidden(accel, fvBuffers, saturationCount); };

        Layer::ComputeConfig = [this](LayerConfiguration &layerConfiguration, acceleration accel, KernelBuffers *fvBuffers, uint32_t *saturationCount)
        {this->computeConfig(layerConfiguration, accel, fvBuffers, saturationCount); };
    }
}

void AffineBaseLayer::UpdateKernelConfigs(LayerConfiguration& layerConfiguration) const
{
    auto const inputs = layerConfiguration.InputBuffer ? *layerConfiguration.InputBuffer : Input.Buffer;

    auto const outputs = layerConfiguration.OutputBuffer ? *layerConfiguration.OutputBuffer : Output.Buffer;

    auto& configs = layerConfiguration.Configs;

    if (Activation)
    {
        configs.PwlOutput = Activation->GetOutputConfig(outputs);
        configs.Affine = Affine->GetRunConfig(inputs, Output.ScratchPad);
    }
    else
    {
        configs.Affine = Affine->GetRunConfig(inputs, outputs);
    }
}

void AffineBaseLayer::computeHidden(acceleration accel, KernelBuffers *fvBuffers, uint32_t *saturationCount) const
{
    Affine->ComputeHidden(accel, saturationCount, fvBuffers);
}

void AffineBaseLayer::computeHiddenPwl(acceleration accel, KernelBuffers *fvBuffers, uint32_t *saturationCount) const
{
    Affine->ComputeHidden(accel, saturationCount, fvBuffers);

    Activation->ComputeHidden(accel, saturationCount);
}

void AffineBaseLayer::computeConfig(const LayerConfiguration& layerConfiguration, acceleration accel, KernelBuffers *fvBuffers, uint32_t *saturationCount) const
{
    Affine->ComputeConfig(layerConfiguration, accel, saturationCount, fvBuffers);
}

void AffineBaseLayer::computeConfigPwl(const LayerConfiguration& layerConfiguration, acceleration accel, KernelBuffers *fvBuffers, uint32_t *saturationCount) const
{
    Affine->ComputeConfig(layerConfiguration, accel, saturationCount, fvBuffers);

    Activation->ComputeConfig(layerConfiguration, accel, saturationCount);
}

AffineLayer::AffineLayer(const nn_layer *layer) :
    AffineBaseLayer(layer)
{};

void AffineLayer::UpdateKernelConfigs(LayerConfiguration& layerConfiguration) const
{
    AffineBaseLayer::UpdateKernelConfigs(layerConfiguration);
    if (Activation)
    {
        auto const outputCount = layerConfiguration.ActiveList ?
            layerConfiguration.ActiveList->IndicesCount : Output.ElementCount;
        layerConfiguration.Configs.PwlOutput->rowLast = outputCount - 1;
    }
}

AffineDiagonalLayer::AffineDiagonalLayer(const nn_layer *layer) :
    AffineBaseLayer(layer)
{
    Expect::True(Input.ElementCount == Output.ElementCount, XNN_ERR_LYR_CFG);
    Expect::True(Input.VectorCount == Output.VectorCount, XNN_ERR_LYR_CFG);
}
