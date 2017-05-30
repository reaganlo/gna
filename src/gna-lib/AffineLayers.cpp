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

#include "AccelerationDetector.h"
#include "AffineLayers.h"
#include "LayerConfiguration.h"
#include "Validator.h"

using std::make_unique;

using namespace GNA;

AffineLayer::AffineLayer(const nn_layer *layer) :
    Layer(layer),
    Affine(AffineFunction::Create(&static_cast<const nn_layer_affine*>(layer->pLayerStruct)->affine)),
    Activation(ActivationFunction::Create(&static_cast<const nn_layer_affine*>(layer->pLayerStruct)->pwl, false,
        Output.ScratchPad, 
        PwlOutputConfig{0, Output.ElementCount - 1, 0, Input.VectorCount - 1, Output.ElementCount, Output.Buffer})),
    affineKernels{ AccelerationDetector::GetKernelMap<AffineKernel>(Affine->GetWeightMode(), Config.Kind)},
    affineKernelsAl{ AccelerationDetector::GetKernelMap<AffineActiveListKernel>(Affine->GetWeightMode())},
    affineHiddenConfig{ Output.ElementCount, Input.VectorCount, Input.ElementCount, Input.Buffer, Output.Buffer,
        Affine->GetWeights(), Affine->GetBiases(), nullptr, 0 }
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
};

void AffineLayer::UpdateKernelConfigs(LayerConfiguration& layerConfiguration) const
{
    auto nOutputs = layerConfiguration.ActiveList ? layerConfiguration.ActiveList->IndicesCount : Output.ElementCount;

    auto inputBuffer = layerConfiguration.InputBuffer
        ? layerConfiguration.InputBuffer->Get<int16_t>() : Input.Buffer;

    const InOutBuffer outputBuffer = layerConfiguration.OutputBuffer
        ? *layerConfiguration.OutputBuffer : Output.Buffer;

    auto& configs = layerConfiguration.Configs;

    if(!configs.Affine)
        configs.Affine = make_unique<AffineConfig>(affineHiddenConfig);
    configs.Affine->input = inputBuffer;

    if (Activation)
    {
        if(!configs.PwlOutput)
            configs.PwlOutput = Activation->GetOutputConfig(outputBuffer);
        configs.PwlOutput->rowLast = nOutputs - 1;
        configs.Affine->output = Output.ScratchPad;
    }
    else
    {
        configs.Affine->output = outputBuffer;
    }
}

void AffineLayer::computeHidden(acceleration accel, KernelBuffers *fvBuffers, uint32_t *saturationCount) const
{
    auto config = AffineConfig{&affineHiddenConfig, saturationCount, fvBuffers};

    affineKernels.at(accel)(&config);
}

void AffineLayer::computeHiddenPwl(acceleration accel, KernelBuffers *fvBuffers, uint32_t *saturationCount) const
{
    computeHidden(accel, fvBuffers, saturationCount);

    Activation->computeHidden(accel, saturationCount);
}

void AffineLayer::computeConfig(const LayerConfiguration& layerConfiguration, acceleration accel, KernelBuffers *fvBuffers, uint32_t *saturationCount) const
{
    auto config = AffineConfig{layerConfiguration.Configs.Affine.get(), saturationCount, fvBuffers};

    if (layerConfiguration.ActiveList)
    {
        auto alConfig = AffineConfigAl{layerConfiguration.ActiveList->Indices, layerConfiguration.ActiveList->IndicesCount};
        affineKernelsAl.at(accel)(&config, &alConfig);
    }
    else
    {
        affineKernels.at(accel)(&config);
    }
}

void AffineLayer::computeConfigPwl(const LayerConfiguration& layerConfiguration, acceleration accel, KernelBuffers *fvBuffers, uint32_t *saturationCount) const
{
    computeConfig(layerConfiguration, accel, fvBuffers, saturationCount);

    Activation->computeConfig(layerConfiguration, accel, saturationCount);
}

AffineMultiBiasLayer::AffineMultiBiasLayer(const nn_layer *layer) :
    Layer(layer),
    Affine(AffineFunction::Create(&static_cast<const nn_layer_affine_multi*>(layer->pLayerStruct)->affine)),
    Activation(ActivationFunction::Create(&static_cast<const nn_layer_affine_multi*>(layer->pLayerStruct)->pwl, false,
        Output.ScratchPad,
        PwlOutputConfig{0, Output.ElementCount - 1, 0, Input.VectorCount - 1, Output.ElementCount, Output.Buffer})),
    multibiasKernels{AccelerationDetector::GetKernelMap<AffineKernel>(Affine->GetWeightMode(), Config.Kind)},
    affineHiddenConfig{Output.ElementCount, Input.VectorCount, Input.ElementCount, Input.Buffer, Output.Buffer,
        Affine->GetWeights(), Affine->GetBiases(), Affine->GetMultibias(), Affine->BiasVectorCount}
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
};

void AffineMultiBiasLayer::UpdateKernelConfigs(LayerConfiguration& layerConfiguration) const
{
    auto inputBuffer = layerConfiguration.InputBuffer
        ? layerConfiguration.InputBuffer->Get<int16_t>() : Input.Buffer;

    auto outputBuffer = layerConfiguration.OutputBuffer
        ? layerConfiguration.OutputBuffer->Get<int32_t>() : Output.Buffer;

    auto& configs = layerConfiguration.Configs;

    if(!configs.Affine)
        configs.Affine = make_unique<AffineConfig>(affineHiddenConfig);
    configs.Affine->input = inputBuffer;
    configs.Affine->output = outputBuffer;

    if (Activation)
    {
        if(!configs.PwlOutput)
            configs.PwlOutput = Activation->GetOutputConfig(reinterpret_cast<int16_t*>(outputBuffer));
    }
}

void AffineMultiBiasLayer::computeHidden(acceleration accel, KernelBuffers *fvBuffers, uint32_t *saturationCount) const
{
    auto config = AffineConfig{&affineHiddenConfig, saturationCount, fvBuffers};

    multibiasKernels.at(accel)(&config);
}

void AffineMultiBiasLayer::computeHiddenPwl(acceleration accel, KernelBuffers *fvBuffers, uint32_t *saturationCount) const
{
    computeHidden(accel, fvBuffers, saturationCount);

    Activation->computeHidden(accel, saturationCount);
}

void AffineMultiBiasLayer::computeConfig(const LayerConfiguration &layerConfiguration, acceleration accel, KernelBuffers *fvBuffers, uint32_t *saturationCount) const
{
    auto config = AffineConfig{layerConfiguration.Configs.Affine.get(), saturationCount, fvBuffers};

    multibiasKernels.at(accel)(&config);
}

void AffineMultiBiasLayer::computeConfigPwl(const LayerConfiguration &layerConfiguration, acceleration accel, KernelBuffers *fvBuffers, uint32_t *saturationCount) const
{
    computeConfig(layerConfiguration, accel, fvBuffers, saturationCount);

    Activation->computeConfig(layerConfiguration, accel, saturationCount);
}

AffineDiagonalLayer::AffineDiagonalLayer(const nn_layer *layer) :
    AffineLayer{ layer },
    diagonalKernels {AccelerationDetector::GetKernelMap<AffineKernel>(Affine->GetWeightMode(), Config.Kind)}
{
    Expect::True(Input.ElementCount == Output.ElementCount, XNN_ERR_LYR_CFG);
    Expect::True(Input.VectorCount == Output.VectorCount, XNN_ERR_LYR_CFG);

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

void AffineDiagonalLayer::UpdateKernelConfigs(LayerConfiguration& layerConfiguration) const
{
    auto inputBuffer = layerConfiguration.InputBuffer
        ? layerConfiguration.InputBuffer->Get<int16_t>() : Input.Buffer;

    // TODO: critical: fix output buffer setting for activation and normal out.
    auto outputBuffer = layerConfiguration.OutputBuffer 
        ? layerConfiguration.OutputBuffer->Get<int32_t>() : Output.Buffer;

    auto& configs = layerConfiguration.Configs;

    if(!configs.Affine)
        configs.Affine = make_unique<AffineConfig>(affineHiddenConfig);
    configs.Affine->input = inputBuffer;
    configs.Affine->output = outputBuffer;

    if (Activation)
    {
        if(!configs.PwlOutput)
            configs.PwlOutput = Activation->GetOutputConfig(reinterpret_cast<int16_t*>(outputBuffer));
    }
}

void AffineDiagonalLayer::computeHidden(acceleration accel, KernelBuffers *fvBuffers, uint32_t *saturationCount) const
{
    auto config = AffineConfig{&affineHiddenConfig, saturationCount, fvBuffers};

    diagonalKernels.at(accel)(&config);
}

void AffineDiagonalLayer::computeHiddenPwl(acceleration accel, KernelBuffers *fvBuffers, uint32_t *saturationCount) const
{
    computeHidden(accel, fvBuffers, saturationCount);

    Activation->computeHidden(accel, saturationCount);
}

void AffineDiagonalLayer::computeConfig(const LayerConfiguration &layerConfiguration, acceleration accel, KernelBuffers *fvBuffers, uint32_t *saturationCount) const
{
    auto config = AffineConfig{layerConfiguration.Configs.Affine.get(), saturationCount, fvBuffers};

    diagonalKernels.at(accel)(&config);
}

void AffineDiagonalLayer::computeConfigPwl(const LayerConfiguration &layerConfiguration, acceleration accel, KernelBuffers *fvBuffers, uint32_t *saturationCount) const
{
    computeConfig(layerConfiguration, accel, fvBuffers, saturationCount);

    Activation->computeConfig(layerConfiguration, accel, saturationCount);
}
