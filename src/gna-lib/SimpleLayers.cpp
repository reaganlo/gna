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

#include "SimpleLayers.h"

#include "LayerConfiguration.h"
#include "Validator.h"

using namespace GNA;

TransposeLayer::TransposeLayer(nn_layer const * const layer) :
    Layer(layer),
    transposeKernels{ AccelerationDetector::TransposeKernels },
    transposeHiddenConfig{ Input.ElementCount, Input.VectorCount, Input.Buffer, Output.Buffer }
{
    Output.SetOutputMode(LayerOutput::NonActivatedOutput, layer->nBytesPerOutput);
    Expect::True(Input.ElementCount == Output.VectorCount, XNN_ERR_LYR_CFG);
    Expect::True(Input.VectorCount == Output.ElementCount, XNN_ERR_LYR_CFG);
    Expect::Null(layer->pLayerStruct); // transpose layers do not have layer details
    Expect::Null(Output.ScratchPad); // in transpose layer no 4B output array is allowed

    ComputeHidden = [this](acceleration accel, KernelBuffers *fvBuffers, uint32_t *saturationCount) 
                    {this->computeHidden(accel, fvBuffers, saturationCount); };

    ComputeConfig = [this](LayerConfiguration &layerConfiguration, acceleration accel, KernelBuffers *fvBuffers, uint32_t *saturationCount) 
                    {this->computeConfig(layerConfiguration, accel, fvBuffers, saturationCount); };
}

void TransposeLayer::UpdateKernelConfigs(LayerConfiguration& layerConfiguration) const
{
    auto inputBuffer = layerConfiguration.InputBuffer
        ? layerConfiguration.InputBuffer->Get<int16_t>() : Input.Buffer;

    auto outputBuffer = layerConfiguration.OutputBuffer
        ? layerConfiguration.OutputBuffer->Get<int16_t>() : Output.Buffer;

    if(!layerConfiguration.transposeConfig)
        layerConfiguration.transposeConfig = std::make_unique<TransposeConfig>(transposeHiddenConfig);
    layerConfiguration.transposeConfig->input = inputBuffer;
    layerConfiguration.transposeConfig->output = outputBuffer;
}

void TransposeLayer::computeHidden(acceleration accel, KernelBuffers *fvBuffers, uint32_t *saturationCount) const
{
    auto transposeConfig = transposeHiddenConfig;
    transposeKernels.at(accel)(&transposeConfig);
}

void TransposeLayer::computeConfig(const LayerConfiguration& layerConfiguration, acceleration accel, KernelBuffers *fvBuffers, uint32_t *saturationCount) const
{
    auto transposeConfig = *layerConfiguration.transposeConfig;
    transposeKernels.at(accel)(&transposeConfig);
}

CopyLayer::CopyLayer(const nn_layer *layer) :
    Layer(layer),
    ColumnCount{ static_cast<const nn_layer_copy*>(layer->pLayerStruct)->nCopyCols },
    RowCount{ static_cast<const nn_layer_copy*>(layer->pLayerStruct)->nCopyRows },
    copyKernels{ AccelerationDetector::CopyKernels },
    copyHiddenConfig{ RowCount, ColumnCount, Input.VectorCount, Output.VectorCount, Input.Buffer, Output.Buffer }
{
    Output.SetOutputMode(LayerOutput::NonActivatedOutput, layer->nBytesPerOutput);
    Expect::MultiplicityOf(ColumnCount, XNN_N_IN_ELEMS_MPLY);
    Expect::InRange(ColumnCount, XNN_N_IN_ELEMS_MPLY, XNN_N_IN_ELEMS_MAX, XNN_ERR_LYR_CFG);
    Expect::True(RowCount <= Input.VectorCount, XNN_ERR_LYR_CFG);

    ComputeHidden = [this](acceleration accel, KernelBuffers *fvBuffers, uint32_t *saturationCount) 
                    {this->computeHidden(accel, fvBuffers, saturationCount); };

    ComputeConfig = [this](LayerConfiguration &layerConfiguration, acceleration accel, KernelBuffers *fvBuffers, uint32_t *saturationCount) 
                    {this->computeConfig(layerConfiguration, accel, fvBuffers, saturationCount); };
}

void CopyLayer::UpdateKernelConfigs(LayerConfiguration& layerConfiguration) const
{
    auto inputBuffer = layerConfiguration.InputBuffer
        ? layerConfiguration.InputBuffer->Get<int16_t>() : Input.Buffer;

    auto outputBuffer = layerConfiguration.OutputBuffer
        ? layerConfiguration.OutputBuffer->Get<int16_t>() : Output.Buffer;

    if(!layerConfiguration.copyConfig)
        layerConfiguration.copyConfig = std::make_unique<CopyConfig>(copyHiddenConfig);
    layerConfiguration.copyConfig->input = inputBuffer;
    layerConfiguration.copyConfig->output = outputBuffer;
}

void CopyLayer::computeHidden(acceleration accel, KernelBuffers *fvBuffers, uint32_t *saturationCount) const
{
    auto copyConfig = copyHiddenConfig;
    copyKernels.at(accel)(&copyConfig);
}

void CopyLayer::computeConfig(const LayerConfiguration& layerConfiguration, acceleration accel, KernelBuffers *fvBuffers, uint32_t *saturationCount) const
{
    auto copyConfig = *layerConfiguration.copyConfig;
    copyKernels.at(accel)(&copyConfig);
}
