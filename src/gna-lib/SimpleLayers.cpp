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

#include "AccelerationDetector.h"
#include "LayerConfiguration.h"
#include "Expect.h"

using namespace GNA;

TransposeLayer::TransposeLayer(nn_layer const * const layer, const BaseValidator& validatorIn) :
    Layer(layer, validatorIn, {}, BaseAddress()),
    transposeKernels{ AccelerationDetector::GetKernelMap<TransposeKernel>(KERNEL_TRANSPOSE,  KernelMode{Input.Mode}) },
    transposeHiddenConfig{ Operation == INTEL_INTERLEAVE ? Input.at(GNA_DIM_N) : Input.at(GNA_DIM_W),
                           Operation == INTEL_INTERLEAVE ? Input.at(GNA_DIM_W) : Input.at(GNA_DIM_N),
                           Input.Buffer, Output.Buffer }
{
    Expect::Equal(Input.at(GNA_DIM_W), Output.at(GNA_DIM_H), XNN_ERR_LYR_CFG);
    Expect::Equal(Input.at(GNA_DIM_N), Output.at(GNA_DIM_N), XNN_ERR_LYR_CFG);
    Expect::Null(layer->pLayerStruct); // transpose layers do not have layer details
    Expect::Null(Output.ScratchPad); // in transpose layer no 4B output array is allowed

    ComputeHidden = [this](acceleration accel, KernelBuffers *fvBuffers, uint32_t *saturationCount)
                    {this->computeHidden(accel, fvBuffers, saturationCount); };

    Compute = [this](LayerConfiguration &layerConfiguration, acceleration accel, KernelBuffers *fvBuffers, uint32_t *saturationCount)
                    {this->compute(layerConfiguration, accel, fvBuffers, saturationCount); };
}

void TransposeLayer::UpdateKernelConfigs(LayerConfiguration& layerConfiguration) const
{
    BaseAddress inputBuffer = Input;
    if (layerConfiguration.Buffers.count(InputComponent))
    {
        inputBuffer = layerConfiguration.Buffers[InputComponent];
        Input.ValidateBuffer(inputBuffer);
    }

    BaseAddress outputBuffer = Output;
    if (layerConfiguration.Buffers.count(OutputComponent))
    {
        outputBuffer = layerConfiguration.Buffers[OutputComponent];
        Output.ValidateBuffer(outputBuffer);
    }

    auto& configs = layerConfiguration.Configs;
    if(!configs.Transpose)
        configs.Transpose = std::make_unique<TransposeConfig>(transposeHiddenConfig);

    configs.Transpose->input = inputBuffer;
    configs.Transpose->output = outputBuffer;
}

void TransposeLayer::computeHidden(acceleration accel, KernelBuffers *fvBuffers, uint32_t *saturationCount) const
{
    UNREFERENCED_PARAMETER(fvBuffers);
    UNREFERENCED_PARAMETER(saturationCount);
    transposeKernels.at(accel)(&transposeHiddenConfig);
}

void TransposeLayer::compute(const LayerConfiguration& layerConfiguration, acceleration accel, KernelBuffers *fvBuffers, uint32_t *saturationCount) const
{
    UNREFERENCED_PARAMETER(fvBuffers);
    UNREFERENCED_PARAMETER(saturationCount);
    auto transposeConfig = layerConfiguration.Configs.Transpose.get();
    transposeKernels.at(accel)(transposeConfig);
}

CopyLayer::CopyLayer(const nn_layer *layer, const BaseValidator& validatorIn) :
    Layer(layer, validatorIn, {}, BaseAddress()),
    ColumnCount{ static_cast<const nn_layer_copy*>(layer->pLayerStruct)->nCopyCols },
    RowCount{ static_cast<const nn_layer_copy*>(layer->pLayerStruct)->nCopyRows },
    copyKernels{ AccelerationDetector::GetKernelMap<CopyKernel>(KERNEL_COPY, KernelMode {Input.Mode}) },
    copyHiddenConfig{ RowCount, ColumnCount, Input.at(GNA_DIM_W), Output.at(GNA_DIM_H), Input.Buffer, Output.Buffer }
{
    // TODO:3: refactor to use scalars/component and validator
    Expect::MultiplicityOf(ColumnCount, XNN_N_IN_ELEMS_MPLY);
    Expect::InRange(ColumnCount, XNN_N_IN_ELEMS_MPLY, XNN_N_IN_ELEMS_MAX, XNN_ERR_LYR_CFG);
    Expect::True(RowCount <= Input.at(GNA_DIM_N), XNN_ERR_LYR_CFG);

    ComputeHidden = [this](acceleration accel, KernelBuffers *fvBuffers, uint32_t *saturationCount)
                    {this->computeHidden(accel, fvBuffers, saturationCount); };

    Compute = [this](LayerConfiguration &layerConfiguration, acceleration accel, KernelBuffers *fvBuffers, uint32_t *saturationCount)
                    {this->compute(layerConfiguration, accel, fvBuffers, saturationCount); };
}

void CopyLayer::UpdateKernelConfigs(LayerConfiguration& layerConfiguration) const
{
    BaseAddress inputBuffer = Input;
    if (layerConfiguration.Buffers.count(InputComponent))
    {
        inputBuffer = layerConfiguration.Buffers[InputComponent];
        Input.ValidateBuffer(inputBuffer);
    }

    BaseAddress outputBuffer = Output;
    if (layerConfiguration.Buffers.count(OutputComponent))
    {
        outputBuffer = layerConfiguration.Buffers[OutputComponent];
        Output.ValidateBuffer(outputBuffer);
    }

    auto& configs = layerConfiguration.Configs;
    if(!configs.Copy)
        configs.Copy = std::make_unique<CopyConfig>(copyHiddenConfig);

    configs.Copy->input = inputBuffer;
    configs.Copy->output = outputBuffer;
}

void CopyLayer::computeHidden(acceleration accel, KernelBuffers *fvBuffers, uint32_t *saturationCount) const
{
    UNREFERENCED_PARAMETER(fvBuffers);
    UNREFERENCED_PARAMETER(saturationCount);
    copyKernels.at(accel)(&copyHiddenConfig);
}

void CopyLayer::compute(const LayerConfiguration& layerConfiguration, acceleration accel, KernelBuffers *fvBuffers, uint32_t *saturationCount) const
{
    UNREFERENCED_PARAMETER(fvBuffers);
    UNREFERENCED_PARAMETER(saturationCount);
    auto copyConfig = layerConfiguration.Configs.Copy.get();
    copyKernels.at(accel)(copyConfig);
}
