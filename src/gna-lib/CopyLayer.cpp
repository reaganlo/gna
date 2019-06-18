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

#include "CopyLayer.h"

#include "AccelerationDetector.h"
#include "Address.h"
#include "Capabilities.h"
#include "Expect.h"
#include "LayerConfiguration.h"
#include "LayerInput.h"
#include "LayerOutput.h"
#include "Macros.h"
#include "ParameterLimits.h"

#include "gna2-common-api.h"
#include "gna2-model-api.h"

#include "gna-api-types-xnn.h"
#include "gna-api.h"

#include <algorithm>
#include <memory>

namespace GNA
{
class BaseValidator;
}

using namespace GNA;

static const std::pair<gna_tensor_dim, RangeLimits<uint32_t>> copyShapeWlimit =
{GNA_DIM_W, {XNN_N_IN_ELEMS_MPLY, XNN_N_IN_ELEMS_MAX, XNN_N_IN_ELEMS_MPLY, Gna2StatusXnnErrorLyrCfg}};

const FullCapabilitiesMap CopyLayer::limits
{
    { INTEL_COPY, {
        { GNA_0_9, std::make_shared<ComponentLimits>(ComponentLimits(
            {GNA_TENSOR_HW},
            {{GNA_DIM_H, {1, XNN_N_GROUP_MAX, 1, Gna2StatusXnnErrorLyrCfg}},
            copyShapeWlimit}))},
        { GNA_3_0, std::make_shared<ComponentLimits>(ComponentLimits(
            {GNA_TENSOR_HW},
            {{GNA_DIM_H, {1, COPY_N_GROUP_MAX, 1, Gna2StatusXnnErrorLyrCfg}},
            copyShapeWlimit}))},
    }},
};

CopyLayer::CopyLayer(const nn_layer& layer, const BaseValidator& validatorIn) :
    Layer(layer, validatorIn, {}, BaseAddress()),
    ColumnCount{ static_cast<const nn_layer_copy*>(layer.pLayerStruct)->nCopyCols },
    RowCount{ static_cast<const nn_layer_copy*>(layer.pLayerStruct)->nCopyRows },
    copyKernels{ AccelerationDetector::GetKernelMap<CopyKernel>(KERNEL_COPY, KernelMode {Input.Mode}) },
    copyHiddenConfig{ RowCount, ColumnCount, Input.Dimensions.at('W'), Output.Dimensions.at('W'), Input.Buffer, Output.Buffer }
{
    auto copyParams = std::make_unique<const Component>(Shape{GNA_TENSOR_HW, RowCount, ColumnCount},
        Validator{ *validator, limits });
    Expect::True(RowCount <= Input.Dimensions.at('H'), Gna2StatusXnnErrorLyrCfg);

    ComputeHidden = [this](AccelerationMode accel, ExecutionConfig const & executionConfig)
                    {this->computeHidden(accel, executionConfig); };

    Compute = [this](LayerConfiguration &layerConfiguration, AccelerationMode accel, ExecutionConfig const & executionConfig)
                    {this->compute(layerConfiguration, accel, executionConfig); };
}

CopyLayer::CopyLayer(const Gna2Operation& operation, const BaseValidator& validatorIn) :
    Layer(operation, validatorIn, {}, BaseAddress()),
    ColumnCount{ GetCopyShape(operation).at('W') },
    RowCount{ GetCopyShape(operation).at('H') },
    copyKernels{ AccelerationDetector::GetKernelMap<CopyKernel>(KERNEL_COPY, KernelMode {Input.Mode}) },
    copyHiddenConfig{ RowCount, ColumnCount, Input.Dimensions.at('W'), Output.Dimensions.at('W'), Input.Buffer, Output.Buffer }
{
    auto copyParams = std::make_unique<const Component>(Shape{GNA_TENSOR_HW, ColumnCount, RowCount},
        Validator{ *validator, limits });
    Expect::True(RowCount <= Input.Dimensions.at('H'), Gna2StatusXnnErrorLyrCfg);

    ComputeHidden = [this](AccelerationMode accel, ExecutionConfig const & executionConfig)
    {this->computeHidden(accel, executionConfig); };

    Compute = [this](LayerConfiguration &layerConfiguration, AccelerationMode accel, ExecutionConfig const & executionConfig)
    {this->compute(layerConfiguration, accel, executionConfig); };
}

void CopyLayer::UpdateKernelConfigs(LayerConfiguration& layerConfiguration) const
{
    BaseAddress inputBuffer = Input;
    if (layerConfiguration.Buffers.count(InputComponent) > 0)
    {
        inputBuffer = layerConfiguration.Buffers[InputComponent];
        Input.ValidateBuffer(inputBuffer);
    }

    BaseAddress outputBuffer = Output;
    if (layerConfiguration.Buffers.count(OutputComponent) > 0)
    {
        outputBuffer = layerConfiguration.Buffers[OutputComponent];
        Output.ValidateBuffer(outputBuffer);
    }

    auto& configs = layerConfiguration.Configs;
    if(!configs.Copy)
    {
        configs.Copy = std::make_unique<CopyConfig>(copyHiddenConfig);
    }

    configs.Copy->input = inputBuffer;
    configs.Copy->output = outputBuffer;
}

DataConfig CopyLayer::GetDataMode() const
{
    return DataConfig(Input.Mode, GNA_DATA_DISABLED, GNA_DATA_DISABLED, Output.Mode);
}

void CopyLayer::computeHidden(AccelerationMode accel, ExecutionConfig const & executionConfig) const
{
    UNREFERENCED_PARAMETER(executionConfig);
    copyKernels.at(accel)(&copyHiddenConfig);
}

void CopyLayer::compute(const LayerConfiguration& layerConfiguration, AccelerationMode accel, ExecutionConfig const & executionConfig) const
{
    UNREFERENCED_PARAMETER(executionConfig);
    auto copyConfig = layerConfiguration.Configs.Copy.get();
    copyKernels.at(accel)(copyConfig);
}

Shape CopyLayer::GetCopyShape(const Gna2Operation& operation)
{
    return Shape::Create(*reinterpret_cast<const Gna2Shape *>(operation.Parameters[0]), GNA_TENSOR_HW);
}
