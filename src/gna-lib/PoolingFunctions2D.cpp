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

#include "PoolingFunctions2D.h"

#include "OperationConfig.h"
#include "AccelerationDetector.h"
#include "Capabilities.h"
#include "ConvolutionalLayer2DCapabilities.h"
#include "DataMode.h"
#include "Expect.h"
#include "KernelArguments.h"
#include "ParameterLimits.h"
#include "PoolingMode.h"
#include "Shape.h"
#include "Tensor.h"
#include "Validator.h"

#include "gna-api-types-xnn.h"
#include "gna-api.h"

#include <map>
#include <memory>
#include <utility>

using namespace GNA;

const SetLimits<KernelPoolingMode> PoolingFunction2D::modeLimits =
{
    { KernelPoolingModeMax, KernelPoolingModeSum }, Gna2StatusCnnErrorPoolType
};

std::unique_ptr<PoolingFunction2D> PoolingFunction2D::Create(
    const TransformFactoryConfig& config,
    const OperationConfig& operation)
{
    return create(config, operation);
}

std::unique_ptr<PoolingFunction2D> PoolingFunction2D::create(
    const TransformFactoryConfig& config,
    const OperationConfig& operation)
{
    auto poolingMode = operation.Mode;

    if (Gna2PoolingModeDisabled != poolingMode)
    {
        auto stride = OperationConfig::CreateCnnComponent(operation.PoolingStride,
            config.validator, ConvolutionalLayer2DCapabilities::GetParameters(PoolingStrideParamIndex));
        auto window = OperationConfig::CreateCnnComponent(operation.PoolingWindow,
            config.validator, ConvolutionalLayer2DCapabilities::GetParameters(PoolingWindowParamIndex));

        return std::make_unique<PoolingFunction2D>(
            BaseTransformConfig<PoolingKernel2D>{config,
            AccelerationDetector::GetKernelMap<PoolingKernel2D>(KERNEL_POOLING_2D, { config.input->Mode.Value })},
            poolingMode, std::move(window), std::move(stride));
    }
    return std::unique_ptr<PoolingFunction2D>(nullptr);
}

// unreachable code warning suppression
#if defined(_WIN32)
#pragma warning(disable : 702)
#endif
PoolingFunction2D::PoolingFunction2D(const BaseTransformConfig<PoolingKernel2D>& config,
    const PoolingMode mode, std::unique_ptr<const Component> window,
    std::unique_ptr<const Component> stride) :
    Transform{ PoolingTransform2D, &config.kernels, config.input },
    Mode{ mode },
    Window{ std::move(window) },
    Stride{ std::move(stride) }
{
    Expect::InSet(Mode, modeLimits);

    if (INTEL_CONVOLUTIONAL_1D == Window->GetEffectiveOperationType() ||
        INTEL_CONVOLUTIONAL_1D == Stride->GetEffectiveOperationType())
    {
        is1D = true;
        Expect::InRange(Window->at(GNA_DIM_W), Input->at(GNA_DIM_W),
            Gna2StatusCnnErrorPoolSize);
        /*Expect::InRange(Stride->at(GNA_DIM_W), Window->at(GNA_DIM_W),
            Gna2StatusCnnErrorPoolStride);*/
    }

    Shape outputDims;
    outputDims[GNA_DIM_N] = Input->Dimensions.at(GNA_DIM_N);
    outputDims[GNA_DIM_D] = Input->Dimensions.at(GNA_DIM_D);
    outputDims.LayoutOrder = Input->Dimensions.LayoutOrder;

    for (const auto& iter : Stride->Dimensions)
    {
        auto const dim = iter.first;
        auto const diff = Input->Dimensions.at(dim) - Window->Dimensions.at(dim);
        outputDims[dim] = 1;
        if (diff > 0)
        {
            outputDims[dim] += GnaCeilDiv(diff, iter.second);
        }
    }

    Output = std::make_unique<Tensor>(outputDims, Input->Mode, config.outputBuffer,
        Validator{ config.validator, ConvolutionalLayer2DCapabilities::GetOperands(OutputOperandIndex) });

    const auto output = Output->Dimensions;
    Expect::Fits(output, Input->Dimensions);

    const gna_3d_dimensions input = Input->Dimensions;
    const gna_3d_dimensions poolingStride = Stride->Dimensions;
    const gna_3d_dimensions poolingWindow = Window->Dimensions;

    PoolingConfig2D kernelPoolingConfiguration{ input.width, input.height, input.depth,
        Mode, poolingStride.width, poolingStride.height,
        poolingWindow.width, poolingWindow.height };

    hiddenConfig = std::make_unique<KernelConfig<PoolingConfig2D>>(kernelPoolingConfiguration,
        BaseConfig{ Input->Buffer, Output->Buffer });
}
