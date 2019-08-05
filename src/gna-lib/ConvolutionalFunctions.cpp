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

#include "ConvolutionalFunctions.h"

#include "AccelerationDetector.h"
#include "Capabilities.h"
#include "Expect.h"
#include "HardwareCapabilities.h"
#include "HardwareLayer.h"
#include "ModelWrapper.h"
#include "Tensor.h"
#include "Validator.h"

#include "gna-api-types-xnn.h"
#include "gna-api.h"

#include <memory>
#include <utility>

using namespace GNA;

FiltersTensor::FiltersTensor(const Shape& dimensions, const DataMode & dataMode, void * buffer,
    const LayerValidator& validatorIn) :
    WeightTensor{ dimensions, dataMode, buffer, validatorIn },
    Count{ Dimensions.at(GNA_DIM_N) },
    CoefficientCount{ Dimensions.at(GNA_DIM_W) }
{
    // validate buffer size with padding
    if (GNA_DATA_DISABLED != Mode)
    {
        const auto kernelMemorySize = HardwareLayerCnn2D::GetKernelMemorySize(
            validator->HwCapabilities.GetDeviceVersion(), this);
        const auto caps = static_cast<const TensorLimits *>(validator->Capabilities);
        validator->ValidateBufferIfSet(Buffer, kernelMemorySize * Count, caps->Align);
    }
}

std::unique_ptr<const FiltersTensor> FiltersTensor::Create(const Gna2Tensor& filtersTensor, const LayerValidator& validatorIn)
{
    return std::make_unique<const FiltersTensor>(
        Shape::Create(filtersTensor.Shape, GNA_TENSOR_NHWD),
        DataMode{filtersTensor.Type},
        filtersTensor.Data,
        validatorIn );
}

const FullCapabilitiesMap ConvolutionFunction::strideLimits
{
    {INTEL_CONVOLUTIONAL, {
        { GNA_1_0, std::make_shared<ComponentLimits>(ComponentLimits(
            {GNA_TENSOR_W},
            { { GNA_DIM_W, { 1, CNN_N_FLT_COEFF_MAX, 1, Gna2StatusCnnErrorConvFltStride}}}))}
    }},
};

ConvolutionFunction::ConvolutionFunction(const KernelMap<ConvolutionKernel>& kernelsIn,
    const Tensor* input, const Tensor* output, std::unique_ptr<const FiltersTensor> filters,
    std::unique_ptr<const BiasTensor> biases, std::unique_ptr<const Component> stride) :
    Biases{ move(biases) },
    Filters{ move(filters) },
    Stride{ move(stride) },
    kernels{ kernelsIn }
{
    // save #filters as Depth dimension of output (D in filters is used for 3D convolution)
    Output[GNA_DIM_D] = Filters->at(GNA_DIM_N);
    OutputsPerFilterCount = 1;
    for (const auto& dim : Stride->Dimensions)
    {
        // TODO:3: add Expect::Fits() method
        Expect::InRange(dim.second, ui32_1, Filters->at(dim.first), Gna2StatusXnnErrorLyrCfg);
        Output[dim.first] =
            (input->Dimensions.at(dim.first) - Filters->at(dim.first)) / dim.second + 1;
        OutputsPerFilterCount *= Output[dim.first];
    }

    for (const auto& dim : Filters->Dimensions)
    {
        if (GNA_DIM_N != dim.first)
        {
            Expect::True(dim.second <= input->Dimensions.at(dim.first), Gna2StatusXnnErrorLyrCfg);
        }
    }

    hiddenConfig = std::make_unique<ConvolutionConfig>(Stride->at(GNA_DIM_W), OutputsPerFilterCount,
        Filters->at(GNA_DIM_N), Filters->at(GNA_DIM_W),
        input->Buffer, Filters->Buffer, Biases->Buffer, output->Buffer, Biases->Mode, Filters->Mode);
}

std::unique_ptr<const ConvolutionConfig> ConvolutionFunction::GetRequestConfig(const BaseAddress& inputs, const BaseAddress& outputs) const
{
    return std::make_unique<const ConvolutionConfig>(hiddenConfig.get(), inputs, outputs);
}

void ConvolutionFunction::ComputeHidden(AccelerationMode accel, ExecutionConfig const & execution) const
{
    auto convConfig = ConvolutionConfig{hiddenConfig.get(), execution};

    kernels.at(accel)(&convConfig);
}

void ConvolutionFunction::Compute(const ConvolutionConfig* const config, AccelerationMode accel, ExecutionConfig const & execution) const
{
    auto convConfig = ConvolutionConfig{ config, execution };

    kernels.at(accel)(&convConfig);
}

std::unique_ptr<const ConvolutionFunction> ConvolutionFunction::finalizeCreation(
    const Tensor* input, const Tensor* output, std::unique_ptr<const FiltersTensor> filters,
    std::unique_ptr<const Component> stride, std::unique_ptr<const BiasTensor> biases)
{
    return std::make_unique<const ConvolutionFunction>(
        AccelerationDetector::GetKernelMap<ConvolutionKernel>(
            static_cast<kernel_op>(INTEL_CONVOLUTIONAL), KernelMode{ input->Mode.Value }),
        input, output, std::move(filters), std::move(biases), std::move(stride));
}

std::unique_ptr<const FiltersTensor> ConvolutionFunction::createFilters(const Gna2Operation & apiOperation,
    const LayerValidator& validatorIn)
{
    const auto& apiFilters = ModelWrapper::GetOperand(apiOperation, FilterOperandIndex);
    return std::make_unique<const FiltersTensor>(Shape::Create(apiFilters.Shape, GNA_TENSOR_NWH),
        apiFilters.Type, apiFilters.Data, validatorIn);
}

std::unique_ptr<const FiltersTensor> ConvolutionFunction::createFilters(const nn_layer_conv& cnn,
    const LayerValidator& validatorIn)
{
    return std::make_unique<const FiltersTensor>(Shape(GNA_TENSOR_NWH, cnn.nFilters, cnn.nFilterCoefficients, 0u),
        cnn.nBytesFilterCoefficient, cnn.pFilters, validatorIn);
}

std::unique_ptr<const Component> ConvolutionFunction::createStride(const Gna2Operation & apiOperation,
    const LayerValidator & validatorIn)
{
    const auto& strideShape = ModelWrapper::GetParameter<Gna2Shape>(apiOperation, ParameterIndexConvolutionStride);
    const Shape stride{ GNA_TENSOR_W, strideShape.Dimensions[0] };
    return std::make_unique<const Component>(stride, Validator{ validatorIn, strideLimits });
}

std::unique_ptr<const Component> ConvolutionFunction::createStride(const nn_layer_conv & cnn,
    const LayerValidator & validatorIn)
{
    const Shape stride{ GNA_TENSOR_W, cnn.nFeatureMaps * cnn.nFeatureMapColumns };
    return std::make_unique<const Component>(stride, Validator{ validatorIn, strideLimits });
}

std::unique_ptr<const BiasTensor> ConvolutionFunction::createBiases(const Gna2Operation & apiOperation,
    const LayerValidator & validatorIn)
{
    const auto& apiBias = ModelWrapper::GetOperand(apiOperation, BiasOperandIndex);
    return std::make_unique<const BiasTensor>(Shape::Create(apiBias.Shape, GNA_TENSOR_N),
        0, apiBias.Type, apiBias.Data, validatorIn);
}

std::unique_ptr<const BiasTensor> ConvolutionFunction::createBiases(const nn_layer_conv & cnn,
    const LayerValidator & validatorIn)
{
    return  std::make_unique<const BiasTensor>(Shape(GNA_TENSOR_N, cnn.nFilters),
        0, cnn.nBytesBias, cnn.pBiases, validatorIn);
}

void ConvolutionFunction::expectValid(const Gna2Operation& apiOperation)
{
    const auto& apiInput = ModelWrapper::GetOperand(apiOperation, InputOperandIndex);

    const auto featureCount = ModelWrapper::ShapeGetNumberOfElements(&apiInput.Shape);
    Expect::True(featureCount >= CNN_N_FLT_COEFF_MIN, Gna2StatusXnnErrorLyrCfg);
    // TODO:3:P2 Consider validating filterRows API2 equivalent
}

void ConvolutionFunction::expectValid(const nn_layer_conv& cnn)
{
    const auto featureCount = cnn.nFeatureMaps * cnn.nFeatureMapRows * cnn.nFeatureMapColumns;
    Expect::True(featureCount >= CNN_N_FLT_COEFF_MIN, Gna2StatusXnnErrorLyrCfg);
    Expect::InRange(cnn.nFilterRows, ui32_1, CNN_N_FLT_COEFF_MAX, Gna2StatusXnnErrorLyrCfg);
}
