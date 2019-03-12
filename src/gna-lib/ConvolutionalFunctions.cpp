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
#include "HardwareCapabilities.h"
#include "Expect.h"
#include "HardwareLayer.h"

using std::make_unique;
using std::unique_ptr;
using std::move;

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
            HardwareCapabilities::GetDeviceVersion(validator->Device), this);
        auto caps = static_cast<const TensorLimits* const>(validator->Capabilities);
        validator->ValidateBufferIfSet(Buffer, kernelMemorySize * Count, caps->Align);
    }
}

const FullCapabilitiesMap ConvolutionFunction::strideLimits
{
    {INTEL_CONVOLUTIONAL, {
        { GNA_1_0, std::make_shared<ComponentLimits>(ComponentLimits(
            {GNA_TENSOR_W},
            { { GNA_DIM_W, { 1, CNN_N_FLT_COEFF_MAX, 1, CNN_ERR_CONV_FLT_STRIDE}}}))}
    }},
};

unique_ptr<const ConvolutionFunction> ConvolutionFunction::Create(const Tensor* input, const Tensor* output,
        void const * layerDetails, const LayerValidator& validatorIn)
{
    Shape stride;
    unique_ptr<const FiltersTensor> filters;
    unique_ptr<const Component> strideComponent;
    unique_ptr<const BiasTensor> biases;
    uint32_t filterMode = 0;
    uint32_t biasMode = 0;
    void* filtersBuffer = nullptr;
    void* biasesBuffer = nullptr;

    switch (validatorIn.Operation)
    {
    case INTEL_CONVOLUTIONAL:
    {
        auto cnn = static_cast<const nn_layer_conv*>(layerDetails);
        filterMode = cnn->nBytesFilterCoefficient;
        filtersBuffer = cnn->pFilters;
        biasMode = cnn->nBytesBias;
        biasesBuffer = cnn->pBiases;

        stride[GNA_DIM_W] = cnn->nFeatureMaps * cnn->nFeatureMapColumns;

        auto featureCount = cnn->nFeatureMapRows  * stride[GNA_DIM_W];
        Expect::True(featureCount >= CNN_N_FLT_COEFF_MIN, XNN_ERR_LYR_CFG);
        Expect::InRange(cnn->nFilterRows, ui32_1, CNN_N_FLT_COEFF_MAX, XNN_ERR_LYR_CFG);

        filters = make_unique<const FiltersTensor>(Shape(cnn->nFilters, cnn->nFilterCoefficients, 0),
            filterMode, filtersBuffer, validatorIn);
        strideComponent = make_unique<const Component>(stride, Validator{ validatorIn, strideLimits });
        biases = make_unique<const BiasTensor>(Shape(0, 0, cnn->nFilters),
            0, biasMode, biasesBuffer, validatorIn);

        break;
    }
    default:
        throw GnaException(XNN_ERR_LYR_OPERATION);
    }

    switch (validatorIn.Operation)
     {
     case INTEL_CONVOLUTIONAL:
         return make_unique<const ConvolutionFunction>(
             AccelerationDetector::GetKernelMap<ConvolutionKernel>(
                 static_cast<kernel_op>(validatorIn.Operation), KernelMode { input->Mode.Value }),
             input, output, move(filters), move(biases), move(strideComponent));
     default:
        throw GnaException(XNN_ERR_LYR_OPERATION);
     }
}

ConvolutionFunction::ConvolutionFunction(const KernelMap<ConvolutionKernel>& kernelsIn,
    const Tensor* input, const Tensor* output, unique_ptr<const FiltersTensor> filters,
    unique_ptr<const BiasTensor> biases, std::unique_ptr<const Component> stride) :
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
        Expect::InRange(dim.second, ui32_1, Filters->at(dim.first), XNN_ERR_LYR_CFG);
        Output[dim.first] =
            (input->Dimensions.at(dim.first) - Filters->at(dim.first)) / dim.second + 1;
        OutputsPerFilterCount *= Output[dim.first];
    }

    for (const auto& dim : Filters->Dimensions)
    {
        if (GNA_DIM_N != dim.first)
        {
            Expect::True(dim.second <= input->Dimensions.at(dim.first), XNN_ERR_LYR_CFG);
        }
    }

    hiddenConfig = make_unique<ConvolutionConfig>(Stride->at(GNA_DIM_W), OutputsPerFilterCount,
        Filters->at(GNA_DIM_N), Filters->at(GNA_DIM_W),
        input->Buffer, Filters->Buffer, Biases->Buffer, output->Buffer, Biases->Mode, Filters->Mode);
}

unique_ptr<const ConvolutionConfig> ConvolutionFunction::GetRequestConfig(const BaseAddress& inputs, const BaseAddress& outputs) const
{
    return make_unique<const ConvolutionConfig>(hiddenConfig.get(), inputs, outputs);
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
