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

#include "AffineFunctions.h"

#include "ActiveList.h"
#include "AccelerationDetector.h"
#include "Bias.h"
#include "Capabilities.h"
#include "DataMode.h"
#include "Expect.h"
#include "GnaException.h"
#include "Shape.h"
#include "Validator.h"
#include "LayerConfiguration.h"
#include "Weight.h"

#include "gna2-common-api.h"

#include "common.h"
#include "gna-api.h"
#include "gna-api-types-xnn.h"

#include <algorithm>
#include <cstdint>
#include <memory>

using namespace GNA;

const FullCapabilitiesMap AffineFunctionMulti::Capabilities =
{
    {INTEL_AFFINE_MULTIBIAS, {
        {GNA_2_0, std::make_shared<TensorLimits>(TensorLimits{
            {GNA_TENSOR_H},
            {{GNA_DIM_H, {1, XNN_N_IN_ELEMS_MAX, 1, Gna2StatusXnnErrorBiasVolume}}},
            {{ GNA_DATA_RICH_FORMAT }, Gna2StatusXnnErrorBiasBytes }})}
    }}
};

const std::map<Gna2OperationType, kernel_op> AffineFunction::kernelOperationMap
{
    { Gna2OperationTypeFullyConnectedAffine, KERNEL_AFFINE },
    { Gna2OperationTypeElementWiseAffine, KERNEL_AFFINE_DIAGONAL },
    { Gna2OperationTypeRecurrent, KERNEL_RECURRENT }
};

// Could not split into separate methods for each component as multibias weight scaling is using bias' and weights; tensors...
std::unique_ptr<const AffineFunction> AffineFunction::Create(const Tensor& input, const Tensor& output,
        const Gna2Operation& operation, const LayerValidator& validatorIn)
{

    switch (operation.Type)
    {
    case Gna2OperationTypeFullyConnectedAffine:
        if (HasGroupedBias(operation))
        {
            return createAffineMultiFunction(input, output, operation, validatorIn);
        }
    case Gna2OperationTypeElementWiseAffine:
    case Gna2OperationTypeRecurrent:
        return createAffineSingleFunction(input, output, operation, validatorIn);
    default:
        throw GnaException(Gna2StatusXnnErrorLyrOperation);
    }
}

std::unique_ptr<const AffineFunction> AffineFunction::createAffineSingleFunction(
    const Tensor& input, const Tensor& output,
    const Gna2Operation& operation, const LayerValidator& validatorIn)
{
    auto kernelOperation = kernelOperationMap.at(operation.Type);
    auto weightTensor = static_cast<const Gna2Tensor&>(*operation.Operands[2]);
    auto biasTensor = static_cast<const Gna2Tensor&>(*operation.Operands[3]);
    auto weights = std::make_unique<const WeightTensor>(weightTensor, validatorIn);
    auto biases = std::make_unique<const BiasTensor>(
            biasTensor, 0, Gna2BiasModeDefault, validatorIn);
    auto kernelMode = KernelMode { input.Mode, weights->Mode, biases->Mode };
    auto& affineKernel = AccelerationDetector::GetKernelMap<AffineKernel>(
            static_cast<kernel_op>(kernelOperation), kernelMode);
    return std::make_unique<AffineFunctionSingle>(
            input, output, std::move(weights), std::move(biases), affineKernel,
            AccelerationDetector::GetKernelMap<AffineActiveListKernel>(KERNEL_AFFINE_AL, kernelMode));
}

std::unique_ptr<const AffineFunction> AffineFunction::createAffineMultiFunction(
    const Tensor& input, const Tensor& output,
    const Gna2Operation& operation, const LayerValidator& validatorIn)

{
    std::unique_ptr<const Tensor> weightScales;
    auto weightTensor = static_cast<const Gna2Tensor&>(*operation.Operands[2]);
    auto biasTensor = static_cast<const Gna2Tensor&>(*operation.Operands[3]);
    auto biasVectorIndex = *static_cast<uint32_t *>(operation.Parameters[0]);
    auto weights = std::make_unique<const WeightTensor>(weightTensor, validatorIn);
    auto biases = std::make_unique<const BiasTensor>(biasTensor, biasVectorIndex,
            Gna2BiasModeGrouping, validatorIn);

    // GNA 2.0 backward compatibility only
    if (Gna2DataTypeInt8 == weightTensor.Type
            && Gna2DataTypeInt16 == input.Mode.Type)
    {
        weightScales = std::make_unique<const Tensor>(*operation.Operands[5],
                Validator{ validatorIn, AffineFunctionMulti::Capabilities });
        Expect::ValidBuffer(*weightScales);
    }

    auto kernelOperation = KERNEL_AFFINE_MULTIBIAS;
    auto kernelMode = KernelMode { input.Mode, weights->Mode, biases->Mode };
    auto& affineKernel = AccelerationDetector::GetKernelMap<AffineKernel>(
            static_cast<kernel_op>(kernelOperation), kernelMode);
    return std::make_unique<AffineFunctionMulti>(input, output,
            std::move(weights), std::move(biases),
            std::move(weightScales), affineKernel);
}

std::unique_ptr<const AffineFunction> AffineFunction::Create(const Tensor& input, const Tensor& output,
        void const * layerDetails, const LayerValidator& validatorIn)
{
    auto operation = validatorIn.Operation;
    Shape biasShape;
    auto weightShape = (operation == INTEL_RECURRENT)
        ? Shape{GNA_TENSOR_HW, output.Dimensions.at('W'),
                input.Dimensions.at('W') + output.Dimensions.at('W')}
        : Shape{GNA_TENSOR_HW, output.Dimensions.at('H'), input.Dimensions.at('H')};
    uint32_t biasVectorIndex = 0;
    std::unique_ptr<const WeightTensor> weights;
    std::unique_ptr<const Tensor> weightScales;
    std::unique_ptr<const BiasTensor> biases;
    uint32_t weightMode = 0;
    uint32_t biasMode = 0;
    void* weightsBuffer = nullptr;
    void* biasesBuffer = nullptr;

    switch (operation)
    {
    case INTEL_AFFINE:
    case INTEL_AFFINE_DIAGONAL:
    case INTEL_RECURRENT:
    {
        const intel_affine_func_t* affine = nullptr;
        if (INTEL_RECURRENT == operation)
        {
            affine = &static_cast<const nn_layer_recurrent*>(layerDetails)->affine;
            biasShape = Shape{GNA_TENSOR_H, output.Dimensions.at('W')};
        }
        else
        {
            biasShape = Shape{GNA_TENSOR_H, output.Dimensions.at('H')};
            affine = &static_cast<const nn_layer_affine*>(layerDetails)->affine;
        }
        weightMode = affine->nBytesPerWeight;
        weightsBuffer = affine->pWeights;
        biasMode = affine->nBytesPerBias;
        biasesBuffer = affine->pBiases;
        break;
    }
    case INTEL_AFFINE_MULTIBIAS:
    {
        auto affine = &static_cast<const nn_layer_affine_multi*>(layerDetails)->affine;
        weightMode = affine->nBytesPerWeight;
        weightsBuffer = affine->pWeights;
        biasShape = Shape{GNA_TENSOR_HW, output.Dimensions.at('H'), affine->biasVectorCount};
        biasVectorIndex = affine->biasVectorIndex;
        biasMode = affine->nBytesPerBias;
        biasesBuffer = affine->pBiases;

        // GNA 2.0 backward compatibility only
        if (GNA_INT8 == static_cast<gna_data_mode>(weightMode)
            && GNA_INT16 == input.Mode)
        {
            weightScales = std::make_unique<const Tensor>(
                    Shape(GNA_TENSOR_H, output.Dimensions.at('H')),
                          GNA_DATA_RICH_FORMAT, affine->weightScaleFactors,
                          Validator{ validatorIn, AffineFunctionMulti::Capabilities });
            Expect::ValidBuffer(*weightScales);
        }
        break;
    }
    default:
        throw GnaException(Gna2StatusXnnErrorLyrOperation);
    }

    weights = std::make_unique<const WeightTensor>(weightShape, weightMode,
        weightsBuffer, validatorIn);
    biases = std::make_unique<const BiasTensor>(biasShape, biasVectorIndex,
        biasMode, biasesBuffer, validatorIn);
    KernelMode kernelMode = { input.Mode, weights->Mode, biases->Mode };
    auto& affineKernel = AccelerationDetector::GetKernelMap<AffineKernel>(
                                static_cast<kernel_op>(operation), kernelMode);

    // TODO:3: only affine has active list kernel, recurrent has its own kernel

    switch (operation)
    {
     case INTEL_AFFINE:
     case INTEL_AFFINE_DIAGONAL:
     case INTEL_RECURRENT:
         return std::make_unique<AffineFunctionSingle>(input, output,
             std::move(weights), std::move(biases), affineKernel,
             AccelerationDetector::GetKernelMap<AffineActiveListKernel>(KERNEL_AFFINE_AL, kernelMode));
     case INTEL_AFFINE_MULTIBIAS:
         return std::make_unique<AffineFunctionMulti>(input, output,
             std::move(weights), std::move(biases), std::move(weightScales), affineKernel);
     default:
        throw GnaException(Gna2StatusXnnErrorLyrOperation);
     }
}

AffineFunction::AffineFunction(const KernelMap<AffineKernel>& kernelsIn,
    std::unique_ptr<const WeightTensor> weights, std::unique_ptr<const BiasTensor> biases) :
    Weights{ std::move(weights) },
    Biases{ std::move(biases) },
    kernels{kernelsIn}
{
}

bool AffineFunction::HasGroupedBias(const Gna2Operation& operation)
{
    const auto& biasTensor = *operation.Operands[3];
    return biasTensor.Mode == Gna2TensorModeDefault
        && biasTensor.Shape.NumberOfDimensions == 2
        && biasTensor.Type == Gna2DataTypeInt32
        && nullptr != operation.Parameters[0] // bias mode
        && *static_cast<Gna2BiasMode *>(operation.Parameters[0]) == Gna2BiasModeGrouping;
}

std::unique_ptr<const AffineConfig> AffineFunction::GetRequestConfig(const BaseAddress&inputs, const BaseAddress& outputs) const
{
    return std::make_unique<const AffineConfig>(inputs, outputs, hiddenConfig.get());
}

void AffineFunction::ComputeHidden(AccelerationMode accel, ExecutionConfig const & execution) const
{
    auto kernelConfig = AffineConfig{hiddenConfig.get(), execution};

    kernels.at(accel)(&kernelConfig);
}

void AffineFunction::Compute(const LayerConfiguration& layerConfiguration, AccelerationMode accel,
    ExecutionConfig const & execution) const
{
    auto kernelConfig = AffineConfig{layerConfiguration.Configs.Affine.get(), execution};

    kernels.at(accel)(&kernelConfig);
}

AffineFunctionSingle::AffineFunctionSingle(const Tensor& inputTensor, const Tensor& outputTensor,
    std::unique_ptr<const WeightTensor> weights, std::unique_ptr<const BiasTensor> biases,
    const KernelMap<AffineKernel>& kernelsIn,
    const KernelMap<AffineActiveListKernel>& kernelsAlIn) :
    AffineFunction(kernelsIn, std::move(weights), std::move(biases)),
    kernelsAl{kernelsAlIn}
{
    //// TODO:3: move to layer/hw capabilities as this differ for hws
    //Expect::True(GNA_INT32 == Biases->Mode, Gna2StatusXnnErrorBiasBytes);
    //Expect::True(GNA_DATA_RICH_FORMAT == Biases->Mode, Gna2StatusXnnErrorBiasBytes);
     hiddenConfig = std::make_unique<const AffineConfig>(
         outputTensor.Dimensions.at('H'), inputTensor.Dimensions.at('W'), inputTensor.Dimensions.at('H'),
         inputTensor.Buffer, outputTensor.Buffer, *Weights, *Biases, nullptr, 0, Biases->Mode.Size);
}

void AffineFunctionSingle::Compute(const LayerConfiguration& layerConfiguration, AccelerationMode accel,
    ExecutionConfig const & execution) const
{
    auto kernelConfig = AffineConfig{layerConfiguration.Configs.Affine.get(), execution};

    if (layerConfiguration.ActList)
    {
        auto alConfig = AffineConfigAl{layerConfiguration.ActList->Indices, layerConfiguration.ActList->IndicesCount};
        kernelsAl.at(accel)(&kernelConfig, &alConfig);
    }
    else
    {
        kernels.at(accel)(&kernelConfig);
    }
}

AffineFunctionMulti::AffineFunctionMulti(const Tensor& inputTensor, const Tensor& outputTensor,
    std::unique_ptr<const WeightTensor> weights, std::unique_ptr<const BiasTensor> biases,
    std::unique_ptr<const Tensor> weightScaleFactors,
    const KernelMap<AffineKernel>& kernelsIn) :
    AffineFunction(kernelsIn, std::move(weights), std::move(biases)),
    WeightScaleFactors{ std::move(weightScaleFactors) }
{
    hiddenConfig = std::make_unique<const AffineConfig>(AffineConfig(
        outputTensor.Dimensions.at('H'), inputTensor.Dimensions.at('W'), inputTensor.Dimensions.at('H'),
        inputTensor.Buffer, outputTensor.Buffer, *Weights, ( WeightScaleFactors ? static_cast<const void*>(*WeightScaleFactors) : nullptr ),
        *Biases, Biases->Dimensions.at('W'), Biases->Mode.Size));
}
