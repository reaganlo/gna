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

#include "AffineFunctions.h"

#include "AccelerationDetector.h"
#include "Bias.h"
#include "LayerConfiguration.h"
#include "Weight.h"

using std::make_unique;
using std::unique_ptr;
using std::move;

using namespace GNA;

using AD = GNA::AccelerationDetector;

const FullCapabilitiesMap AffineFunctionMulti::Capabilities =
{
    {INTEL_AFFINE_MULTIBIAS, {
        {GNA_2_0, std::make_shared<TensorLimits>(TensorLimits{
            {GNA_TENSOR_NH},
                {{GNA_DIM_N, {1, XNN_N_GROUP_MAX, 1, XNN_ERR_BIAS_VOLUME}},
                {GNA_DIM_H, {1, XNN_N_IN_ELEMS_MAX, 1, XNN_ERR_BIAS_VOLUME}}},
            {{ GNA_DATA_RICH_FORMAT }, XNN_ERR_BIAS_BYTES }})}
    }}
};


// Could not split into separate methods for each component as multibias weight scaling is using bias' and weights; tensors...
unique_ptr<const AffineFunction> AffineFunction::Create(const Tensor* input, const Tensor* output,
        void const * layerDetails, const LayerValidator& validatorIn)
{
    auto operation = validatorIn.Operation;
    auto dimensions = Shape(input->at(GNA_DIM_N), input->at(GNA_DIM_W), output->at(GNA_DIM_H));
    auto biasTensorDimensions = dimensions;
    uint32_t biasVectorIndex = 0;
    unique_ptr<const WeightTensor> weights;
    unique_ptr<const Tensor> weightScales;
    unique_ptr<const BiasTensor> biases;
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
            affine = &static_cast<const nn_layer_reccurent*>(layerDetails)->affine;
        }
        else
        {
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
        biasTensorDimensions[GNA_DIM_N] = affine->biasVectorCount;
        biasVectorIndex = affine->biasVectorIndex;
        biasMode = affine->nBytesPerBias;
        biasesBuffer = affine->pBiases;

        // GNA 2.0 backward compatibility only
        if (GNA_INT8 == static_cast<gna_data_mode>(weightMode)
            && GNA_INT16 == input->Mode)
        {
            weightScales = make_unique<const Tensor>(Shape(1, 0, output->at(GNA_DIM_H)), GNA_DATA_RICH_FORMAT,
                affine->weightScaleFactors, Validator{ validatorIn, AffineFunctionMulti::Capabilities });
        }
        break;
    }
    default:
        throw GnaException(XNN_ERR_LYR_OPERATION);
    }

    weights = make_unique<const WeightTensor>(dimensions, weightMode,
        weightsBuffer, validatorIn);
    biases = make_unique<const BiasTensor>(biasTensorDimensions, biasVectorIndex,
        biasMode, biasesBuffer, validatorIn);
    KernelMode kernelMode = { input->Mode, weights->Mode, biases->Mode };
    auto& affineKernel = AD::GetKernelMap<AffineKernel>(static_cast<kernel_op>(operation), kernelMode);

    // TODO:3: only affine has active list kernel, recurrent has its own kernel

    switch (operation)
    {
     case INTEL_AFFINE:
     case INTEL_AFFINE_DIAGONAL:
     case INTEL_RECURRENT:
         return make_unique<AffineFunctionSingle>(input->Buffer, output->Buffer,
             input->at(GNA_DIM_N), move(weights), move(biases), affineKernel,
             AD::GetKernelMap<AffineActiveListKernel>(KERNEL_AFFINE_AL, kernelMode));
             // TODO:3: pass Input and Output tensors as arguments to Create, use Input->Mode here and create dimensions locally
     case INTEL_AFFINE_MULTIBIAS:
         return make_unique<AffineFunctionMulti>(input->Buffer, output->Buffer, input->at(GNA_DIM_N),
             move(weights), move(biases), move(weightScales), affineKernel);
     default:
        throw GnaException(XNN_ERR_LYR_OPERATION);
     }
}

AffineFunction::AffineFunction(const KernelMap<AffineKernel>& kernelsIn,
    std::unique_ptr<const WeightTensor> weights, std::unique_ptr<const BiasTensor> biases) :
    Weights{ move(weights) },
    Biases{ move(biases) },
    kernels{kernelsIn}
{
}

unique_ptr<const AffineConfig> AffineFunction::GetRequestConfig(const BaseAddress&inputs, const BaseAddress& outputs) const
{
    return make_unique<const AffineConfig>(inputs, outputs, hiddenConfig.get());
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

AffineFunctionSingle::AffineFunctionSingle(const BaseAddress& input, const BaseAddress& output, const uint32_t vectorCount,
    std::unique_ptr<const WeightTensor> weights, std::unique_ptr<const BiasTensor> biases,
    const KernelMap<AffineKernel>& kernelsIn,
    const KernelMap<AffineActiveListKernel>& kernelsAlIn) :
    AffineFunction(kernelsIn, move(weights), move(biases)),
    kernelsAl{kernelsAlIn}
{
    //// TODO:3: move to layer/hw capabilities as this differ for hws
    //Expect::True(GNA_INT32 == Biases->Mode, XNN_ERR_BIAS_BYTES);
    //Expect::True(GNA_DATA_RICH_FORMAT == Biases->Mode, XNN_ERR_BIAS_BYTES);
     hiddenConfig = make_unique<const AffineConfig>(
         Biases->at(GNA_DIM_H), vectorCount, Weights->at(GNA_DIM_W),
         input, output, *Weights, *Biases, nullptr, 0, Biases->Mode);
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

AffineFunctionMulti::AffineFunctionMulti(const BaseAddress& input, const BaseAddress& output, const uint32_t vectorCount,
    std::unique_ptr<const WeightTensor> weights, std::unique_ptr<const BiasTensor> biases,
    unique_ptr<const Tensor> weightScaleFactors,
    const KernelMap<AffineKernel>& kernelsIn) :
    AffineFunction(kernelsIn, move(weights), move(biases)),
    WeightScaleFactors{ move(weightScaleFactors) }
{
    //// TODO:3: move to layer/hw capabilities as this differ for hws
    //auto mode = DataConfig{ GNA_INT16, Weights->Mode, Biases->Mode, GNA_INT16 };
    //auto support = DataConfig::Capabilities.at(mode).at(INTEL_AFFINE_MULTIBIAS).Api.at(GNA_API_3_0);
  /*  if (GNA_INT8 == Weights->Mode)
    {
        Expect::True(GNA_DATA_RICH_FORMAT == WeightScaleFactors->Mode, XNN_ERR_WEIGHT_BYTES);
        Expect::ValidBuffer(*WeightScaleFactors);
    }*/

    hiddenConfig = make_unique<const AffineConfig>(AffineConfig(
        Biases->at(GNA_DIM_H), vectorCount, Weights->at(GNA_DIM_W),
        input, output, *Weights, ( WeightScaleFactors ? static_cast<const void*>(*WeightScaleFactors) : nullptr ),
        *Biases, Biases->at(GNA_DIM_N), Biases->Mode));
}
