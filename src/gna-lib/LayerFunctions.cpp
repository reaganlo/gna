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

#include "LayerFunctions.h"

#include "AccelerationDetector.h"
#include "LayerConfiguration.h"
#include "Validator.h"

using std::make_unique;
using std::unique_ptr;

using namespace GNA;

unique_ptr<const AffineFunction> AffineFunction::Create(intel_layer_kind_t const kind, void const * layerDetails,
    AffineBaseConfig const & affineBase)
{
    switch (kind)
    {
    case INTEL_AFFINE:
    case INTEL_AFFINE_DIAGONAL:
    case INTEL_RECURRENT:
        {
            auto affine = &static_cast<const nn_layer_affine*>(layerDetails)->affine;
            auto const mode = static_cast<const WeightMode>(affine->nBytesPerWeight % 2);
            if (GNA_WEIGHT_2B == mode)
            {
                return make_unique<AffineFunctionSingle2B>(affine, affineBase,
                    AccelerationDetector::GetKernelMap<AffineKernel>(mode, kind),
                    AccelerationDetector::GetKernelMap<AffineActiveListKernel>(mode));
            }
            else
            {
                return make_unique<AffineFunctionSingle1B>(affine, affineBase,
                    AccelerationDetector::GetKernelMap<AffineKernel>(mode, kind),
                    AccelerationDetector::GetKernelMap<AffineActiveListKernel>(mode));
            }
        }
    case INTEL_AFFINE_MULTIBIAS:
        {
            auto affine = &static_cast<const nn_layer_affine_multi*>(layerDetails)->affine;
            auto const mode = static_cast<const WeightMode>(affine->nBytesPerWeight % 2);
            if (GNA_WEIGHT_2B == mode)
            {
                return make_unique<AffineFunctionMulti2B>(affine, affineBase,
                    AccelerationDetector::GetKernelMap<AffineKernel>(mode, kind));
            }
            else
            {
                return make_unique<AffineFunctionMulti1B>(affine, affineBase,
                    AccelerationDetector::GetKernelMap<AffineKernel>(mode, kind));
            }
        }
    default:
        throw GnaException(XNN_ERR_LYR_KIND);
    }
}

AffineFunction::AffineFunction(const std::map<const acceleration, const AffineKernel>& kernelsIn, const WeightMode mode,
    const Weight weights, const Bias biases) :
    Mode{mode},
    Weights{weights},
    Biases{biases},
    kernels{kernelsIn}
{
    Expect::ValidBuffer(Weights);
    Expect::ValidBuffer(Biases);
}

unique_ptr<const AffineConfig> AffineFunction::GetRunConfig(int16_t const * const inputs, int32_t * const outputs) const
{
    return make_unique<const AffineConfig>(hiddenConfig.get(), inputs, outputs);
}

void AffineFunction::ComputeHidden(acceleration accel, uint32_t *saturationCount, KernelBuffers *fvBuffers) const
{
    auto kernelConfig = AffineConfig{hiddenConfig.get(), saturationCount, fvBuffers};

    kernels.at(accel)(&kernelConfig);
}

void AffineFunction::ComputeConfig(const LayerConfiguration& layerConfiguration, acceleration accel,
    uint32_t *saturationCount, KernelBuffers *fvBuffers) const
{
    auto kernelConfig = AffineConfig{layerConfiguration.Configs.Affine.get(), saturationCount, fvBuffers};

    kernels.at(accel)(&kernelConfig);
}

AffineFunctionSingle::AffineFunctionSingle(const std::map<const acceleration, const AffineKernel>& kernelsIn,
        const std::map<const acceleration, const AffineActiveListKernel>& kernelsAlIn, const WeightMode mode,
    const Weight weights, const Bias biases) :
    AffineFunction(kernelsIn, mode, weights, biases),
    kernelsAl{kernelsAlIn}
{
}

void AffineFunctionSingle::ComputeConfig(const LayerConfiguration& layerConfiguration, acceleration accel,
    uint32_t *saturationCount, KernelBuffers *fvBuffers) const
{
    auto kernelConfig = AffineConfig{layerConfiguration.Configs.Affine.get(), saturationCount, fvBuffers};

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

AffineFunctionSingle2B::AffineFunctionSingle2B(const nn_func_affine *affine, AffineBaseConfig const & affineBase,
    const std::map<const acceleration, const AffineKernel>& kernelsIn,
    const std::map<const acceleration, const AffineActiveListKernel>& kernelsAlIn) :
    AffineFunctionSingle(kernelsIn, kernelsAlIn, GNA_WEIGHT_2B, affine->pWeights, affine->pBiases)
{
    Expect::True(sizeof(uint16_t) == affine->nBytesPerWeight, XNN_ERR_WEIGHT_BYTES);
    Expect::True(sizeof(nn_bias_s) == affine->nBytesPerBias, XNN_ERR_BIAS_BYTES);
    hiddenConfig = make_unique<const AffineConfig>(affineBase.OutputElementCount, affineBase.InputVectorCount,
        affineBase.InputElementCount, affineBase.Inputs, affineBase.Outputs, Weights, Biases, nullptr, 0);
}

AffineFunctionSingle1B::AffineFunctionSingle1B(const nn_func_affine *affine, AffineBaseConfig const & affineBase,
    const std::map<const acceleration, const AffineKernel>& kernelsIn,
    const std::map<const acceleration, const AffineActiveListKernel>& kernelsAlIn) :
    AffineFunctionSingle(kernelsIn, kernelsAlIn, GNA_WEIGHT_1B, affine->pWeights, affine->pBiases)
{
    Expect::True(sizeof(uint8_t) == affine->nBytesPerWeight, XNN_ERR_WEIGHT_BYTES);
    Expect::True(sizeof(nn_bias_c) == affine->nBytesPerBias, XNN_ERR_BIAS_BYTES);
    hiddenConfig = make_unique<const AffineConfig>(affineBase.OutputElementCount, affineBase.InputVectorCount,
        affineBase.InputElementCount, affineBase.Inputs, affineBase.Outputs, Weights, Biases, nullptr, 0);
}

AffineFunctionMulti::AffineFunctionMulti(const nn_func_affine_multi *affine,
    const std::map<const acceleration, const AffineKernel>& kernelsIn, const WeightMode mode,
    const Weight weights, const Bias biases) :
    AffineFunction(kernelsIn, mode, weights, biases),
    BiasVectorCount{affine->biasVectorCount},
    BiasVectorIndex{affine->biasVectorIndex}
{
    Expect::InRange(BiasVectorCount, 1, XNN_N_GROUP_MAX, XNN_ERR_GROUPING);
    Expect::InRange(BiasVectorIndex, 0, BiasVectorCount - 1, XNN_ERR_GROUPING);
}

const nn_bias_s * const AffineFunctionMulti::GetMultibias() const
{
    return Biases + BiasVectorIndex;
}

AffineFunctionMulti2B::AffineFunctionMulti2B(const nn_func_affine_multi *affine, AffineBaseConfig const & affineBase,
        const std::map<const acceleration, const AffineKernel>& kernelsIn) :
    AffineFunctionMulti(affine, kernelsIn, GNA_WEIGHT_2B, affine->pWeights, affine->pBiases)
{
    Expect::True(sizeof(uint16_t) == affine->nBytesPerWeight, XNN_ERR_WEIGHT_BYTES);
    hiddenConfig = make_unique<const AffineConfig>(affineBase.OutputElementCount, affineBase.InputVectorCount,
        affineBase.InputElementCount, affineBase.Inputs, affineBase.Outputs, Weights, nullptr,
        GetMultibias(), BiasVectorCount);
}

AffineFunctionMulti1B::AffineFunctionMulti1B(const nn_func_affine_multi *affine, AffineBaseConfig const & affineBase,
        const std::map<const acceleration, const AffineKernel>& kernelsIn) :
    AffineFunctionMulti(affine, kernelsIn, GNA_WEIGHT_1B, affine->pWeights, affine->pBiases),
    WeightScaleFactors{affine->weightScaleFactors}
{
    Expect::True(sizeof(uint8_t) == affine->nBytesPerWeight, XNN_ERR_WEIGHT_BYTES);
    Expect::ValidBuffer(WeightScaleFactors);
    hiddenConfig = make_unique<const AffineConfig>(affineBase.OutputElementCount, affineBase.InputVectorCount,
        affineBase.InputElementCount, affineBase.Inputs, affineBase.Outputs, Weights, WeightScaleFactors,
        GetMultibias(), BiasVectorCount);
}

std::unique_ptr<const ActivationFunction> ActivationFunction::Create(nn_layer_kind layerKind, void const *layerDetails,
    int32_t const * const Inputs, const PwlOutputConfig& outputConfig)
{
    bool mandatory = false;
    const nn_func_pwl *pwl = nullptr;
    switch (layerKind)
    {
    case INTEL_AFFINE:
        /* FALLTHRU */
    case INTEL_AFFINE_DIAGONAL:
        pwl = &static_cast<nn_layer_affine const*>(layerDetails)->pwl;
        break;
    case INTEL_AFFINE_MULTIBIAS:
        pwl = &static_cast<nn_layer_affine_multi const*>(layerDetails)->pwl;
        break;
    case INTEL_CONVOLUTIONAL:
    {
        auto cnn = static_cast<nn_layer_conv const*>(layerDetails);
        if (INTEL_NO_POOLING != cnn->poolType) mandatory = true;
        pwl = &cnn->pwl;
        break;
    }
    case INTEL_RECURRENT:
        mandatory = true;
        pwl = &static_cast<nn_layer_reccurent const*>(layerDetails)->pwl;
        break;
    default:
        throw GnaException{ XNN_ERR_LYR_KIND };
    }

    if (mandatory || IsActivationFunctionEnabled(pwl))
    {
        Expect::ValidBuffer(pwl->pSegments, XNN_ERR_PWL_DATA);
        Expect::InRange(pwl->nSegments, SegmentCountMin, SegmentCountMax, XNN_ERR_PWL_SEGMENTS);
        return make_unique<ActivationFunction>(pwl, Inputs, outputConfig);
    }
    else
    {
        return unique_ptr<const ActivationFunction>(nullptr);
    }
}

inline bool ActivationFunction::IsActivationFunctionEnabled(const intel_pwl_func_t * const pwl)
{
    return (nullptr != pwl->pSegments) && (pwl->nSegments > 0);
}

ActivationFunction::ActivationFunction(const nn_func_pwl *pwl, int32_t const * const Inputs,
    const PwlOutputConfig& outputConfig) :
    SegmentCount{pwl->nSegments},
    Segments{static_cast<nn_pwl_seg*>(pwl->pSegments)},
    Pwl{Inputs, outputConfig.elementCount, Segments, SegmentCount},
    Kernels{ AccelerationDetector::GetKernelMap<PwlKernel>()},
    OutputConfig{outputConfig}
{
}

void ActivationFunction::ComputeHidden(acceleration accel, uint32_t *saturationCount) const
{
    auto outConfig = PwlOutputConfig(&OutputConfig, saturationCount);
    Kernels.at(accel)(&Pwl, &outConfig);
}

void ActivationFunction::ComputeConfig(const LayerConfiguration& layerConfiguration, acceleration accel,
    uint32_t *saturationCount) const
{
    auto outConfig = PwlOutputConfig(layerConfiguration.Configs.PwlOutput.get(), saturationCount);
    Kernels.at(accel)(&Pwl, &outConfig);
}

unique_ptr<PwlOutputConfig> ActivationFunction::GetOutputConfig(int16_t * const outputsIn) const
{
    auto outputConfig = make_unique<PwlOutputConfig>(OutputConfig);
    outputConfig->output = outputsIn;
    return move(outputConfig);
}
