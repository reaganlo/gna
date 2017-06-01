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

Weight1B::Weight1B(uint32_t size, const void *weights) :
    Weights{static_cast<const uint8_t*>(weights)}
{
    Expect::ValidBuffer(Weights);
    Expect::True(sizeof(uint8_t) == size, XNN_ERR_WEIGHT_BYTES);
}

Weight2B::Weight2B(uint32_t size, const void *weights) :
    Weights{static_cast<const uint16_t*>(weights)}
{
    Expect::ValidBuffer(Weights);
    Expect::True(sizeof(uint16_t) == size, XNN_ERR_WEIGHT_BYTES);
}

BiasSimple::BiasSimple(uint32_t size, const void *biases) :
    Biases{static_cast<const nn_bias_s*>(biases)}
{
    Expect::ValidBuffer(Biases);
    Expect::True(sizeof(nn_bias_s) == size, XNN_ERR_BIAS_BYTES);
}

BiasCompound::BiasCompound(uint32_t size, const void *biases) :
    Biases{static_cast<const nn_bias_c*>(biases)}
{
    Expect::ValidBuffer(Biases);
    Expect::True(sizeof(nn_bias_c) == size, XNN_ERR_BIAS_BYTES);
}

unique_ptr<const AffineFunction> AffineFunction::Create(intel_layer_kind_t const kind, void const * layerDetails,
    AffineBaseConfig const & affineBase)
{
    switch (kind)
    {
    case INTEL_AFFINE:
    case INTEL_AFFINE_DIAGONAL:
        {
            auto affine = &static_cast<const nn_layer_affine*>(layerDetails)->affine;
            auto const mode = static_cast<const WeightMode>(affine->nBytesPerWeight);
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
            auto const mode = static_cast<const WeightMode>(affine->nBytesPerWeight);
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

AffineFunction::AffineFunction(const std::map<const acceleration, const AffineKernel>& kernelsIn) :
    kernels{kernelsIn}
{
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
        const std::map<const acceleration, const AffineActiveListKernel>& kernelsAlIn) :
    AffineFunction(kernelsIn),
    kernelsAl{kernelsAlIn}
{
}

void AffineFunctionSingle::ComputeConfig(const LayerConfiguration& layerConfiguration, acceleration accel,
    uint32_t *saturationCount, KernelBuffers *fvBuffers) const
{
    auto kernelConfig = AffineConfig{layerConfiguration.Configs.Affine.get(), saturationCount, fvBuffers};

    if (layerConfiguration.ActiveList)
    {
        auto alConfig = AffineConfigAl{layerConfiguration.ActiveList->Indices, layerConfiguration.ActiveList->IndicesCount};
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
    AffineFunctionSingle(kernelsIn, kernelsAlIn),
    Weight2B{affine->nBytesPerWeight, affine->pWeights},
    BiasSimple{affine->nBytesPerBias, affine->pBiases}
{
    hiddenConfig = make_unique<const AffineConfig>(affineBase.OutputElementCount, affineBase.InputVectorCount,
        affineBase.InputElementCount, affineBase.Inputs, affineBase.Outputs, Weights, Biases, nullptr, 0);
}

const void * AffineFunctionSingle2B::GetWeights() const
{
    return Weights;
}

const void * AffineFunctionSingle2B::GetBiases() const
{
    return Biases;
}

WeightMode AffineFunctionSingle2B::GetWeightMode() const
{
    return Weight2B::Mode;
}

AffineFunctionSingle1B::AffineFunctionSingle1B(const nn_func_affine *affine, AffineBaseConfig const & affineBase,
    const std::map<const acceleration, const AffineKernel>& kernelsIn,
    const std::map<const acceleration, const AffineActiveListKernel>& kernelsAlIn) :
    AffineFunctionSingle(kernelsIn, kernelsAlIn),
    Weight1B{affine->nBytesPerWeight, affine->pWeights},
    BiasCompound{affine->nBytesPerBias, affine->pBiases}
{
    hiddenConfig = make_unique<const AffineConfig>(affineBase.OutputElementCount, affineBase.InputVectorCount,
        affineBase.InputElementCount, affineBase.Inputs, affineBase.Outputs, Weights, Biases, nullptr, 0);
}

const void * AffineFunctionSingle1B::GetWeights() const
{
    return Weights;
}

const void * AffineFunctionSingle1B::GetBiases() const
{
    return Biases;
}

WeightMode AffineFunctionSingle1B::GetWeightMode() const
{
    return Weight1B::Mode;
}

AffineFunctionMulti::AffineFunctionMulti(const nn_func_affine_multi *affine,
    const std::map<const acceleration, const AffineKernel>& kernelsIn) :
    AffineFunction(kernelsIn),
    BiasSimple{sizeof(nn_bias_s), affine->pBiases},
    BiasVectorCount{affine->biasVectorCount},
    BiasVectorIndex{affine->biasVectorIndex}
{
    Expect::InRange(BiasVectorCount, 1, XNN_N_GROUP_MAX, XNN_ERR_GROUPING);
    Expect::InRange(BiasVectorIndex, 0, BiasVectorCount - 1, XNN_ERR_GROUPING);
}

AffineFunctionMulti2B::AffineFunctionMulti2B(const nn_func_affine_multi *affine, AffineBaseConfig const & affineBase,
        const std::map<const acceleration, const AffineKernel>& kernelsIn) :
    AffineFunctionMulti(affine, kernelsIn),
    Weight2B{affine->nBytesPerWeight, affine->pWeights}
{
    hiddenConfig = make_unique<const AffineConfig>(affineBase.OutputElementCount, affineBase.InputVectorCount,
        affineBase.InputElementCount, affineBase.Inputs, affineBase.Outputs, Weights, Biases, 
        GetMultibias(), BiasVectorCount);
}

const void * AffineFunctionMulti2B::GetWeights() const
{
    return Weights;
}

const void * AffineFunctionMulti2B::GetBiases() const
{
    return Biases;
}

WeightMode AffineFunctionMulti2B::GetWeightMode() const
{
    return Weight2B::Mode;
}

const nn_bias_s * const AffineFunctionMulti2B::GetMultibias() const
{
    return static_cast<const nn_bias_s* const>(GetBiases()) + BiasVectorIndex;
}

AffineFunctionMulti1B::AffineFunctionMulti1B(const nn_func_affine_multi *affine, AffineBaseConfig const & affineBase,
        const std::map<const acceleration, const AffineKernel>& kernelsIn) :
    AffineFunctionMulti(affine, kernelsIn),
    Weight1B{affine->nBytesPerWeight, affine->pWeights},
    WeightScaleFactors{affine->weightScaleFactors}
{
    Expect::ValidBuffer(WeightScaleFactors);
    hiddenConfig = make_unique<const AffineConfig>(affineBase.OutputElementCount, affineBase.InputVectorCount,
        affineBase.InputElementCount, affineBase.Inputs, affineBase.Outputs, Weights, Biases, 
        GetMultibias(), BiasVectorCount);
}

const void * AffineFunctionMulti1B::GetWeights() const
{
    return Weights;
}

const void * AffineFunctionMulti1B::GetBiases() const
{
    return Biases;
}

WeightMode AffineFunctionMulti1B::GetWeightMode() const
{
    return Weight1B::Mode;
}

const nn_bias_s * const AffineFunctionMulti1B::GetMultibias() const
{
    return static_cast<const nn_bias_s* const>(GetBiases()) + BiasVectorIndex;
}

const unique_ptr<const ActivationFunction> ActivationFunction::Create(const nn_func_pwl * const pwl,
    const bool mandatory, int32_t const * const Inputs, const PwlOutputConfig& outputConfig)
{
    if (mandatory || IsActivationFunctionEnabled(pwl))
    {
        return make_unique<ActivationFunction>(pwl, Inputs, outputConfig);
    }
    else
    {
        return unique_ptr<const ActivationFunction>(nullptr);
    }
}

ActivationFunction::ActivationFunction(const nn_func_pwl *pwl, int32_t const * const Inputs,
    const PwlOutputConfig& outputConfig) :
    SegmentCount{pwl->nSegments},
    Segments{static_cast<nn_pwl_seg*>(pwl->pSegments)},
    Pwl{Inputs, Segments, SegmentCount},
    Kernels{ AccelerationDetector::GetKernelMap<PwlKernel>()},
    OutputConfig{outputConfig}
{
    Expect::ValidBuffer(Segments, XNN_ERR_PWL_DATA);
    Expect::InRange(SegmentCount, SegmentCountMin, SegmentCountMax, XNN_ERR_PWL_SEGMENTS);
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
