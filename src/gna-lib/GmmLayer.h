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

#pragma once

#include "Bias.h"
#include "KernelArguments.h"
#include "Layer.h"
#include "Weight.h"
#include "gmm.h"
#include "GmmLayerCapabilities.h"

#include <cstdint>
#include <map>

namespace GNA
{
class BaseValidator;
struct ActiveList;
struct LayerConfiguration;

// GMM Calculation configuration
class GmmOperation : public Layer
{
public:
    GmmOperation(const ApiOperation& layer, const BaseValidator& validatorIn);

    virtual ~GmmOperation() = default;

    virtual Tensor const & GetOperand(uint32_t operandIndex) const override;

    virtual void VerifyHas1BInputAnd2BWeight() override;
};

class GmmFunction : public TransformAl<GmmConfig, GmmMaxMix, GmmMaxMixActiveList>
{
public:
    static std::unique_ptr<GmmFunction> Create(
        const TransformFactoryConfig& config,
        const OperationConfig& operation);

    virtual ~GmmFunction() = default;

    Tensor const& GetOperand(uint32_t operandIndex) const override;

    void ValidateActiveList(ActiveList const & activeList) const override;

    virtual DataConfig GetDataMode() const = 0;

    std::unique_ptr<const WeightTensor> Means;
    std::unique_ptr<const WeightTensor> InverseCovariances;
    std::unique_ptr<const BiasTensor> GaussianConstants;

    BaseAddress MeanBuffer;
    BaseAddress InverseCovarianceBuffer;
    BaseAddress GaussianConstantBuffer;

    uint32_t InverseCovarianceSize;
    uint32_t const MaximumScore;
    uint32_t MeanSetOffsetSize;
    uint32_t VarSetOffsetSize;
    uint32_t GaussConstSetOffsetSize;
    uint32_t StateCount;

protected:
    GmmFunction(const BaseTransformConfig<GmmMaxMix>& config,
        std::unique_ptr<const WeightTensor> means,
        std::unique_ptr<const WeightTensor> inverseCovariances,
        std::unique_ptr<const BiasTensor> gaussianConstants,
        uint32_t const maximumScore,
        const KernelMap<GmmMaxMixActiveList>& kernelsAl);

    void InitHiddenConfig();

    static const FullCapabilitiesMap & getOutputCapabilities();
};

class GmmFunctionFlat : public GmmFunction
{
public:
    GmmFunctionFlat(const BaseTransformConfig<GmmMaxMix>& config,
        std::unique_ptr<const WeightTensor> means,
        std::unique_ptr<const WeightTensor> inverseCovariances,
        std::unique_ptr<const BiasTensor> gaussianConstants,
        uint32_t const maximumScore,
        const KernelMap<GmmMaxMixActiveList>& kernelsAl);

    virtual ~GmmFunctionFlat() = default;

    virtual DataConfig GetDataMode() const override;
};

class GmmFunctionInterleaved : public GmmFunction
{
public:
    GmmFunctionInterleaved(const BaseTransformConfig<GmmMaxMix>& config,
        std::unique_ptr<const WeightTensor> interleavedData,
        uint32_t const maximumScore,
        const KernelMap<GmmMaxMixActiveList>& kernelsAl);

    virtual ~GmmFunctionInterleaved() = default;

    virtual DataConfig GetDataMode() const override;
};


}
