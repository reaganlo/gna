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

#pragma once

#include "ActivationFunction.h"
#include "ActivationHelper.h"
#include "ConvolutionalFunctions.h"
#include "Layer.h"
#include "PoolingFunctions.h"

#include <memory>

struct ExecutionConfig;

namespace GNA
{
class BaseValidator;
struct LayerConfiguration;

class CnnLayer : public Layer
{
public:
    CnnLayer(const ApiOperation& apiLayer, const BaseValidator& validatorIn);
    virtual ~CnnLayer() = default;
    virtual void UpdateKernelConfigs(LayerConfiguration& layerConfiguration) const override;

    static std::unique_ptr<Layer> CreateEnforced(const Gna2Operation& operation, const BaseValidator& base_validator);

    static bool IsForced(const Gna2Operation& operation);

    std::unique_ptr<const ConvolutionFunction> Convolution;
    std::unique_ptr<const PoolingFunction> Pooling;
    std::unique_ptr<const ActivationFunction> Activation;

    virtual Tensor const & GetOperand(uint32_t operandIndex) const override;

protected:
    void Init();

    std::unique_ptr<const ConvolutionFunction> GetConvolution(const Gna2Operation& apiOperation) const
    {
        const Tensor* convolutionOutput = &Output;
        // TODO:3: use Input,Output tensor everywhere
        if (ActivationHelper::IsEnabled(apiOperation))
        {
            convolutionOutput = &Output.ScratchPad;
        }
        return ConvolutionFunction::Create(&Input, convolutionOutput,
            apiOperation, *validator);
    }

    void ExpectValid() const;

    std::unique_ptr<const PoolingFunction> GetPooling(const Gna2Operation & apiOperation) const;

private:
    void computePool(const LayerConfiguration& layerConfiguration, AccelerationMode accel, ExecutionConfig const & execution) const;
    void computeHiddenPool(AccelerationMode accel, ExecutionConfig const & execution) const;
    void computeHidden(AccelerationMode accel, ExecutionConfig const & execution) const;
    void computeHiddenPwl(AccelerationMode accel, ExecutionConfig const & execution) const;
    void compute(const LayerConfiguration& layerConfiguration, AccelerationMode accel, ExecutionConfig const & execution) const;
    void computePwl(const LayerConfiguration& layerConfiguration, AccelerationMode accel, ExecutionConfig const & execution) const;

    static const Gna2Operation& getDetails(const Gna2Operation& operation);
};

}
