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

#include "Address.h"
#include "DataMode.h"
#include "GnaException.h"
#include "LayerConfiguration.h"
#include "ModelWrapper.h"
#include "Tensor.h"

#include "KernelArguments.h"
#include "XnnKernel.h"

#include "common.h"
#include "gna-api-status.h"
#include "gna-api.h"

#include <memory>
#include <stdexcept>

namespace GNA
{
class LayerValidator;

struct TransformFactoryConfig
{
    const Tensor * input;
    const Tensor * output;
    const DataMode outputMode;
    const BaseAddress outputBuffer;
    const LayerValidator& validator;

    template<class T>
    TransformFactoryConfig(const Tensor *inputIn, const Tensor *outputIn, DataMode outputModeIn,
        BaseAddress outputBufferIn, const T& operation, const LayerValidator& validatorIn) :
        input{ inputIn }, output{ outputIn }, outputMode{ outputModeIn }, outputBuffer{ outputBufferIn },
        validator{ validatorIn }
    {
        InitActivation(operation);
    }

    TransformFactoryConfig(const TransformFactoryConfig&) = delete;
    TransformFactoryConfig() = delete;
    // TODO:P2: Remove Activation and make Transform class generic
    bool HasMandatoryActivation() const;
    bool IsActivationNotSupported() const;
    Gna2Tensor GetActivation() const;
    static Gna2Tensor GetActivation(const void * layerDetails, nn_operation operationType);

protected:
    void InitActivation(const nn_layer& layer);
    void InitActivation(const Gna2Operation& operation);
private:

    bool HasMandatoryActivation(const void * layerDetails) const;

    static bool HasMandatoryActivation(const Gna2Operation& operation);

    static nn_func_pwl GetActivationImpl(const void * layerDetails, nn_operation operationType);

    //TODO:3:P1:Move to operation/model wrapper
    static Gna2Tensor GetActivation(const Gna2Operation& operation);

    bool mandatoryActivation;
    Gna2Tensor activation;
};

template<typename KernelType>
struct BaseTransformConfig
{
    const Tensor * input;
    const Tensor * output; // TODO:3: verify if needed for reference or can be removed/replaced by output shape/mode
    const DataMode outputMode;
    const BaseAddress& outputBuffer; // transform output buffer, usually layer intermediate buffer
    const LayerValidator& validator;
    const KernelMap<KernelType>& kernels;

    BaseTransformConfig(const TransformFactoryConfig& config, const KernelMap<KernelType>& kernelsIn) :
        input{ config.input },
        output{ config.output },
        outputMode{ config.output->Mode },
        outputBuffer{ config.outputBuffer },
        validator{ config.validator },
        kernels{ kernelsIn }
    {}

    BaseTransformConfig(const BaseTransformConfig&) = delete;
    BaseTransformConfig() = delete;
};

class BaseTransform
{
public:
    virtual ~BaseTransform() = default;

    virtual void Compute(AccelerationMode accel, LayerConfiguration const * layerConfiguration,
        ExecutionConfig const & execution) const = 0;

    virtual void UpdateConfigBuffers(std::unique_ptr<BaseConfig> configs[], const BufferMap& buffers) const = 0;
    virtual void SetOutput(const BaseAddress& outputBuffer) = 0;

    const Tensor * const Input;
    std::unique_ptr<Tensor> Output;
    TransformOperation const Operation;

protected:
    BaseTransform(TransformOperation operation, Tensor const * input) :
        Input{input},
        Operation{operation}
    {};
    BaseTransform(const BaseTransform&) = delete;
    BaseTransform(const BaseTransform&&) = delete;
};

template<typename TransformType, typename KernelType> class Transform : public BaseTransform
{
public:
    virtual void Compute(AccelerationMode accel,
        LayerConfiguration const * layerConfiguration, ExecutionConfig const & execution) const
    {
        auto executionConfig = createExecutionConfig(layerConfiguration, execution);
        try
        {
            kernels->at(accel)(executionConfig.get());
        }
        catch (const std::out_of_range&)
        {
            throw GnaException(Gna2StatusNotImplemented);
        }
    }

    virtual void UpdateConfigBuffers(std::unique_ptr<BaseConfig> configs[], const BufferMap& buffers) const
    {
        auto* config = GetConfig(configs);
        config->Update(buffers);
    }

    // set output when transform is final layer transform and uses user provided layer output buffer
    virtual void SetOutput(const BaseAddress& outputBuffer)
    {
        Output->UpdateBuffer(outputBuffer);
        hiddenConfig->Update({{OutputComponent, outputBuffer}});
    }

protected:
    Transform(TransformOperation operation, const KernelMap<KernelType>* kernelsIn, Tensor const * input) :
        BaseTransform{operation, input},
        kernels{kernelsIn}
    {};
    Transform(const Transform&) = delete;
    Transform(const Transform&&) = delete;
    virtual ~Transform() = default;

    std::unique_ptr<BaseConfig> GetRequestConfig(
        const BufferMap& buffers) const
    {
        return std::make_unique<KernelConfig<TransformType>>(*hiddenConfig, buffers);
    }

    inline KernelConfig<TransformType>* GetConfig(std::unique_ptr<BaseConfig> configs[]) const
    {
        auto& config = configs[Operation];
        if (!config)
        {
            config = std::make_unique<KernelConfig<TransformType>>(*hiddenConfig);
        }
        return static_cast<KernelConfig<TransformType>*>(config.get());
    }

    const KernelMap<KernelType>* kernels;
    std::unique_ptr<KernelConfig<TransformType>> hiddenConfig;

private:
   inline std::unique_ptr<ExecutionKernelConfig<TransformType>> createExecutionConfig(
       const LayerConfiguration* layerConfiguration, ExecutionConfig const & execution) const
    {
        if (nullptr == layerConfiguration)
            return std::make_unique<ExecutionKernelConfig<TransformType>>(
                hiddenConfig.get(), execution);
        else
            return std::make_unique<ExecutionKernelConfig<TransformType>>(
                static_cast<KernelConfig<TransformType>*>(layerConfiguration->ConfigList[Operation].get()),
                    execution);
    }
};

}

