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

#include "gna2-inference-impl.h"

#include <memory>

#include "DeviceLayerSupport.h"
#include "LayerInput.h"
#include "LayerOutput.h"
#include "Transform.h"
#include "TransformMap.h"

#include "Address.h"
#include "KernelArguments.h"
#include "Validator.h"

#include "common.h"
#include "gna-api.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace GNA
{

class BufferMap;
struct LayerConfiguration;

class AbstractOperation
{
public:
    //TODO:3:P3 remove or change name when API2 reliable enough
    const nn_operation Operation;
    const Gna2OperationType OperationNew;
protected:
    AbstractOperation(const Gna2Operation& operation, const BaseValidator& validator) :
        Operation{ toLegacy(operation, validator) },
        OperationNew{ operation.Type }
    {
        //TODO:3:P1 Add operation validation
    }

    AbstractOperation(const nn_layer& layer, const BaseValidator& validator) :
        Operation{ layer.operation },
        OperationNew{ fromLegacy(layer.operation) }
    {
        UNREFERENCED_PARAMETER(validator);
    }
private:
    static nn_operation toLegacy(const Gna2Operation& operation, const BaseValidator& validator);
    static Gna2OperationType fromLegacy(const nn_operation& layerType);
};

class Layer : public AbstractOperation
{
public:
    static std::unique_ptr<Layer> Create(const nn_layer& layer, const BaseValidator& validator);
    static std::unique_ptr<Layer> Create(const Gna2Operation& operation, const BaseValidator& validator);

    template<typename X = const Layer> X* Get() const
    {
        return static_cast<const X*>(this);
    }

    virtual ~Layer() = default;

    // TODO:3: reduce complexity and optimization level to simplify code and logic, use single function
    std::function<void(AccelerationMode accel, ExecutionConfig const & executionConfig)> ComputeHidden;
    std::function<void(LayerConfiguration &layerConfiguration, AccelerationMode accel, ExecutionConfig const & executionConfig)> Compute;

    virtual void UpdateKernelConfigs(LayerConfiguration& layerConfiguration) const;
    virtual DataConfig GetDataMode() const;

    TransformList Transforms;

    BaseTransform const * GetInputTransform() const
    {
        return inputTransform;
    };
    BaseTransform const * GetOutputTransform() const
    {
        return outputTransform;
    };

    virtual Tensor const & GetOperand(uint32_t operandIndex) const;
    Tensor const * TryGetOperand(uint32_t operandIndex) const;

    uint32_t TryGetOperandSize(uint32_t operandIndex) const;

    // verifies if layer has ADL bug workaround needed
    bool Is1BInputAnd2BWeight() const
    {
        return has1BInputAnd2BWeight;
    }

    // verifies and stores info if layer has ADL bug workaround needed
    virtual void VerifyHas1BInputAnd2BWeight();

protected:
    std::unique_ptr<const LayerValidator> validator;

public:
    const LayerInput Input;
    const LayerOutput Output;

protected:
    template <class T>
    Layer(const T& layer, const BaseValidator& validatorIn,
        const std::vector<TransformOperation>& transforms,
        const BaseAddress& intermediateBuffer) :
        AbstractOperation{ layer, validatorIn },
        validator{ std::make_unique<const LayerValidator>(validatorIn, Operation) },
        Input{ layer, *validator },
        Output{ layer, *validator }
    {
        Expect::InRange<uint32_t>(Operation, 0, LAYER_OPERATION_TYPE_COUT - 1, Gna2StatusXnnErrorLyrOperation);

        //TODO:3: uncomment when all layers are Transform-based, remove if below
        //Expect::False(transforms.empty(), Gna2StatusNullArgumentNotAllowed);
        if (false == transforms.empty())
        {
            auto&& commonConfig = TransformFactoryConfig(&Input, &Output, Output.Mode, intermediateBuffer,
                layer, *validator);
            const OperationConfig operationConfig{ layer };
            initTransforms(transforms, commonConfig, operationConfig);
        }

        initComputeFunctions();
    }

    void initTransforms(const std::vector<TransformOperation>& transforms, TransformFactoryConfig& commonConfig,
        const OperationConfig& operationConfig);

    void initComputeFunctions();

    void compute(const LayerConfiguration* layerConfiguration,
        AccelerationMode accel, ExecutionConfig const & execution) const;

    Tensor const & getTransformOperand(TransformOperation operation, uint32_t operandIndex) const;

    BaseTransform const * inputTransform = nullptr;
    BaseTransform * outputTransform = nullptr;

private:
    void addBufferAs(const BufferMap& source, uint32_t sourceType,
        BufferMap& destination, uint32_t destinationType) const;

    // defines layer as ADL bug workaround enabled
    bool has1BInputAnd2BWeight = false;
    // defines ADL bug workaround has been already verified
    bool is1BInputAnd2BWeightVerified = false;
};

}
