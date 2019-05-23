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
#include "TransformMap.h"

#include "Address.h"
#include "KernelArguments.h"
#include "Transform.h"
#include "Validator.h"

#include "common.h"
#include "gna-api.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace GNA
{
struct LayerConfiguration;

class AbstractOperation
{
public:
    //TODO:3:P3 remove or change name when API2 reliable enough
    const nn_operation Operation;
protected:
    AbstractOperation(const Gna2Operation& operation):
        Operation{ ToGnaApi1OperationType(operation)}
    {
        //TODO:3:P1 Add operation validation
    }

    AbstractOperation(nn_operation operationType):
        Operation{ operationType }
    {
    }
private:
    static nn_operation ToGnaApi1OperationType(const Gna2Operation& operation)
    {
        //TODO:3:P1: Add remaining cases
        Expect::Equal(operation.Type, Gna2OperationTypeCopy, Gna2StatusNotImplemented);
        return INTEL_COPY;
    }
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

    uint32_t GetOperandSize(GnaComponentType componentType) const;

    static uint32_t GetShapeDimension(const Gna2Shape& shape, uint32_t dimensionIndex)
    {
        Expect::True(dimensionIndex < shape.NumberOfDimensions,
            Gna2StatusModelConfigurationInvalid);
        return shape.Dimensions[dimensionIndex];
    }

protected:
    std::unique_ptr<const LayerValidator> validator;

public:
    const LayerInput Input;
    const LayerOutput Output;

protected:
    Layer(const nn_layer& layer, const BaseValidator& validator,
        const std::vector<TransformOperation>& transforms,
        const BaseAddress& intermediateBuffer);
    Layer(const Gna2Operation& operation, const BaseValidator& validator,
        const std::vector<TransformOperation>& transforms,
        const BaseAddress& intermediateBuffer);
    void compute(const LayerConfiguration* layerConfiguration,
        AccelerationMode accel, ExecutionConfig const & execution) const
    {
        for (const auto& transform : Transforms)
        {
            if (transform)
                transform->Compute(accel, layerConfiguration, execution);
        }
    }

private:
    BaseTransform const * inputTransform = nullptr;
    BaseTransform * outputTransform = nullptr;

    void addBufferAs(const BufferMap& source, GnaComponentType sourceType,
        BufferMap& destination, GnaComponentType destinationType) const;
};

}
