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

#include <memory>

#include "DeviceLayerSupport.h"
#include "LayerInput.h"
#include "LayerOutput.h"
#include "TransformMap.h"

namespace GNA
{
struct LayerConfiguration;

class Layer
{
public:
    static std::unique_ptr<Layer> Create(const nn_layer *layer, const BaseValidator& validator);

    template<typename X = const Layer> X* Get() const
    {
        return static_cast<const X*>(this);
    }

    virtual ~Layer() = default;

    // TODO:3: reduce complexity and optimization level to simplify code and logic, use single function
    std::function<void(acceleration accel, KernelBuffers* kernelBuffers, uint32_t* saturationCount)> ComputeHidden;
    std::function<void(LayerConfiguration &layerConfiguration, acceleration accel, KernelBuffers* kernelBuffers, uint32_t* saturationCount)> Compute;

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


protected:
    std::unique_ptr<const LayerValidator> validator;

public:
    const nn_operation Operation;
    const LayerInput Input;
    const LayerOutput Output;

protected:
    Layer(const nn_layer *layer, const BaseValidator& validator,
        const std::vector<TransformOperation>& transforms,
        const BaseAddress& intermediateBuffer);

    void compute(const LayerConfiguration* layerConfiguration,
        acceleration accel, ExecutionConfig const & execution) const
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
        BufferMap& destination, GnaComponentType destinationType, uint32_t size) const;
};

}
