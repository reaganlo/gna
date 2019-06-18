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

#include "Bias.h"
#include "Capabilities.h"
#include "Component.h"
#include "ConvolutionalFunctions.h"
#include "OperationConfig.h"
#include "Transform.h"
#include "XnnKernel.h"

#include <memory>

namespace GNA
{
class LayerValidator;

struct ConvolutionFunction2D : public Transform<ConvolutionConfig2D, ConvolutionKernel2D>
{
    static std::unique_ptr<ConvolutionFunction2D> Create(
        const TransformFactoryConfig & config,
        const OperationConfig& operationConfig);

    ConvolutionFunction2D(const BaseTransformConfig<ConvolutionKernel2D> & config,
        std::unique_ptr<const FiltersTensor> filters,
        std::unique_ptr<const BiasTensor> biases,
        std::unique_ptr<const Component> stride,
        std::unique_ptr<const Component> padding);

    virtual ~ConvolutionFunction2D() = default;

    virtual Tensor const & GetOperand(uint32_t operandIndex) const override;

    std::unique_ptr<const BiasTensor> Biases;

    std::unique_ptr<const FiltersTensor> Filters;

    std::unique_ptr<const Component> Stride;

    std::unique_ptr<const Component> Padding;

protected:
    static std::unique_ptr<ConvolutionFunction2D> create(
        const TransformFactoryConfig & config,
        const OperationConfig& operationConfig);

    static Shape CalculateBiasShape(Gna2BiasMode mode, uint32_t filterCount, Shape const & outputShape);

    static std::unique_ptr<const BiasTensor> CreateBiasTensor(
        Gna2Tensor const & apiTensor, Gna2BiasMode biasMode, uint32_t filtersCount,
        Shape const & outputShape, const LayerValidator & validatorIn);

    static Shape GetOutputShape(Shape const & inputShape,
        Shape const & filerShape, Shape const & strideShape, Shape const & paddingShape);

    static const FullCapabilitiesMap strideLimits;
    static const FullCapabilitiesMap paddingLimits;
    static const FullCapabilitiesMap outputCapabilities;
};
}
