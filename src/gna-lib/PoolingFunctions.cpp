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

#include "PoolingFunctions.h"

#include "AccelerationDetector.h"
#include "Expect.h"

using std::make_unique;
using std::unique_ptr;
using std::move;

using namespace GNA;

const std::map<const nn_operation, const ShapeLimits> PoolingFunction::windowLimits =
{
    {INTEL_CONVOLUTIONAL,
        {{GNA_DIM_W, {CNN_POOL_SIZE_MIN, CNN_POOL_SIZE_MAX, 1, CNN_ERR_POOL_SIZE}}}
    },
};

const std::map<const nn_operation, const ShapeLimits> PoolingFunction::strideLimits =
{
    {INTEL_CONVOLUTIONAL,
        {{GNA_DIM_W, {CNN_POOL_SIZE_MIN, CNN_POOL_SIZE_MAX, 1, CNN_ERR_POOL_STRIDE}}}
    },
};

unique_ptr<const PoolingFunction> PoolingFunction::Create(void const * layerDetails,
    const Shape & inputDimensions, const LayerValidator& validatorIn, gna_data_mode inputMode)
{
    Shape window;
    Shape stride;
    nn_pool_type type = INTEL_NO_POOLING;

    switch (validatorIn.Operation)
    {
    case INTEL_CONVOLUTIONAL:
    {
        auto cnn = static_cast<const nn_layer_conv*>(layerDetails);
        type = cnn->poolType;
        stride[GNA_DIM_W] = cnn->nPoolStride;
        window[GNA_DIM_W] = cnn->nPoolSize;

        break;
    }
    default:
        throw GnaException(XNN_ERR_LYR_OPERATION);
    }

    if (INTEL_NO_POOLING != type)
    {
        switch (validatorIn.Operation)
        {
        case INTEL_CONVOLUTIONAL:
            return make_unique<const PoolingFunction>(validatorIn.Operation, inputDimensions, window,
                stride, type,
                AccelerationDetector::GetKernelMap<ConvolutionPoolingKernel>(KERNEL_POOLING, inputMode));
        default:
            throw GnaException(XNN_ERR_LYR_OPERATION);
        }
    }
    else
    {
       return unique_ptr<const PoolingFunction>(nullptr);
    }
}

// TODO:3: Each transform/function should be independent - have its own input/output component
// to be able to compose different macro layers from this primitives
// TODO:3: create base Function class
PoolingFunction::PoolingFunction(nn_operation const operation, const Shape& inputDimensions,
    const Shape& window, const Shape& stride,
    const nn_pool_type type, const KernelMap<ConvolutionPoolingKernel>& kernelsIn) :
    Type{ type },
    Window{ window },
    Stride{ stride },
    kernels{ kernelsIn },
    hiddenConfig{ make_unique<PoolingConfig>(Type, Window.at(GNA_DIM_W), Stride.at(GNA_DIM_W)) }
{
    Expect::InSet(Type, { INTEL_MAX_POOLING, INTEL_SUM_POOLING }, CNN_ERR_POOL_TYPE);
    // TODO:3: use ShapeIsValid where applicable
    Expect::ShapeIsValid(Stride, strideLimits.at(operation));
    Expect::ShapeIsValid(Window, windowLimits.at(operation));

    OutputsPerFilterCount = 1;
    OutputDimensions[GNA_DIM_D] = inputDimensions.at(GNA_DIM_D);
    for (const auto& dim : Stride)
    {
        if (GNA_DIM_D != dim.first)
        {
            // TODO:3: verify if -1 or -Window.dim
            OutputDimensions[dim.first] =  ((inputDimensions.at(dim.first) - 1) / dim.second + 1);
            OutputsPerFilterCount *= OutputDimensions[dim.first];
            Expect::InRange<uint32_t>(OutputDimensions[dim.first], 1, inputDimensions.at(dim.first), CNN_ERR_POOL_SIZE);
        }
    }
}

// TODO:3: extract activation from pooling
void PoolingFunction::Compute(const ConvolutionConfig * convolutionConfig, acceleration accel, int64_t * poolScratchPad,
    const PwlCached * pwl) const
{
    auto poolConfig = PoolingConfig{ hiddenConfig.get(), poolScratchPad };
    kernels.at(accel)(convolutionConfig, &poolConfig, pwl);
}

