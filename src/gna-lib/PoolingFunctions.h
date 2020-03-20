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

#include "KernelArguments.h"
#include "ParameterLimits.h"
#include "PoolingMode.h"
#include "Shape.h"
#include "XnnKernel.h"

#include "common.h"
#include "gna-api-types-xnn.h"

#include <cstdint>
#include <map>
#include <memory>

namespace GNA
{
class LayerValidator;
struct PwlCached;

struct PoolingFunction
{
    static std::unique_ptr<const PoolingFunction> Create(void const * layerDetails,
        const Shape& inputDimensions, const LayerValidator& validatorIn, gna_data_mode inputMode);

    static std::unique_ptr<const PoolingFunction> Create(Gna2Operation const & apiOperation,
        const Shape & inputDimensions, const LayerValidator & validatorIn, gna_data_mode inputMode);

    PoolingFunction(nn_operation const operation, const Shape& inputDimensions,
        const Shape& window, const Shape& stride, PoolingMode mode,
        const KernelMap<ConvolutionPoolingKernel>& kernelsIn);
    ~PoolingFunction() = default;

    void Compute(const ConvolutionConfig * convolutionConfig, AccelerationMode accel,
                int64_t * poolScratchPad, const PwlCached * pwl) const;

    const KernelPoolingMode Mode;

    //TODO:3: refactor to components
    // Pooling window dimensions (in # of elements).
    const Shape Window;

    // Sizes of Pooling window stride in each dimension (in # of elements).
    const Shape Stride;

    // Dimensions of output tensor after pooling (in # elements).
    Shape OutputDimensions;

    // Total number of elements in output tensor per filter after pooling.
    uint32_t OutputsPerFilterCount;
protected:
    const KernelMap<ConvolutionPoolingKernel>& kernels;
    const  std::unique_ptr<PoolingConfig> hiddenConfig;
    static const std::map<const nn_operation, const ShapeLimits> windowLimits;
    static const std::map<const nn_operation, const ShapeLimits> strideLimits;

    static void ExpectValid(Gna2Operation const & apiOperation);
};

}
