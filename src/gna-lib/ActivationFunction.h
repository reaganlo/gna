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

#include "Expect.h"
#include "DataMode.h"
#include "KernelArguments.h"
#include "Tensor.h"
#include "Transform.h"
#include "XnnKernel.h"

#include "common.h"
#include "gna-api-types-xnn.h"
#include "pwl.h"

#include <cstdint>
#include <memory>

namespace GNA
{

class FullCapabilitiesMap;

class ActivationFunction : public Transform<ActivationConfig, ActivationKernel>
{
public:
    static std::unique_ptr<ActivationFunction> Create(const TransformFactoryConfig& config);

    static inline bool IsEnabled(const intel_pwl_func_t * const pwl)
    {
        return (nullptr != pwl && nullptr != pwl->pSegments) && (pwl->nSegments > 0);
    }

    static inline bool IsEnabled(const Gna2Tensor& activation)
    {
        return activation.Type == Gna2DataTypePwlSegment &&
            activation.Mode == Gna2TensorModeDefault &&
            nullptr != activation.Data &&
            activation.Shape.NumberOfDimensions == 1 &&
            activation.Shape.Dimensions[0] > 0;
    }

    static inline bool IsEnabled(const nn_layer *layer)
    {
        Expect::NotNull(layer);
        return IsEnabled(TransformFactoryConfig::GetActivation(layer->pLayerStruct, layer->operation));
    }

    void UpdateActiveOutputCount(std::unique_ptr<BaseConfig> configs[], uint32_t outputCount) const
    {
        auto config = GetConfig(configs);
        config->Transform.ElementCount = outputCount;
    }

    ActivationFunction(const BaseTransformConfig<ActivationKernel>& config,
                        DataMode mode, std::unique_ptr<Tensor> pwl);
    ActivationFunction() = delete;
    virtual ~ActivationFunction() = default;

    std::unique_ptr<Tensor> Segments;
    PwlCached const Pwl;

protected:
    static PwlCached createPwlCached(const gna_data_mode mode,
        nn_pwl_seg const * const segmentsIn, uint32_t segmentCountIn);

    static const FullCapabilitiesMap capabilities;
    static const FullCapabilitiesMap outputCapabilities;
};

}
