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
#include "AffineFunctions.h"
#include "Layer.h"

#include "common.h"

#include <memory>

namespace GNA
{
class BaseValidator;
struct LayerConfiguration;

class AffineBaseLayer : public Layer
{
public:
    virtual ~AffineBaseLayer() = default;

    virtual Tensor const & GetOperand(uint32_t operandIndex) const override;

    static void *GetGlobal2MBScratchpad();

protected:
    AffineBaseLayer(
            const nn_layer& layer, std::vector<TransformOperation> transforms,
            const BaseValidator& validatorIn);

    AffineBaseLayer(
            const Gna2Operation& operation, std::vector<TransformOperation> transforms,
            const BaseValidator& validatorIn);

    virtual DataConfig GetDataMode() const override;

    template<typename TransformFunction>
    DataConfig getDataMode(TransformFunction transform) const
    {
        auto weightMode = transform->Weights->Mode.Value;
        auto biasMode = transform->Biases->Mode.Value;
        return DataConfig(Input.Mode, weightMode, biasMode, Output.Mode);
    }
};

class AffineLayer : public AffineBaseLayer
{
public:
    AffineLayer(const nn_layer& layer, const BaseValidator& validatorIn);
    AffineLayer(const Gna2Operation& operation, const BaseValidator& validatorIn);
    virtual ~AffineLayer() = default;

    virtual void UpdateKernelConfigs(LayerConfiguration& layerConfiguration) const override;
};

}
