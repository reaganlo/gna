/*
 INTEL CONFIDENTIAL
 Copyright 2017 Intel Corporation.

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

#include "RecurrentLayer.h"

#include "AccelerationDetector.h"
#include "ActivationFunction.h"
#include "AffineFunctions.h"
#include "Bias.h"
#include "DataMode.h"
#include "Expect.h"
#include "GnaException.h"
#include "Layer.h"
#include "LayerConfiguration.h"
#include "LayerInput.h"
#include "LayerOutput.h"
#include "Tensor.h"
#include "Weight.h"

#include "gna-api.h"
#include "gna-api-status.h"
#include "gna-api-types-xnn.h"

#include <algorithm>
#include <memory>

namespace GNA
{
class BaseValidator;
}

using namespace GNA;

RecurrentLayer::RecurrentLayer(const nn_layer& layer, const BaseValidator& validatorIn) :
    AffineBaseLayer(layer, { RecurrentTransform }, validatorIn)
{
}

RecurrentLayer::RecurrentLayer(const Gna2Operation& operation, const BaseValidator& validatorIn) :
    AffineBaseLayer(operation, { RecurrentTransform }, validatorIn)
{
    const std::function<void()> command = [&]()
    {
        const auto weights = Shape::Create(operation.Operands[WeightOperandIndex]->Shape, GNA_TENSOR_HW);
        const auto expectedWeights = Shape{ GNA_TENSOR_HW, Output.Dimensions.at('W'), Output.Dimensions.at('W') + Input.Dimensions.at('W') };
        weights.ExpectEqual(expectedWeights);
    };
    ModelErrorHelper::ExecuteForModelItem(command, WeightOperandIndex);
}

DataConfig RecurrentLayer::GetDataMode() const
{
    auto affineTransform = Transforms.Get<AffineFunction>(RecurrentTransform);
    return AffineBaseLayer::getDataMode(affineTransform);
}
