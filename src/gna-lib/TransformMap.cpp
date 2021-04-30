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

#include "TransformMap.h"

#include "ActivationFunction.h"
#include "AffineFunctions.h"
#include "ConvolutionalFunctions2D.h"
#include "GmmLayer.h"
#include "PoolingFunctions2D.h"
#include "RecurrentFunction.h"
#include "Transform.h"

using namespace GNA;

BaseTransform * TransformList::Emplace(TransformOperation operation,
    const TransformFactoryConfig& config,
    const OperationConfig& operationConfig)
{
    switch (operation)
    {
    case AffineTransform:
    case AffineDiagonalTransform:
        return emplace(AffineFunction::Create(config, operationConfig));
    case RecurrentTransform:
        return emplace(RecurrentFunction::Create(config, operationConfig));
    case ActivationTransform:
        return emplace(ActivationFunction::Create(config));
    case ConvolutionalTransform2D:
        return emplace(ConvolutionFunction2D::Create(config, operationConfig));
    case PoolingTransform2D:
        return emplace(PoolingFunction2D::Create(config, operationConfig));
    case GmmTransform:
        return emplace(GmmFunction::Create(config, operationConfig));
    default:
        throw GnaException(Gna2StatusXnnErrorLyrOperation);
    }
}

// Emplaces transform only if transform is enabled, returns current last transform
BaseTransform * TransformList::emplace(std::unique_ptr<BaseTransform>&& transform)
{
    if (transform)
    {
        // no option for inserting same transform type twice
        if (findTransform(transform->Operation) != __TransformList::cend())
        {
            throw GnaException(Gna2StatusXnnErrorLyrCfg);
        }

        // transform is disabled
        if (!transform)
            return __TransformList::back().get();

        // place transform at the end
        __TransformList::emplace_back(std::move(transform));
    }
    return __TransformList::back().get();
}

__TransformList::const_iterator TransformList::findTransform(TransformOperation transformOperation) const
{
    return std::find_if(__TransformList::cbegin(), __TransformList::cend(),
        [transformOperation](const std::unique_ptr<BaseTransform>& t)
    {
        return t->Operation == transformOperation;
    });
}
