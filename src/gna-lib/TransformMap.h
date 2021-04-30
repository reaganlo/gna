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

#include "GnaException.h"
#include "OperationConfig.h"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <vector>

namespace GNA
{

struct TransformFactoryConfig;
class BaseTransform;

using __TransformList =
    std::vector<std::unique_ptr<BaseTransform>>;

class TransformList : public __TransformList
{
public:
    BaseTransform * Emplace(TransformOperation operation, const TransformFactoryConfig& config,
        const OperationConfig& operationConfig);

    template<typename TransformFunction = BaseTransform>
    TransformFunction * GetOptional(TransformOperation operation) const
    {
        try
        {
            const auto transform = findTransform(operation);
            if (transform != __TransformList::cend())
                return static_cast<TransformFunction *>(transform->get());
        }
        catch (const std::out_of_range&)
        {
        }
        // finally:
        return nullptr;
    }

    template<typename TransformFunction = BaseTransform>
    TransformFunction & Get(TransformOperation operation) const
    {
        auto * transform = GetOptional<TransformFunction>(operation);
        Expect::NotNull(transform, Gna2StatusXnnErrorLyrCfg);
        return *(static_cast<TransformFunction *>(transform));
    }

private:
    // Emplaces transform only if transform is enabled, returns current last transform
    BaseTransform * emplace(std::unique_ptr<BaseTransform>&& transform);

    __TransformList::const_iterator findTransform(TransformOperation transformOperation) const;
};

}
