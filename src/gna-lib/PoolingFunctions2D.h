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

#include "Capabilities.h"
#include "Component.h"
#include "OperationConfig.h"
#include "PoolingMode.h"
#include "Transform.h"
#include "XnnKernel.h"

#include <memory>

namespace GNA
{
class FullCapabilitiesMap;
template<typename T> struct SetLimits;

class PoolingFunction2D : public Transform<PoolingConfig2D, PoolingKernel2D>
{
public:
    static std::unique_ptr<PoolingFunction2D> Create(
        const TransformFactoryConfig& config,
        const OperationConfig& operation);

    PoolingFunction2D(const BaseTransformConfig<PoolingKernel2D>& config,
        PoolingMode mode, std::unique_ptr<const Component> window,
        std::unique_ptr<const Component> stride);

    ~PoolingFunction2D() = default;

    virtual bool Is1D() const override
    {
        return is1D;
    }

    const KernelPoolingMode Mode;

    std::unique_ptr<const Component> Window;

    std::unique_ptr<const Component> Stride;

protected:
    static const FullCapabilitiesMap windowLimits;
    static const FullCapabilitiesMap strideLimits;
    static const SetLimits<KernelPoolingMode> modeLimits;
    static const FullCapabilitiesMap outputCapabilities;

    static std::unique_ptr<PoolingFunction2D> create(
        const TransformFactoryConfig& config,
        const OperationConfig& operation);

    bool is1D = false;
};

}
