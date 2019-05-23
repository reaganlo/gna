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

#include "common.h"
#include "gna-api.h"
#include "gna-api-types-xnn.h"

#include <map>
#include <memory>

namespace GNA
{

class LayerValidator;
struct ComponentLimits;

using OperationCapabilityMap = std::map<const gna_device_generation, std::shared_ptr<ComponentLimits>>;

class FullCapabilitiesMap : public std::map<const nn_operation, OperationCapabilityMap>
{
public:
    using std::map<const nn_operation, OperationCapabilityMap>::map;

    gna_tensor_order GetOrder(const LayerValidator& validator) const;

    ComponentLimits * GetLatestCaps(const LayerValidator& validator) const;
};

// TODO:3: Refactor to plugin-able component registration during DLL load/device open
/*class CapabilityMap : private std::map<GnaComponentType, std::map<TransformOperation, FullCapabilitiesMap>>
{
public:
    static void Add(GnaComponentType component, TransformOperation transform,
        FullCapabilitiesMap& caps)
    {
        capabilities()[component][transform] = caps;
    }

    static FullCapabilitiesMap& Get(GnaComponentType component, TransformOperation transform)
    {
        return capabilities()[component][transform];
    }

    CapabilityMap(CapabilityMap const&) = delete;
    void operator=(CapabilityMap const&) = delete;

private:
    CapabilityMap() :
        map()
    {};

    static CapabilityMap& capabilities()
    {
        static CapabilityMap instance;
        return instance;
    }
};*/

}
