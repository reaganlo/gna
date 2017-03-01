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

#include "RequestConfiguration.h"
#include "GnaException.h"
#include "Validator.h"

using std::make_unique;

using namespace GNA;

void RequestConfiguration::AddBuffer(gna_buffer_type type, uint32_t layerIndex, void *address)
{
    switch(type)
    {
    case GNA_IN:
    case GNA_OUT:
        RequestBuffers[type].emplace_back(type, layerIndex, address);
        break;
    default:
        throw GnaException(GNA_ERR_UNKNOWN);
    }
}

void RequestConfiguration::AddActiveList(uint32_t layerIndex, uint32_t indicesCount, uint32_t *indices)
{
    ActiveLists.emplace_back(layerIndex, indicesCount, indices);
}

void ConfigurationBuffer::validate() const
{
    Validate::IsNull(address);
}

ConfigurationBuffer::ConfigurationBuffer(gna_buffer_type type, uint32_t layerIndex, void* address)
    : type(type), layerIndex(layerIndex), address(address)
{
    validate();
}
