/*
 INTEL CONFIDENTIAL
 Copyright 2017-2020 Intel Corporation.

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

#include "IModelSetup.h"
#include "DeviceController.h"

class ModelSetup : public IModelSetup
{
public:
    uint32_t ModelId(uint32_t /*modelIndex*/) const override
    {
        return modelId;
    }

    uint32_t ConfigId(uint32_t /*modelIndex*/, uint32_t /*configIndex*/) const override
    {
        // this one model setup has only one Request Configuration
        return configId;
    }

    ModelSetup(DeviceController & deviceCtrl, const void* referenceOutputIn);

    virtual ~ModelSetup() = default;

protected:

    void checkReferenceOutput(uint32_t modelIndex, uint32_t configIndex) const override;

    DeviceController & deviceController;
    uint32_t modelId;
    uint32_t configId;

    void * inputBuffer = nullptr;
    void * outputBuffer = nullptr;

    const void* const referenceOutput;
};
