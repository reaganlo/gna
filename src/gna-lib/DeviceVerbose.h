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

#ifndef _DEVICEVERBOSE_H
#define _DEVICEVERBOSE_H

#include "Device.h"
#include "Memory.h"
#include "MemoryVerbose.h"

namespace GNA
{
    class DeviceVerbose : public Device
    {
    public:
        DeviceVerbose(gna_device_id deviceId, uint8_t threadCount = 1) :
            Device::Device(deviceId, threadCount)
        { }

        void SetPrescoreScenario(gna_model_id modelId, uint32_t nActions, dbg_action *actions)
        {
            auto memoryId = getMemoryId(modelId);
            auto memory = getMemory(memoryId);
            auto& model = static_cast<CompiledModelVerbose&>(memory->GetModel(modelId));
            model.SetPrescoreScenario(nActions, actions);
        }

        void SetAfterscoreScenario(gna_model_id modelId, uint32_t nActions, dbg_action *actions)
        {
            auto memoryId = getMemoryId(modelId);
            auto memory = getMemory(memoryId);
            auto& model = static_cast<CompiledModelVerbose&>(memory->GetModel(modelId));
            model.SetAfterscoreScenario(nActions, actions);
        }

        std::unique_ptr<Memory> createMemoryObject(const uint32_t requestedSize,
            const uint16_t layerCount, const uint16_t gmmCount)
        {
            return std::make_unique<MemoryVerbose>(requestedSize, layerCount, gmmCount, *ioctlSender);
        }
    };
}
#endif // _DEVICEVERBOSE_H
