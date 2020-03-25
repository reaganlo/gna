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

#include "gna2-memory-api.h"

#include "ApiWrapper.h"
#include "Device.h"
#include "DeviceManager.h"

#include "gna2-common-api.h"

#include <cstdint>
#include <functional>

using namespace GNA;

GNA2_API enum Gna2Status Gna2MemoryAlloc(
    uint32_t sizeRequested,
    uint32_t * sizeGranted,
    void ** memoryAddress)
{
    const std::function<ApiStatus()> command = [&]()
    {
        auto& deviceManager = DeviceManager::Get();
        deviceManager.AllocateMemory(sizeRequested, sizeGranted, memoryAddress);
        return Gna2StatusSuccess;
    };
    return ApiWrapper::ExecuteSafely(command);
}

GNA2_API enum Gna2Status Gna2MemoryFree(
    void * memory)
{
    const std::function<ApiStatus()> command = [&]()
    {
        DeviceManager::Get().FreeMemory(memory);
        return Gna2StatusSuccess;
    };
    return ApiWrapper::ExecuteSafely(command);
}

GNA2_API enum Gna2Status Gna2MemorySetTag(
    void * memory,
    uint32_t tag)
{
    const std::function<ApiStatus()> command = [&]()
    {
        DeviceManager::Get().TagMemory(memory, tag);
        return Gna2StatusSuccess;
    };
    return ApiWrapper::ExecuteSafely(command);
}