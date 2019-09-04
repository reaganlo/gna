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

#include "Device.h"

#include "gna2-common-impl.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace GNA
{

class DeviceManager
{
public:
    static DeviceManager& Get()
    {
        static DeviceManager deviceManager;
        return deviceManager;
    }

    DeviceManager(DeviceManager const&) = delete;
    void operator=(DeviceManager const&) = delete;

    Device& GetDevice(uint32_t deviceIndex);

    uint32_t GetDeviceCount() const;

    DeviceVersion GetDeviceVersion(uint32_t deviceIndex);

    void SetThreadCount(uint32_t deviceIndex, uint32_t threadCount);

    uint32_t GetThreadCount(uint32_t deviceIndex);

    void OpenDevice(uint32_t deviceIndex);

    void CloseDevice(uint32_t deviceIndex);

    Device& GetDeviceForModel(uint32_t modelId);
    Device* TryGetDeviceForModel(uint32_t modelId);

    void AllocateMemory(uint32_t requestedSize, uint32_t * sizeGranted, void **memoryAddress);
    std::pair<bool, std::vector<std::unique_ptr<Memory>>::const_iterator> HasMemory(void * buffer) const;
    void FreeMemory(void * memory);

    void MapMemoryToAll(Memory& memoryObject);
    void UnMapMemoryFromAll(Memory& memoryObject);

    Device& GetDeviceForRequestConfigId(uint32_t requestConfigId);

    Device * TryGetDeviceForRequestConfigId(uint32_t requestConfigId);

    Device& GetDeviceForRequestId(uint32_t requestId);

    const std::vector<std::unique_ptr<Memory>>& GetAllAllocated() const;

    static constexpr uint32_t DefaultThreadCount = 1;

private:
    void UnMapAllFromDevice(Device& device);
    void MapAllToDevice(Device& device);

    static std::unique_ptr<Memory> createMemoryObject(const uint32_t requestedSize);

    static constexpr uint32_t MaximumReferenceCount = 1024;

    struct DeviceContext
    {
        DeviceContext() = default;
        DeviceContext(std::unique_ptr<Device> handle, uint32_t referenceCount);

        std::unique_ptr<Device> Handle;
        uint32_t ReferenceCount;
    };

    DeviceManager();
    void CreateDevice(uint32_t deviceIndex);
    bool IsOpened(uint32_t deviceIndex);
    inline DeviceContext& GetDeviceContext(uint32_t deviceIndex);

    std::map<uint32_t, DeviceContext> devices;

    std::map<uint32_t, HardwareCapabilities> capabilities;

    std::vector<std::unique_ptr<Memory>> memoryObjects;
};

}
