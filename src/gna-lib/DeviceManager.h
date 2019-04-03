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

#include <map>
#include <memory>

#include "Device.h"

#include "common.h"

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

    Device& GetDevice(gna_device_id deviceId);

    uint32_t GetDeviceCount();

    DeviceVersion GetDeviceVersion(gna_device_id deviceId);

    void SetThreadCount(gna_device_id deviceId, uint32_t threadCount);

    uint32_t GetThreadCount(gna_device_id deviceId);

    void VerifyDeviceIndex(gna_device_id deviceId);

    void OpenDevice(gna_device_id deviceId);

    void CloseDevice(gna_device_id deviceId);

    static constexpr uint32_t DefaultThreadCount = 1;

private:
    DeviceManager();

    std::vector<std::unique_ptr<Device>> deviceMap;

    std::vector<bool> deviceOpenedMap;
};

}
