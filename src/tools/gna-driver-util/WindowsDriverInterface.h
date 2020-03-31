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


// TODO extract common part from library and util into separate file to omit redundancy.
#pragma once

#include "DriverInterface.h"
#include "GnaUtilConfig.h"

#include <map>
#include <string>
#include <stdexcept>

#define WIN32_NO_STATUS
#include <Windows.h>
#undef WIN32_NO_STATUS

class WindowsDriverInterface : public DriverInterface
{
public:

    static const int WAIT_FOR_MAP_ITERATIONS = 200;
    static const int WAIT_FOR_MAP_MILLISECONDS = 15;
    static const uint64_t FORBIDDEN_MEMORY_ID = 0;

    HANDLE deviceHandle;
    DriverCapabilities driverCapabilities;
    OVERLAPPED overlapped;

    WindowsDriverInterface();

    bool OpenDevice(std::string devicePath);

    bool discoverDevice(uint32_t deviceIndex) override;

    void getDeviceCapabilities();

    void checkStatus(BOOL ioResult) const;

    void wait(LPOVERLAPPED const ioctl, const DWORD timeout) const;

    RequestResult Submit(
        HardwareRequest& hardwareRequest, const GnaUtilConfig& file) const override;

    uint64_t MemoryMap(void *memory, uint32_t memorySize) override;

    void MemoryUnmap(uint64_t memoryId) override;

    gna_status_t parseHwStatus(uint32_t hwStatus) const;

    void Set(HANDLE const handle)
    {
        deviceHandle = handle;
    }

private:
    UINT32 recoveryTimeout;
    std::map<uint64_t, std::unique_ptr<OVERLAPPED>> memoryMapRequests;
};
