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

#pragma once

#include "DriverInterface.h"

#include <map>
#include <memory>

#include <Windows.h>
#include <SetupApi.h>

#include "GnaDrvApiWinDebug.h"
#include "common.h"
#include "Request.h"
#include "Expect.h"

namespace GNA
{

class WinHandle
{
public:
    WinHandle() :
        deviceHandle(INVALID_HANDLE_VALUE)
    {};

    explicit WinHandle(HANDLE const handle) :
        deviceHandle(handle)
    {};

    ~WinHandle()
    {
        if (INVALID_HANDLE_VALUE != deviceHandle)
        {
            CloseHandle(deviceHandle);
            deviceHandle = INVALID_HANDLE_VALUE;
        }
    }

    WinHandle(const WinHandle &) = delete;
    WinHandle& operator=(const WinHandle&) = delete;

    void Set(HANDLE const handle)
    {
        Expect::Equal(INVALID_HANDLE_VALUE, deviceHandle, GNA_INVALIDHANDLE);
        deviceHandle = handle;
    }

    operator HANDLE() const {
        return deviceHandle;
    }

private:
    HANDLE deviceHandle;
};

class WindowsDriverInterface : public DriverInterface
{
public:
    WindowsDriverInterface();

    virtual void OpenDevice() override;

    virtual void IoctlSend(const GnaIoctlCommand command,
        void * const inbuf, const uint32_t inlen,
        void * const outbuf, const uint32_t outlen) override;

    virtual DriverCapabilities GetCapabilities() const override;

    virtual uint64_t MemoryMap(void *memory, size_t memorySize) override;

    virtual void MemoryUnmap(uint64_t memoryId) override;

    virtual RequestResult Submit(
        HardwareRequest& hardwareRequest, RequestProfiler * const profiler) const override;

protected:
    void createRequestDescriptor(HardwareRequest& hardwareRequest) const;

private:
    WindowsDriverInterface(const WindowsDriverInterface &) = delete;
    WindowsDriverInterface& operator=(const WindowsDriverInterface&) = delete;

    inline void printLastError(DWORD error) const;

    void wait(LPOVERLAPPED const ioctl, const DWORD timeout) const;

    void checkStatus(BOOL ioResult) const;

    void getDeviceCapabilities();

    static const std::map<GnaIoctlCommand, decltype(GNA_IOCTL_NOTIFY)> ioctlCommandsMap;

    std::map<uint64_t, std::unique_ptr<OVERLAPPED>> memoryMapRequests;

    WinHandle deviceHandle;
    WinHandle deviceEvent;
    OVERLAPPED overlapped;

    UINT32 recoveryTimeout = DRV_RECOVERY_TIMEOUT;
};

}
