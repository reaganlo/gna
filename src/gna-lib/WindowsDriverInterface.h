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

#ifdef WIN32

#include "DriverInterface.h"

#include "Request.h"
#include "Expect.h"

#include <map>
#include <memory>
#define WIN32_NO_STATUS
#include <Windows.h>
#undef WIN32_NO_STATUS

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
        Expect::Equal(INVALID_HANDLE_VALUE, deviceHandle, Gna2StatusIdentifierInvalid);
        deviceHandle = handle;
    }

    operator HANDLE() const
    {
        return deviceHandle;
    }

private:
    HANDLE deviceHandle;
};

class OverlappedWithEvent
{
public:
    OverlappedWithEvent();
    OverlappedWithEvent(const OverlappedWithEvent&) = delete;
    OverlappedWithEvent(OverlappedWithEvent&&) = delete;
    OverlappedWithEvent& operator = (const OverlappedWithEvent&) = delete;
    OverlappedWithEvent& operator = (OverlappedWithEvent&&) = delete;
    ~OverlappedWithEvent();
    operator OVERLAPPED*()
    {
        return &overlapped;
    }
private:
    OVERLAPPED overlapped = {};
};

class WindowsDriverInterface : public DriverInterface
{
    static const int WAIT_FOR_MAP_ITERATIONS;
    static const int WAIT_FOR_MAP_MILLISECONDS;
    static const uint64_t FORBIDDEN_MEMORY_ID;
public:
    WindowsDriverInterface();
    virtual ~WindowsDriverInterface() override = default;

    virtual bool OpenDevice(uint32_t deviceIndex) override;

    virtual uint64_t MemoryMap(void *memory, uint32_t memorySize) override;

    virtual void MemoryUnmap(uint64_t memoryId) override;

    virtual RequestResult Submit(
        HardwareRequest& hardwareRequest, RequestProfiler& profiler) const override;

protected:
    void createRequestDescriptor(HardwareRequest& hardwareRequest) const;

    Gna2Status parseHwStatus(uint32_t hwStatus) const override;

private:
    WindowsDriverInterface(const WindowsDriverInterface &) = delete;
    WindowsDriverInterface& operator=(const WindowsDriverInterface&) = delete;

    inline static std::string lastErrorToString(DWORD error);

    void wait(LPOVERLAPPED ioctl) const;

    void verify(LPOVERLAPPED ioctl) const;

    void checkStatus(BOOL ioResult) const;

    template <typename Predicate>
    void getOverlappedResult(Predicate predicate,
        LPOVERLAPPED ioctl,
        DWORD timeout,
        Gna2Status status,
        char const * message) const
    {
        auto bytesRead = DWORD{ 0 };
        auto const ioResult = GetOverlappedResultEx(deviceHandle,
            ioctl,
            &bytesRead,
            timeout,
            false);
        throwOnFailedPredicate(predicate,
            ioResult,
            status,
            message);
    }

    template <typename Predicate>
    void throwOnFailedPredicate(Predicate predicate,
        BOOL ioResult,
        Gna2Status status,
        char const * message) const
    {
        auto const error = GetLastError();
        if (predicate(ioResult, error))
        {
            if (message)
            {
                Log->Error(message);
#if DEBUG == 1
                auto const errorDescription = lastErrorToString(error);
                Log->Error("%s\n", errorDescription.c_str());
#endif
            }
            throw GnaException(status);
        }
        // io completed successfully
    }

    void getDeviceCapabilities();

    static uint64_t getPerfCounterFrequency();

    static std::string discoverDevice(uint32_t deviceIndex);

    static const std::map<GnaIoctlCommand, DWORD> ioctlCommandsMap;

    std::map<uint64_t, std::unique_ptr<OverlappedWithEvent>> memoryMapRequests;

    WinHandle deviceHandle;
    WinHandle deviceEvent;
    OVERLAPPED overlapped;

    UINT32 recoveryTimeout;
};

}

#endif // WIN32