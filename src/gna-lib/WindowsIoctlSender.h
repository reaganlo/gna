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

#include "IoctlSender.h"

#include <map>
#include <memory>
#include <Windows.h>
#include <SetupApi.h>

#include "GnaDrvApiWinDebug.h"
#include "common.h"
#include "Request.h"
#include "Validator.h"

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
        Expect::True(INVALID_HANDLE_VALUE == deviceHandle, GNA_UNKNOWN_ERROR);
        deviceHandle = handle;
    }

    operator HANDLE() {
        return deviceHandle;
    }

private:
    HANDLE deviceHandle;
};

class WindowsIoctlSender : public IoctlSender
{
public:
    WindowsIoctlSender();

    virtual void Open() override;

    virtual void IoctlSend(const uint32_t code, void * const inbuf, const uint32_t inlen, void * const outbuf, const uint32_t outlen) override;

    virtual void Submit(void * const inbuf, const uint32_t inlen, RequestProfiler * const profiler) override;

private:
    WindowsIoctlSender(const WindowsIoctlSender &) = delete;
    WindowsIoctlSender& operator=(const WindowsIoctlSender&) = delete;

    inline void printLastError(DWORD error);

    void wait(LPOVERLAPPED const ioctl, const DWORD timeout);

    void checkStatus(BOOL ioResult);

    WinHandle deviceHandle;
    WinHandle deviceEvent;
    OVERLAPPED overlapped;
    std::map<uint64_t, std::unique_ptr<OVERLAPPED>> memoryMapRequests;
};

}
