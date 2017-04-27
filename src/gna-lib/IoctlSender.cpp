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

#include "IoctlSender.h"

#include "GnaException.h"
#include "Logger.h"

using namespace GNA;

using std::unique_ptr;

#define MAX_D0_STATE_PROBES 10
#define WAIT_PERIOD			200		// in miliseconds

WinHandle IoctlSender::deviceHandle;

IoctlSender::IoctlSender() :
    deviceEvent{CreateEvent(nullptr, false, false, nullptr)}
{
    ZeroMemory(&overlapped, sizeof(overlapped));
}

void IoctlSender::Open(const GUID& guid)
{
    auto deviceInfo = SetupDiGetClassDevs(&guid, nullptr, nullptr, DIGCF_PRESENT | DIGCF_DEVICEINTERFACE);
    Expect::False(INVALID_HANDLE_VALUE == deviceInfo, GNA_DEVNOTFOUND);

    auto deviceDetailsData = unique_ptr<char[]>();
    auto deviceDetails = PSP_DEVICE_INTERFACE_DETAIL_DATA{nullptr};
    auto interfaceData = SP_DEVICE_INTERFACE_DATA{0};
    interfaceData.cbSize = sizeof(interfaceData);

    for (auto i = 0; SetupDiEnumDeviceInterfaces(deviceInfo, nullptr, &guid, i, &interfaceData); ++i)
    {
        auto bufferSize = DWORD{0};
        if (!SetupDiGetDeviceInterfaceDetail(deviceInfo, &interfaceData, nullptr, 0, &bufferSize, nullptr))
        {
            auto err = GetLastError();
            if (ERROR_INSUFFICIENT_BUFFER != err)
                continue; // proceed to the next device
        }
        deviceDetailsData.reset(new char[bufferSize]);
        deviceDetails = reinterpret_cast<PSP_DEVICE_INTERFACE_DETAIL_DATA>(deviceDetailsData.get());
        deviceDetails->cbSize = sizeof(SP_DEVICE_INTERFACE_DETAIL_DATA);
        if (!SetupDiGetDeviceInterfaceDetail(deviceInfo, &interfaceData, deviceDetails, bufferSize, nullptr, nullptr))
        {
            //deviceDetailsData.release(); // TODO verify if data is freed
            continue;
        }
        break;
    }
    SetupDiDestroyDeviceInfoList(deviceInfo);
    Expect::NotNull(deviceDetails, GNA_DEVNOTFOUND);

    deviceHandle.Set(CreateFile(deviceDetails->DevicePath,
        GENERIC_READ | GENERIC_WRITE,
        0,
        nullptr,
        CREATE_ALWAYS,
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED,
        nullptr));
    Expect::False(INVALID_HANDLE_VALUE == deviceHandle, GNA_DEVNOTFOUND);
}

void IoctlSender::IoctlSend(const DWORD code, LPVOID const inbuf, const DWORD inlen, LPVOID const outbuf,
    const DWORD outlen, BOOLEAN async)
{
    overlapped.hEvent = deviceEvent;
    auto bytesRead = DWORD{0};

    auto ioResult = DeviceIoControl(deviceHandle, code, inbuf, inlen, outbuf, outlen, &bytesRead, &overlapped);
    checkStatus(ioResult);

    if (!async)
    {
        wait(&overlapped, (DRV_RECOVERY_TIMEOUT + 15) * 1000);
    }
}

void IoctlSender::Submit(LPVOID const inbuf, const DWORD inlen, RequestProfiler * const profiler)
{
    auto ioHandle = OVERLAPPED{0};
    ioHandle.hEvent = CreateEvent(nullptr, false, false, nullptr);

    profilerDTscStart(&profiler->ioctlSubmit);
    auto ioResult = WriteFile(deviceHandle, inbuf, inlen, nullptr, &ioHandle);
    checkStatus(ioResult);
    profilerDTscStop(&profiler->ioctlSubmit);

    profilerDTscStart(&profiler->ioctlWaitOn);
    wait(&ioHandle, GNA_REQUEST_TIMEOUT_MAX);
    profilerDTscStop(&profiler->ioctlWaitOn);
}

void IoctlSender::wait(LPOVERLAPPED const ioctl, const DWORD timeout)
{
    auto bytesRead = DWORD{0};

    auto ioResult = GetOverlappedResultEx(deviceHandle, ioctl, &bytesRead, timeout, false);
    if (ioResult == 0) // io not completed
    {
        auto error = GetLastError();
        if ((ERROR_IO_INCOMPLETE == error && 0 == timeout )||
             WAIT_IO_COMPLETION  == error ||
             WAIT_TIMEOUT        == error)
        {
            throw GnaException(GNA_DEVICEBUSY); // not completed yet
        }
        else // other wait error
        {
            Log->Error("GetOverlappedResult failed, error: \n");
#if DEBUG == 1
            printLastError(error);
#endif
            throw GnaException(GNA_IOCTLRESERR);
        }
    }
    // io completed successfully
}

void IoctlSender::checkStatus(BOOL ioResult)
{
    auto lastError = GetLastError();
    if (ioResult == 0 && ERROR_IO_PENDING != lastError)
    {
#if DEBUG == 1
            printLastError(lastError);
#endif
        throw GnaException(GNA_IOCTLSENDERR);
    }
}

void IoctlSender::printLastError(DWORD error)
{
    LPVOID lpMsgBuf;
    FormatMessage(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr,
        error,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        (LPTSTR)&lpMsgBuf,
        0,
        nullptr);

    // Display the error message
    if (nullptr != lpMsgBuf)
    {
        wprintf(L"%s\n", (wchar_t*)lpMsgBuf);
        LocalFree(lpMsgBuf);
    }
}