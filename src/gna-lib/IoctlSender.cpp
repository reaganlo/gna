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
#include "Validator.h"

#include <SetupApi.h>

using namespace GNA;

#define MAX_D0_STATE_PROBES 10
#define WAIT_PERIOD			200		// in miliseconds

WinHandle IoctlSender::h_;

IoctlSender::IoctlSender()
:evt_(CreateEvent(nullptr, false, false, nullptr))
{
    ZeroMemory(&overlapped_, sizeof(overlapped_));
}

void IoctlSender::Open(const GUID& guid)
{
    HDEVINFO hDeviceInfo = SetupDiGetClassDevs(&guid,
        nullptr, nullptr, DIGCF_PRESENT | DIGCF_DEVICEINTERFACE);
    if (INVALID_HANDLE_VALUE == hDeviceInfo)
        throw GnaException(GNA_DEVNOTFOUND);

    PSP_DEVICE_INTERFACE_DETAIL_DATA deviceDetail = nullptr;

    SP_DEVICE_INTERFACE_DATA interfaceData;
    interfaceData.cbSize = sizeof(interfaceData);
    for (LONG i = 0;
         SetupDiEnumDeviceInterfaces(hDeviceInfo, nullptr, &guid, i, &interfaceData);
         ++i)
    {
        DWORD bufferSize;
        if (!SetupDiGetDeviceInterfaceDetail(hDeviceInfo, &interfaceData, nullptr, 0, &bufferSize, nullptr))
        {
            DWORD err = GetLastError();
            if (ERROR_INSUFFICIENT_BUFFER != err)
                continue; // proceed to the next device
        }
        deviceDetail = (PSP_DEVICE_INTERFACE_DETAIL_DATA) new unsigned char[bufferSize];
        deviceDetail->cbSize = sizeof(SP_DEVICE_INTERFACE_DETAIL_DATA);
        if (!SetupDiGetDeviceInterfaceDetail(hDeviceInfo,
            &interfaceData, deviceDetail, bufferSize, nullptr, nullptr))
        {
            delete [] deviceDetail;
            deviceDetail = nullptr;
            continue;
        }
        break;
    }
    SetupDiDestroyDeviceInfoList(hDeviceInfo);
    if (nullptr == deviceDetail)
        throw GnaException(GNA_DEVNOTFOUND);

    h_.Set(CreateFile(deviceDetail->DevicePath,
        GENERIC_READ | GENERIC_WRITE,
        0,
        nullptr,
        CREATE_ALWAYS,
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED,
        nullptr));

    if (INVALID_HANDLE_VALUE == h_)
        throw GnaException(GNA_DEVNOTFOUND);
}

status_t IoctlSender::IoctlSend(
    DWORD code,
    LPVOID inbuf,
    DWORD inlen,
    LPVOID outbuf,
    DWORD outlen,
    BOOLEAN async)
{
    overlapped_.hEvent = evt_;
    DWORD bytesRead = 0;
    BOOL devOn = false;
    DWORD waitResult = WAIT_OBJECT_0;
    BOOL ioResult = true;

    ioResult = DeviceIoControl(h_, code, inbuf, inlen, outbuf, outlen, &bytesRead, &overlapped_);
    DWORD lastError = GetLastError();

    Expect::False(ioResult == 0 && ERROR_IO_PENDING != lastError, GNA_IOCTLSENDERR);

    if (ERROR_IO_PENDING == lastError && async)
        return GNA_SUCCESS;

    return IoctlWait(&overlapped_, (DRV_RECOVERY_TIMEOUT + 15) * 1000);
}

status_t IoctlSender::Submit(
    LPVOID      inbuf,
    DWORD       inlen,
    prof_tsc_t* ioctlSubmit,
    io_handle_t*  handle)
{
    BOOL ioResult = true;

    profilerDTscStart(ioctlSubmit);
    ioResult = WriteFile(h_, inbuf, inlen, nullptr, handle);
    profilerDTscStop(ioctlSubmit);

    if (ioResult == 0 && ERROR_IO_PENDING != GetLastError())
    {
#if DEBUG == 1
            printLastError(GetLastError());
#endif
        return GNA_IOCTLSENDERR;
    }

    return GNA_SUCCESS;
}

status_t IoctlSender::IoctlWait(
    LPOVERLAPPED ioctl,
    DWORD        timeout)
{
    BOOL  ioResult  = true;
    DWORD bytesRead = 0;
    DWORD error     = WAIT_OBJECT_0;

    ioResult = GetOverlappedResultEx(h_, ioctl, &bytesRead, timeout, false);
    if (ioResult == 0) // io not completed
    {
        error = GetLastError();
        if ((ERROR_IO_INCOMPLETE == error && 0 == timeout )||
             WAIT_IO_COMPLETION  == error ||
             WAIT_TIMEOUT        == error)
        {
            return GNA_DEVICEBUSY; // not completed yet
        }
        else // other wait error
        {
            ERR("GetOverlappedResult failed, error:\n");
#if DEBUG == 1
            printLastError(error);
#endif
            return GNA_IOCTLRESERR;
        }
    }
    return GNA_SUCCESS; // io completed successfully
}

DWORD IoctlSender::Cancel()
{
    if (CancelIoEx(h_, &overlapped_))
    {
        ZeroMemory(&overlapped_, sizeof(overlapped_));
        return true;
    }
    return false;
}