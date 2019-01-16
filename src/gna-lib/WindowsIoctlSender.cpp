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
#include "WindowsIoctlSender.h"

#include "GnaException.h"
#include "Logger.h"
#include "HardwareRequest.h"

using namespace GNA;

using std::unique_ptr;

#define MAX_D0_STATE_PROBES  10
#define WAIT_PERIOD         200        // in miliseconds

const std::map<GnaIoctlCommand, decltype(GNA_IOCTL_CPBLTS2)> WindowsIoctlSender::ioctlCommandsMap =
{
    { GNA_COMMAND_CAPABILITIES, GNA_IOCTL_CPBLTS2 },
    { GNA_COMMAND_MAP, GNA_IOCTL_MEM_MAP2 },
    { GNA_COMMAND_UNMAP, GNA_IOCTL_MEM_UNMAP2 },
#if HW_VERBOSE == 1
    { GNA_COMMAND_READ_REG, GNA_IOCTL_READ_REG },
    { GNA_COMMAND_WRITE_REG, GNA_IOCTL_WRITE_REG }
#endif
};

WindowsIoctlSender::WindowsIoctlSender() :
    deviceEvent{CreateEvent(nullptr, false, false, nullptr)}
{
    ZeroMemory(&overlapped, sizeof(overlapped));
}

void WindowsIoctlSender::Open()
{
    auto guid = GUID_DEVINTERFACE_GNA_DRV;
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
            //deviceDetailsData.release(); // TODO: verify if data is freed
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

    getDeviceCapabilities();
}

void WindowsIoctlSender::IoctlSend(const GnaIoctlCommand command, void * const inbuf, const uint32_t inlen,
    void * const outbuf, const uint32_t outlen)
{
    UNREFERENCED_PARAMETER(command);
    UNREFERENCED_PARAMETER(inbuf);
    UNREFERENCED_PARAMETER(inlen);
    UNREFERENCED_PARAMETER(outbuf);
    UNREFERENCED_PARAMETER(outlen);

#if HW_VERBOSE == 1
    auto bytesRead = DWORD{0};
    auto ioResult = BOOL{};

    overlapped.hEvent = deviceEvent;

    uint32_t code;
    switch(command)
    {
        case GNA_COMMAND_READ_REG:
            /* FALLTHRU */
        case GNA_COMMAND_WRITE_REG:
            code = ioctlCommandsMap.at(command);
            ioResult = DeviceIoControl(deviceHandle, code, inbuf, inlen, outbuf, outlen, &bytesRead, &overlapped);
            checkStatus(ioResult);
            wait(&overlapped, (recoveryTimeout + 15) * 1000);
            break;
        default:
            throw GnaException { GNA_IOCTLSENDERR };
    }
#endif
}

uint64_t WindowsIoctlSender::MemoryMap(void *memory, size_t memorySize)
{
    auto bytesRead = DWORD{0};
    auto ioResult = BOOL{};
    uint64_t memoryId;

    auto memoryMapOverlapped = std::make_unique<OVERLAPPED>();
    memoryMapOverlapped->hEvent = CreateEvent(nullptr, false, false, nullptr);

    ioResult = DeviceIoControl(deviceHandle, static_cast<DWORD>(GNA_IOCTL_NOTIFY),
        nullptr, static_cast<DWORD>(0),
        &memoryId, sizeof(memoryId), &bytesRead, &overlapped);
    checkStatus(ioResult);

    ioResult = DeviceIoControl(deviceHandle, static_cast<DWORD>(GNA_IOCTL_MEM_MAP2),
        nullptr, static_cast<DWORD>(0), memory,
        static_cast<DWORD>(memorySize), &bytesRead, memoryMapOverlapped.get());
    checkStatus(ioResult);

    wait(&overlapped, (recoveryTimeout + 15) * 1000);

    memoryMapRequests[memoryId] = std::move(memoryMapOverlapped);

    return memoryId;
}

void WindowsIoctlSender::MemoryUnmap(uint64_t memoryId)
{
    auto bytesRead = DWORD{0};
    auto ioResult = BOOL{};

    // TODO: remove ummap IOCTL in favor of CancelIoEx (cancel handler in driver, cancel requests)
    ioResult = DeviceIoControl(deviceHandle, static_cast<DWORD>(GNA_IOCTL_MEM_UNMAP2), &memoryId, sizeof(memoryId), nullptr, 0, &bytesRead, &overlapped);
    checkStatus(ioResult);
    wait(&overlapped, (recoveryTimeout + 15) * 1000);

    auto memoryMapOverlapped = memoryMapRequests.at(memoryId).get();
    wait(memoryMapOverlapped, (recoveryTimeout + 15) * 1000);
    memoryMapRequests.erase(memoryId);
}

RequestResult WindowsIoctlSender::Submit(HardwareRequest *hardwareRequest, RequestProfiler * const profiler)
{
    RequestResult result = { 0 };
    auto ioHandle = OVERLAPPED{0};
    ioHandle.hEvent = CreateEvent(nullptr, false, false, nullptr);

    if(!hardwareRequest->SubmitReady)
    {
        createRequestDescriptor(hardwareRequest);
    }

    auto calculationData = reinterpret_cast<PGNA_CALC_IN>(hardwareRequest->CalculationData.get());

    calculationData->ctrlFlags.ddiVersion = 2;
    calculationData->ctrlFlags.activeListOn = hardwareRequest->ActiveListOn;
    calculationData->ctrlFlags.gnaMode = hardwareRequest->Mode;
    calculationData->ctrlFlags.layerCount = hardwareRequest->LayerCount;

    if(xNN == hardwareRequest->Mode)
    {
        calculationData->configBase = hardwareRequest->LayerBase;
    }
    else if(GMM == hardwareRequest->Mode)
    {
        calculationData->configBase = hardwareRequest->GmmOffset;
    }
    else
    {
        throw GnaException { XNN_ERR_LYR_CFG };
    }

    profilerTscStart(&profiler->ioctlSubmit);
    auto ioResult = WriteFile(deviceHandle, calculationData, static_cast<DWORD>(hardwareRequest->CalculationSize), nullptr, &ioHandle);
    checkStatus(ioResult);
    profilerTscStop(&profiler->ioctlSubmit);

    profilerTscStart(&profiler->ioctlWaitOn);
    wait(&ioHandle, GNA_REQUEST_TIMEOUT_MAX);
    profilerTscStop(&profiler->ioctlWaitOn);

    result.hardwarePerf = calculationData->hwPerf;
    result.driverPerf = calculationData->drvPerf;
    result.status = calculationData->status;

    return result;
}

GnaCapabilities WindowsIoctlSender::GetDeviceCapabilities() const
{
    return deviceCapabilities;
}

void WindowsIoctlSender::getDeviceCapabilities()
{
    auto bytesRead = DWORD{0};
    GNA_CPBLTS cpblts;

    auto ioResult = DeviceIoControl(deviceHandle, static_cast<DWORD>(GNA_IOCTL_CPBLTS2), nullptr, 0,
            &cpblts, sizeof(cpblts), &bytesRead, &overlapped);

    checkStatus(ioResult);
    wait(&overlapped, (recoveryTimeout + 15) * 1000);

    deviceCapabilities.hwInBuffSize = cpblts.hwInBuffSize;
    deviceCapabilities.recoveryTimeout = cpblts.recoveryTimeout;
    deviceCapabilities.hwId = static_cast<gna_device_version>(cpblts.deviceType);
}

void WindowsIoctlSender::wait(LPOVERLAPPED const ioctl, const DWORD timeout)
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

void WindowsIoctlSender::checkStatus(BOOL ioResult)
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

void WindowsIoctlSender::printLastError(DWORD error)
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
