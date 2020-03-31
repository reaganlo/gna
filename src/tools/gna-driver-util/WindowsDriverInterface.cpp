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

//TODO extract common part from library and util into separate file to omit redundancy.
#ifdef WIN32
#include "WindowsDriverInterface.h"

#include <iostream>
#include <ntstatus.h>
#include <SetupApi.h>
#pragma comment (lib, "Setupapi.lib")

bool WindowsDriverInterface::discoverDevice(uint32_t deviceIndex)
{
    auto guid = GUID_DEVINTERFACE_GNA_DRV;
    const auto deviceInfo = SetupDiGetClassDevs(&guid, nullptr, nullptr,
        DIGCF_PRESENT | DIGCF_DEVICEINTERFACE);
    if (INVALID_HANDLE_VALUE == deviceInfo)
    {
        return "";
    }

    auto deviceDetailsData = std::unique_ptr<char[]>();
    auto deviceDetails = PSP_DEVICE_INTERFACE_DETAIL_DATA{ nullptr };
    auto interfaceData = SP_DEVICE_INTERFACE_DATA{ 0 };
    interfaceData.cbSize = sizeof(interfaceData);

    uint32_t found = 0;
    auto bufferSize = DWORD{ 0 };
    std::string path;
    for (auto i = 0; SetupDiEnumDeviceInterfaces(deviceInfo, nullptr, &guid, i,
        &interfaceData); ++i)
    {
        bufferSize = DWORD{ 0 };
        path = "";
        if (!SetupDiGetDeviceInterfaceDetail(deviceInfo, &interfaceData,
            nullptr, 0, &bufferSize, nullptr))
        {
            auto err = GetLastError();
            if (ERROR_INSUFFICIENT_BUFFER != err)
                continue; // proceed to the next device
        }
        deviceDetailsData.reset(new char[bufferSize]);
        deviceDetails = reinterpret_cast<PSP_DEVICE_INTERFACE_DETAIL_DATA>
            (deviceDetailsData.get());
        deviceDetails->cbSize = sizeof(SP_DEVICE_INTERFACE_DETAIL_DATA);
        if (!SetupDiGetDeviceInterfaceDetail(deviceInfo, &interfaceData,
            deviceDetails, bufferSize, nullptr, nullptr))
        {
            continue;
        }
        if (found++ == deviceIndex)
        {
            path = std::string(deviceDetails->DevicePath,
                bufferSize - sizeof(deviceDetails->cbSize));
            break;
        }
    }
    SetupDiDestroyDeviceInfoList(deviceInfo);
    return OpenDevice(path);
}

bool WindowsDriverInterface::OpenDevice(std::string devicePath)
{
    deviceHandle = CreateFile(devicePath.c_str(),
        GENERIC_READ | GENERIC_WRITE,
        0,
        nullptr,
        CREATE_ALWAYS,
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED,
        nullptr);
    if (INVALID_HANDLE_VALUE == deviceHandle)
    {
        return false;
    }

    getDeviceCapabilities();

    std::cout << "Hardware buffer in size: " << driverCapabilities.hwInBuffSize
        << "\nDevice version: " << driverCapabilities.deviceVersion
        << "\nRecovery timeout: " << driverCapabilities.recoveryTimeout
        << std::endl;

    return true;
}

void WindowsDriverInterface::checkStatus(BOOL ioResult) const
{
    auto lastError = GetLastError();
    if (ioResult == 0 && ERROR_IO_PENDING != lastError)
    {
        throw std::exception("Error: Failed to sent communication to the device driver.");
    }
}

void WindowsDriverInterface::wait(LPOVERLAPPED const ioctl,
    const DWORD timeout) const
{
    auto bytesRead = DWORD{ 0 };

    auto ioResult = GetOverlappedResultEx(deviceHandle, ioctl, &bytesRead,
        timeout, false);
    if (ioResult == 0) // io not completed
    {
        auto error = GetLastError();
        if ((ERROR_IO_INCOMPLETE == error && 0 == timeout) ||
            WAIT_IO_COMPLETION == error ||
            WAIT_TIMEOUT == error)
        {
            throw std::exception(" Warning: Device is busy.\nGNA is still running, can not enqueue more requests.");
        }
        else // other wait error
        {
            throw std::exception("Error: Failed to receive communication from the device driver.");
        }
    }
    // io completed successfully
}

void WindowsDriverInterface::getDeviceCapabilities()
{
    auto bytesRead = DWORD{ 0 };
    UINT64 params[3] = {
        GNA_PARAM_DEVICE_TYPE,
        GNA_PARAM_INPUT_BUFFER_S,
        GNA_PARAM_RECOVERY_TIMEOUT
    };

    UINT64 values[3];

    for (uint32_t i = 0; i < sizeof(params) / sizeof(UINT64); i++)
    {
        auto ioResult = DeviceIoControl(deviceHandle,
            static_cast<DWORD>(GNA_IOCTL_GET_PARAM),
            &params[i], sizeof(params[i]),
            &values[i], sizeof(values[i]),
            &bytesRead, &overlapped);
        checkStatus(ioResult);
        wait(&overlapped, (recoveryTimeout + 15) * 1000);
    }

    driverCapabilities.deviceVersion = static_cast<gna_device_version>
        (values[0]);
    driverCapabilities.hwInBuffSize = static_cast<uint32_t>(values[1]);
    driverCapabilities.recoveryTimeout = static_cast<uint32_t>(values[2]);
    recoveryTimeout = driverCapabilities.recoveryTimeout;
}

WindowsDriverInterface::WindowsDriverInterface()
{
    ZeroMemory(&overlapped, sizeof(overlapped));
}

uint64_t WindowsDriverInterface::MemoryMap(void *memory, uint32_t memorySize)
{
    auto bytesRead = DWORD{ 0 };

    auto memoryMapOverlapped = std::make_unique<OVERLAPPED>();
    memoryMapOverlapped->hEvent = CreateEvent(nullptr, false, false, nullptr);

    // Memory id is reported form Windows driver at the beginning of mapped memory
    // so we copy it before modifications to restore afterwards
    volatile uint64_t& outMemoryId = *static_cast<uint64_t*>(memory);
    const auto bufferCopy = outMemoryId;
    outMemoryId = FORBIDDEN_MEMORY_ID;
    try
    {
        const auto ioResult = DeviceIoControl(deviceHandle, static_cast<DWORD>(GNA_IOCTL_MEM_MAP2),
            nullptr, static_cast<DWORD>(0), memory,
            static_cast<DWORD>(memorySize), &bytesRead, memoryMapOverlapped.get());
        checkStatus(ioResult);
        int totalWaitForMapMilliseconds = 0;
        for (int i = 0; outMemoryId == FORBIDDEN_MEMORY_ID && i < WAIT_FOR_MAP_ITERATIONS; i++)
        {
            Sleep(WAIT_FOR_MAP_MILLISECONDS);
            totalWaitForMapMilliseconds += WAIT_FOR_MAP_MILLISECONDS;
        }
    }
    catch (...)
    {
        outMemoryId = bufferCopy;
        throw;
    }
    const auto memoryId = outMemoryId;
    outMemoryId = bufferCopy;
    memoryMapRequests[memoryId] = std::move(memoryMapOverlapped);
    return memoryId;
}


void WindowsDriverInterface::MemoryUnmap(uint64_t memoryId)
{
    auto bytesRead = DWORD{ 0 };
    auto ioResult = BOOL{};

    ioResult = DeviceIoControl(static_cast<HANDLE>(deviceHandle),
        static_cast<DWORD>(GNA_IOCTL_MEM_UNMAP2), &memoryId, sizeof(memoryId),
        nullptr, 0, &bytesRead, &overlapped);
    checkStatus(ioResult);
    wait(&overlapped, (recoveryTimeout + 15) * 1000);

    auto memoryMapOverlapped = memoryMapRequests.at(memoryId).get();
    wait(memoryMapOverlapped, (recoveryTimeout + 15) * 1000);
    memoryMapRequests.erase(memoryId);
}


RequestResult WindowsDriverInterface::Submit(HardwareRequest& hardwareRequest,
    const GnaUtilConfig& file) const
{
    UNREFERENCED_PARAMETER(hardwareRequest);
    auto bytesRead = DWORD{ 0 };
    RequestResult result = { 0 };
    auto ioHandle = OVERLAPPED{ 0 };
    ioHandle.hEvent = CreateEvent(nullptr, false, false, nullptr);

    auto ioResult = WriteFile(static_cast<HANDLE>(deviceHandle),
        file.modelConfig.inference, (DWORD)file.inferenceConfigSize,
        nullptr, &ioHandle);
    checkStatus(ioResult);

    GetOverlappedResultEx(deviceHandle, &ioHandle, &bytesRead,
        (recoveryTimeout + 15) * 1000, false);

    auto const output = reinterpret_cast<PGNA_INFERENCE_CONFIG_OUT>
        (file.modelConfig.inference);
    auto const status = output->status;
    auto const writeStatus = (NTSTATUS)ioHandle.Internal;
    switch (writeStatus)
    {
    case STATUS_SUCCESS:
        result.status = GNA_SUCCESS;
        break;
    case STATUS_IO_DEVICE_ERROR:
        result.status = parseHwStatus(status);
        break;
    case STATUS_MORE_PROCESSING_REQUIRED:
        result.status = GNA_DEVICEBUSY;
        break;
    case STATUS_IO_TIMEOUT:
        result.status = GNA_ERR_DEV_FAILURE;
        break;
    default:
        result.status = GNA_IOCTLRESERR;
        break;
    }

    return result;
}

gna_status_t WindowsDriverInterface::parseHwStatus(uint32_t hwStatus) const
{
    if (hwStatus & STS_MMUREQERR_FLAG)
    {
        return GNA_MMUREQERR;
    }
    if (hwStatus & STS_DMAREQERR_FLAG)
    {
        return GNA_DMAREQERR;
    }
    if (hwStatus & STS_UNEXPCOMPL_FLAG)
    {
        return GNA_UNEXPCOMPL;
    }
    if (hwStatus & STS_VA_OOR_FLAG)
    {
        return GNA_VAOUTOFRANGE;
    }
    if (hwStatus & STS_PARAM_OOR_FLAG)
    {
        return GNA_PARAMETEROUTOFRANGE;
    }

    return GNA_ERR_DEV_FAILURE;
}
#endif
