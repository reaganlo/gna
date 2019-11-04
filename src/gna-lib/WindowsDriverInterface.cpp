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

#ifdef WIN32

#include "WindowsDriverInterface.h"

#include "GnaException.h"
#include "HardwareRequest.h"
#include "Logger.h"
#include "Macros.h"
#include "Memory.h"

#if defined(_WIN32)
#if HW_VERBOSE == 1
#include "GnaDrvApiWinDebug.h"
#else
#include "GnaDrvApi.h"
#endif
#else
#error Verbose version of library available only on Windows OS
#endif

#include <SetupApi.h>
#include <ntstatus.h>

using namespace GNA;

#define MAX_D0_STATE_PROBES  10
#define WAIT_PERIOD         200 // in miliseconds

const std::map<GnaIoctlCommand, DWORD> WindowsDriverInterface::ioctlCommandsMap =
{
    { GNA_COMMAND_GET_PARAM, GNA_IOCTL_GET_PARAM },
    { GNA_COMMAND_MAP, GNA_IOCTL_MEM_MAP2 },
    { GNA_COMMAND_UNMAP, GNA_IOCTL_MEM_UNMAP2 },
#if HW_VERBOSE == 1
    { GNA_COMMAND_READ_REG, GNA_IOCTL_READ_REG },
    { GNA_COMMAND_WRITE_REG, GNA_IOCTL_WRITE_REG }
#endif
};

WindowsDriverInterface::WindowsDriverInterface() :
    deviceEvent{ CreateEvent(nullptr, false, false, nullptr) },
    recoveryTimeout{ DRV_RECOVERY_TIMEOUT }
{
    ZeroMemory(&overlapped, sizeof(overlapped));
}

bool WindowsDriverInterface::OpenDevice(uint32_t deviceIndex)
{
    auto devicePath = discoverDevice(deviceIndex);
    if ("" == devicePath)
    {
        return false;
    }

    deviceHandle.Set(CreateFile(devicePath.c_str(),
        GENERIC_READ | GENERIC_WRITE,
        0,
        nullptr,
        CREATE_ALWAYS,
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED,
        nullptr));
    if (INVALID_HANDLE_VALUE == deviceHandle)
    {
        return false;
    }

    getDeviceCapabilities();

    return true;
}

void WindowsDriverInterface::IoctlSend(const GnaIoctlCommand command, void * const inbuf, const uint32_t inlen,
    void * const outbuf, const uint32_t outlen)
{
    UNREFERENCED_PARAMETER(command);
    UNREFERENCED_PARAMETER(inbuf);
    UNREFERENCED_PARAMETER(inlen);
    UNREFERENCED_PARAMETER(outbuf);
    UNREFERENCED_PARAMETER(outlen);

#if HW_VERBOSE == 1
    auto bytesRead = DWORD{ 0 };
    auto ioResult = BOOL{};

    overlapped.hEvent = deviceEvent;

    uint32_t code;
    switch (command)
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
        throw GnaException{ Gna2StatusDeviceOutgoingCommunicationError };
    }
#endif
}

uint64_t WindowsDriverInterface::MemoryMap(void *memory, uint32_t memorySize)
{
    auto bytesRead = DWORD{ 0 };
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

void WindowsDriverInterface::MemoryUnmap(uint64_t memoryId)
{
    auto bytesRead = DWORD{ 0 };
    auto ioResult = BOOL{};

    // TODO: remove ummap IOCTL in favor of CancelIoEx (cancel handler in driver, cancel requests)
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
    RequestProfiler * const profiler) const
{
    auto bytesRead = DWORD{ 0 };
    RequestResult result = { 0 };
    auto ioHandle = OVERLAPPED{ 0 };
    ioHandle.hEvent = CreateEvent(nullptr, false, false, nullptr);

    // TODO:kj:3: add working optimization mechanism to reduce recalculation for same request config
    createRequestDescriptor(hardwareRequest);

    auto input = reinterpret_cast<PGNA_INFERENCE_CONFIG_IN>(hardwareRequest.CalculationData.get());
    auto * const ctrlFlags = &input->ctrlFlags;
    ctrlFlags->ddiVersion = 2;
    ctrlFlags->activeListOn = hardwareRequest.ActiveListOn;
    ctrlFlags->gnaMode = hardwareRequest.Mode;
    ctrlFlags->layerCount = hardwareRequest.LayerCount;

    if (xNN == hardwareRequest.Mode)
    {
        input->configBase = hardwareRequest.LayerBase;
    }
    else if (GMM == hardwareRequest.Mode)
    {
        input->configBase = hardwareRequest.GmmOffset;
    }
    else
    {
        throw GnaException{ Gna2StatusXnnErrorLyrCfg };
    }

    profiler->Measure(Gna2InstrumentationPointLibDeviceRequestReady);

    auto ioResult = WriteFile(static_cast<HANDLE>(deviceHandle),
        input, static_cast<DWORD>(hardwareRequest.CalculationSize),
        nullptr, &ioHandle);
    checkStatus(ioResult);

    profiler->Measure(Gna2InstrumentationPointLibDeviceRequestSent);

    GetOverlappedResultEx(deviceHandle, &ioHandle, &bytesRead, GNA_REQUEST_TIMEOUT_MAX, false);

    profiler->Measure(Gna2InstrumentationPointLibDeviceRequestCompleted);

    auto const output = reinterpret_cast<PGNA_INFERENCE_CONFIG_OUT>(input);
    auto const status = output->status;
    auto const writeStatus = (NTSTATUS)ioHandle.Internal;
    switch (writeStatus)
    {
    case STATUS_SUCCESS:
        memcpy_s(&result.hardwarePerf, sizeof(result.hardwarePerf),
            &output->hardwareInstrumentation, sizeof(GNA_PERF_HW));
        memcpy_s(&result.driverPerf, sizeof(result.driverPerf),
            &output->driverInstrumentation, sizeof(GNA_DRIVER_INSTRUMENTATION));
        result.status = (status & STS_SATURATION_FLAG)
            ? Gna2StatusWarningArithmeticSaturation : Gna2StatusSuccess;
        break;
    case STATUS_IO_DEVICE_ERROR:
        result.status = parseHwStatus(status);
        break;
    case STATUS_MORE_PROCESSING_REQUIRED:
        result.status = Gna2StatusWarningDeviceBusy;
        break;
    case STATUS_IO_TIMEOUT:
        result.status = Gna2StatusDeviceCriticalFailure;
        break;
    default:
        result.status = Gna2StatusDeviceIngoingCommunicationError;
        break;
    }

    return result;
}

void WindowsDriverInterface::createRequestDescriptor(HardwareRequest& hardwareRequest) const
{
    auto& totalConfigSize = hardwareRequest.CalculationSize;
    auto const bufferCount = hardwareRequest.DriverMemoryObjects.size();
    totalConfigSize = sizeof(GNA_INFERENCE_CONFIG_IN) + bufferCount * sizeof(GNA_MEMORY_BUFFER);

    for (const auto &buffer : hardwareRequest.DriverMemoryObjects)
    {
        totalConfigSize += buffer.Patches.size() * sizeof(GNA_MEMORY_PATCH);

        for (const auto &patch : buffer.Patches)
        {
            totalConfigSize += patch.Size;
        }
    }

    totalConfigSize = (((totalConfigSize) > (sizeof(GNA_INFERENCE_CONFIG))) ?
        (totalConfigSize) : (sizeof(GNA_INFERENCE_CONFIG)));
    totalConfigSize = RoundUp(totalConfigSize, sizeof(uint64_t));


    hardwareRequest.CalculationData.reset(new uint8_t[totalConfigSize]);

    auto * const input = reinterpret_cast<GNA_INFERENCE_CONFIG_IN *>(hardwareRequest.CalculationData.get());
    memset(input, 0, totalConfigSize);
    input->ctrlFlags.hwPerfEncoding = hardwareRequest.HwPerfEncoding;
    input->bufferCount = bufferCount;

    auto buffer = reinterpret_cast<GNA_MEMORY_BUFFER *>(input + 1);
    auto patch = reinterpret_cast<GNA_MEMORY_PATCH *>(buffer + bufferCount);

    for (const auto &driverBuffer : hardwareRequest.DriverMemoryObjects)
    {
        buffer->memoryId = driverBuffer.Buffer.GetId();
        buffer->offset = 0;
        buffer->size = driverBuffer.Buffer.GetSize();
        buffer->patchCount = driverBuffer.Patches.size();

        for (const auto &driverPatch : driverBuffer.Patches)
        {
            patch->offset = driverPatch.Offset;
            patch->size = driverPatch.Size;
            memcpy_s(patch->data, driverPatch.Size, &driverPatch.Value, driverPatch.Size);
            patch = reinterpret_cast<GNA_MEMORY_PATCH *>(
                reinterpret_cast<uint8_t*>(patch) + sizeof(GNA_MEMORY_PATCH) + patch->size);
        }

        buffer++;
    }

    hardwareRequest.SubmitReady = true;
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

    driverCapabilities.deviceVersion = static_cast<DeviceVersion>(values[0]);
    driverCapabilities.hwInBuffSize = static_cast<uint32_t>(values[1]);
    driverCapabilities.recoveryTimeout = static_cast<uint32_t>(values[2]);
    recoveryTimeout = driverCapabilities.recoveryTimeout;
}

std::string WindowsDriverInterface::discoverDevice(uint32_t deviceIndex)
{
    auto guid = GUID_DEVINTERFACE_GNA_DRV;
    const auto deviceInfo = SetupDiGetClassDevs(&guid, nullptr, nullptr, DIGCF_PRESENT | DIGCF_DEVICEINTERFACE);
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
    for (auto i = 0; SetupDiEnumDeviceInterfaces(deviceInfo, nullptr, &guid, i, &interfaceData); ++i)
    {
        bufferSize = DWORD{ 0 };
        path = "";
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
            continue;
        }
        if (found++ == deviceIndex)
        {
            path = std::string(deviceDetails->DevicePath, bufferSize - sizeof(deviceDetails->cbSize));
            break;
        }
    }
    SetupDiDestroyDeviceInfoList(deviceInfo);
    return path;
}

void WindowsDriverInterface::wait(LPOVERLAPPED const ioctl, const DWORD timeout) const
{
    auto bytesRead = DWORD{ 0 };

    auto ioResult = GetOverlappedResultEx(deviceHandle, ioctl, &bytesRead, timeout, false);
    if (ioResult == 0) // io not completed
    {
        auto error = GetLastError();
        if ((ERROR_IO_INCOMPLETE == error && 0 == timeout) ||
            WAIT_IO_COMPLETION == error ||
            WAIT_TIMEOUT == error)
        {
            throw GnaException(Gna2StatusWarningDeviceBusy); // not completed yet
        }
        else // other wait error
        {
            Log->Error("GetOverlappedResult failed, error: \n");
#if DEBUG == 1
            printLastError(error);
#endif
            throw GnaException(Gna2StatusDeviceIngoingCommunicationError);
        }
    }
    // io completed successfully
}

void WindowsDriverInterface::checkStatus(BOOL ioResult) const
{
    auto lastError = GetLastError();
    if (ioResult == 0 && ERROR_IO_PENDING != lastError)
    {
#if DEBUG == 1
        printLastError(lastError);
#endif
        throw GnaException(Gna2StatusDeviceOutgoingCommunicationError);
    }
}

void WindowsDriverInterface::printLastError(DWORD error) const
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
        printf("%s\n", static_cast<LPTSTR>(lpMsgBuf));
        LocalFree(lpMsgBuf);
    }
}

Gna2Status WindowsDriverInterface::parseHwStatus(uint32_t hwStatus) const
{
    if (hwStatus & STS_MMUREQERR_FLAG)
    {
        return Gna2StatusDeviceMmuRequestError;
    }
    if (hwStatus & STS_DMAREQERR_FLAG)
    {
        return Gna2StatusDeviceDmaRequestError;
    }
    if (hwStatus & STS_UNEXPCOMPL_FLAG)
    {
        return Gna2StatusDeviceUnexpectedCompletion;
    }
    if (hwStatus & STS_VA_OOR_FLAG)
    {
        return Gna2StatusDeviceVaOutOfRange;
    }
    if (hwStatus & STS_PARAM_OOR_FLAG)
    {
        return Gna2StatusDeviceParameterOutOfRange;
    }

    return Gna2StatusDeviceCriticalFailure;
}

#endif // WIN32
