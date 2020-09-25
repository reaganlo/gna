/*
 INTEL CONFIDENTIAL
 Copyright 2020 Intel Corporation.

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

#include <Cfgmgr32.h>
#include <ntstatus.h>

using namespace GNA;

const int WindowsDriverInterface::WAIT_FOR_MAP_ITERATIONS = 2000;
const int WindowsDriverInterface::WAIT_FOR_MAP_MILLISECONDS = 15;
const uint64_t WindowsDriverInterface::FORBIDDEN_MEMORY_ID = 0;

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
    recoveryTimeout{ (DRV_RECOVERY_TIMEOUT + 1) * 1000 }
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
    std::wstring gnaFileName(devicePath.begin(), devicePath.end());

    CREATEFILE2_EXTENDED_PARAMETERS createFileParams;
    createFileParams.dwSize = sizeof(createFileParams);
    createFileParams.dwFileAttributes = FILE_ATTRIBUTE_NORMAL;
    createFileParams.dwFileFlags = FILE_FLAG_OVERLAPPED;
    createFileParams.dwSecurityQosFlags = SECURITY_IMPERSONATION;
    createFileParams.lpSecurityAttributes = nullptr;
    createFileParams.hTemplateFile = nullptr;

    const auto handleForGna = CreateFile2(gnaFileName.data(),
        GENERIC_READ | GENERIC_WRITE,
        0,
        CREATE_ALWAYS,
        &createFileParams);
    deviceHandle.Set(handleForGna);
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
        wait(&overlapped, recoveryTimeout);
        break;
    default:
        throw GnaException{ Gna2StatusDeviceOutgoingCommunicationError };
    }
#endif
}

uint64_t WindowsDriverInterface::MemoryMap(void *memory, uint32_t memorySize)
{
    auto bytesRead = DWORD{ 0 };

    auto memoryMapOverlapped = std::make_unique<OverlappedWithEvent>();

    // Memory id is reported form Windows driver at the beginning of mapped memory
    // so we copy it before modifications to restore afterwards
    volatile uint64_t& outMemoryId = *static_cast<uint64_t*>(memory);
    const auto bufferCopy = outMemoryId;
    outMemoryId = FORBIDDEN_MEMORY_ID;
    try
    {
        const auto ioResult = DeviceIoControl(deviceHandle, static_cast<DWORD>(GNA_IOCTL_MEM_MAP2),
            nullptr, static_cast<DWORD>(0), memory,
            static_cast<DWORD>(memorySize), &bytesRead, *memoryMapOverlapped);
        checkStatus(ioResult);
        int totalWaitForMapMilliseconds = 0;
        for (int i = 0; outMemoryId == FORBIDDEN_MEMORY_ID && i < WAIT_FOR_MAP_ITERATIONS; i++)
        {
            verify(*memoryMapOverlapped);
            Sleep(WAIT_FOR_MAP_MILLISECONDS);
            totalWaitForMapMilliseconds += WAIT_FOR_MAP_MILLISECONDS;
        }
        Log->Message("Waited %i milliseconds for memory mapping\n", totalWaitForMapMilliseconds);
    }
    catch (...)
    {
        outMemoryId = bufferCopy;
        throw;
    }
    const auto memoryId = outMemoryId;
    outMemoryId = bufferCopy;
    Expect::True(memoryId != FORBIDDEN_MEMORY_ID, Gna2StatusDriverCommunicationMemoryMapError);
    Log->Message("Memory mapped with id = %llX\n", memoryId);
    memoryMapRequests[memoryId] = std::move(memoryMapOverlapped);
    return memoryId;
}

void WindowsDriverInterface::MemoryUnmap(uint64_t memoryId)
{
    auto bytesRead = DWORD{ 0 };

    // TODO: remove ummap IOCTL in favor of CancelIoEx (cancel handler in driver, cancel requests)
    const auto ioResult = DeviceIoControl(static_cast<HANDLE>(deviceHandle),
        static_cast<DWORD>(GNA_IOCTL_MEM_UNMAP2), &memoryId, sizeof(memoryId),
        nullptr, 0, &bytesRead, &overlapped);
    checkStatus(ioResult);
    wait(&overlapped);

    const auto& memoryMapOverlapped = memoryMapRequests.at(memoryId);
    wait(*memoryMapOverlapped);
    memoryMapRequests.erase(memoryId);
}
/**
 Send workload to GNA driver for computation.
 The workload is sent in a form of Write Request.
 Write Request is done in a synchronous manner.
 For request description #GNA_INFERENCE_CONFIG_IN is used.

 @param hardwareRequest Description of request.
 @param profiler Request profiling information to be filled.
 */
RequestResult WindowsDriverInterface::Submit(HardwareRequest& hardwareRequest,
    RequestProfiler & profiler) const
{
    auto bytesRead = DWORD{ 0 };
    RequestResult result = { 0 };
    OverlappedWithEvent ioHandle;

    // TODO:kj:3: add working optimization mechanism to reduce recalculation for same request config
    createRequestDescriptor(hardwareRequest);

    auto input = reinterpret_cast<PGNA_INFERENCE_CONFIG_IN>(hardwareRequest.CalculationData.get());
    auto * const ctrlFlags = &input->ctrlFlags;
    ctrlFlags->ddiVersion = 2;
    ctrlFlags->gnaMode = hardwareRequest.Mode;
    ctrlFlags->layerCount = hardwareRequest.LayerCount;

    if (xNN == hardwareRequest.Mode)
    {
        input->configBase = hardwareRequest.LayerBase;
    }
    else if (GMM == hardwareRequest.Mode)
    {
        input->configBase = hardwareRequest.GmmOffset;
        ctrlFlags->activeListOn = hardwareRequest.GmmModeActiveListOn;
    }
    else
    {
        throw GnaException{ Gna2StatusXnnErrorLyrCfg };
    }

    profiler.Measure(Gna2InstrumentationPointLibDeviceRequestReady);

    auto ioResult = WriteFile(static_cast<HANDLE>(deviceHandle),
        input, static_cast<DWORD>(hardwareRequest.CalculationSize),
        nullptr, ioHandle);
    checkStatus(ioResult);

    profiler.Measure(Gna2InstrumentationPointLibDeviceRequestSent);

    GetOverlappedResultEx(deviceHandle, ioHandle, &bytesRead, recoveryTimeout , false);

    profiler.Measure(Gna2InstrumentationPointLibDeviceRequestCompleted);

    auto const output = reinterpret_cast<PGNA_INFERENCE_CONFIG_OUT>(input);
    auto const status = output->status;
    auto const writeStatus = (NTSTATUS)static_cast<OVERLAPPED*>(ioHandle)->Internal;
    ProfilerConfiguration * profilerConfiguration;
    switch (writeStatus)
    {
    case STATUS_SUCCESS:
        memcpy_s(&result.hardwarePerf, sizeof(result.hardwarePerf),
            &output->hardwareInstrumentation, sizeof(GNA_PERF_HW));
        memcpy_s(&result.driverPerf, sizeof(result.driverPerf),
            &output->driverInstrumentation, sizeof(GNA_DRIVER_INSTRUMENTATION));
        profilerConfiguration = hardwareRequest.GetProfilerConfiguration();
        if (profilerConfiguration != nullptr)
        {
            convertPerfResultUnit(result.driverPerf, profilerConfiguration->GetUnit());
        }
        result.status = (status & STS_SATURATION_FLAG)
            ? Gna2StatusWarningArithmeticSaturation : Gna2StatusSuccess;
        break;
    case STATUS_IO_DEVICE_ERROR:
        result.status = parseHwStatus(status);
        break;
    case STATUS_MORE_PROCESSING_REQUIRED:
        result.status = Gna2StatusWarningDeviceBusy;
        break;
    case STATUS_TOO_MANY_SESSIONS:
        result.status = Gna2StatusDriverQoSTimeoutExceeded;
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
        wait(&overlapped);
    }

    driverCapabilities.deviceVersion = static_cast<DeviceVersion>(values[0]);
    driverCapabilities.hwInBuffSize = static_cast<uint32_t>(values[1]);
    driverCapabilities.recoveryTimeout = static_cast<uint32_t>(values[2]);
    recoveryTimeout = (driverCapabilities.recoveryTimeout + 1 ) * 1000;
    driverCapabilities.perfCounterFrequency = getPerfCounterFrequency();
}

uint64_t WindowsDriverInterface::getPerfCounterFrequency()
{
    uint64_t frequency = 0;
    QueryPerformanceFrequency(
        reinterpret_cast<LARGE_INTEGER*>(&frequency));
    return frequency;
}

template<class T>
std::vector<std::vector<T> > splitIntoNonEmpty(std::vector<T> delimitedList, T delimiter)
{
    std::vector<std::vector<T> > split;
    std::vector<T> candidate;
    for (const auto v : delimitedList)
    {
        if (v != delimiter)
        {
            candidate.push_back(v);
        }
        else if(!candidate.empty())
        {
            split.push_back(candidate);
            candidate.clear();
        }
    }
    if (!candidate.empty())
    {
        split.push_back(candidate);
    }
    return split;
}

std::string WindowsDriverInterface::discoverDevice(uint32_t deviceIndex)
{
    auto gnaGuid = GUID_DEVINTERFACE_GNA_DRV;
    CONFIGRET crStatus;
    std::vector<WCHAR> gnaFileName;

    do
    {
        ULONG cmDeviceInterfaceSize = 0;
        crStatus = CM_Get_Device_Interface_List_SizeW(&cmDeviceInterfaceSize,
            &gnaGuid,
            nullptr,
            CM_GET_DEVICE_INTERFACE_LIST_PRESENT);

        if (crStatus != CR_SUCCESS)
        {
            return "";
        }

        gnaFileName.resize(cmDeviceInterfaceSize);

        crStatus = CM_Get_Device_Interface_ListW(&gnaGuid,
            nullptr,
            gnaFileName.data(),
            cmDeviceInterfaceSize,
            CM_GET_DEVICE_INTERFACE_LIST_PRESENT);
    } while (crStatus == CR_BUFFER_SMALL); // handle changes btw CM_[*]_List_Size() and CM_[*]_List() calls

    if (crStatus != CR_SUCCESS)
    {
        return "";
    }
    auto allDevices = splitIntoNonEmpty(gnaFileName, L'\0');

    if(deviceIndex >= allDevices.size())
    {
        return "";
    }
    std::wstring device(allDevices[deviceIndex].begin(), allDevices[deviceIndex].end());
    return std::string(device.begin(), device.end());
}

void WindowsDriverInterface::wait(LPOVERLAPPED ioctl) const
{
    getOverlappedResult(
        [](BOOL ioResult, DWORD error)
            {return (ioResult == 0); },
        ioctl,
        recoveryTimeout,
        Gna2StatusDeviceIngoingCommunicationError,
        "GetOverlappedResult failed.\n");
}

void WindowsDriverInterface::verify(LPOVERLAPPED ioctl) const
{
    getOverlappedResult(
        [](BOOL ioResult, DWORD error)
            {return (ioResult == 0 && STATUS_SUCCESS != error && ERROR_IO_PENDING != error); },
        ioctl,
        0,
        Gna2StatusDriverCommunicationMemoryMapError,
        "MemoryMap failed.\n");
}

void WindowsDriverInterface::checkStatus(BOOL ioResult) const
{
    throwOnFailedPredicate(
        [](BOOL ioResult, DWORD error)
            {return (ioResult == 0 && ERROR_IO_PENDING != error); },
        0,
        Gna2StatusDeviceOutgoingCommunicationError,
        nullptr);
}

std::string WindowsDriverInterface::lastErrorToString(DWORD lastError)
{
    std::string errorDescription = "GetLastError==[" + std::to_string(lastError) + "] ";
    LPVOID lpMsgBuf;
    FormatMessage(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr,
        lastError,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        (LPTSTR)&lpMsgBuf,
        0,
        nullptr);

    if (nullptr != lpMsgBuf)
    {
        errorDescription += static_cast<LPTSTR>(lpMsgBuf);
        LocalFree(lpMsgBuf);
    }
    return errorDescription;
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

OverlappedWithEvent::OverlappedWithEvent()
{
    overlapped.hEvent = CreateEvent(nullptr, false, false, nullptr);
    Expect::NotNull(overlapped.hEvent, Gna2StatusResourceAllocationError);
}

OverlappedWithEvent::~OverlappedWithEvent()
{
    const auto success = CloseHandle(overlapped.hEvent);
    if (!success)
    {
        Log->Error("CloseHandle() failed\n");
    }
}

#endif // WIN32
