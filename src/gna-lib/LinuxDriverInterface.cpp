/*
 INTEL CONFIDENTIAL
 Copyright 2018-2019 Intel Corporation.

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

#ifndef WIN32

#include "LinuxDriverInterface.h"

#include "GnaException.h"
#include "HardwareRequest.h"
#include "Memory.h"
#include "Request.h"

#include "gna-h-wrapper.h"

#include "gna-api.h"
#include "gna-api-status.h"
#include "profiler.h"

#include "gna2-common-impl.h"

#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace GNA;

bool LinuxDriverInterface::OpenDevice(uint32_t deviceIndex)
{
    struct gna_getparam params[] =
    {
        { GNA_PARAM_DEVICE_TYPE, 0 },
        { GNA_PARAM_INPUT_BUFFER_S, 0 },
        { GNA_PARAM_RECOVERY_TIMEOUT, 0 },
    };
    constexpr size_t paramsNum = sizeof(params)/sizeof(params[0]);

    const auto found = discoverDevice(deviceIndex, params, paramsNum);
    if (found == -1)
    {
        return false;
    }
    gnaFileDescriptor = found;

    try
    {
        driverCapabilities.deviceVersion = static_cast<DeviceVersion>(params[0].value);
        driverCapabilities.recoveryTimeout = static_cast<uint32_t>(params[2].value);
        driverCapabilities.hwInBuffSize = static_cast<uint32_t>(params[1].value);
    }
    catch(std::out_of_range&)
    {
        return false;
    }

    return true;
}

LinuxDriverInterface::~LinuxDriverInterface()
{
    if (gnaFileDescriptor != -1)
    {
        close(gnaFileDescriptor);
    }
}

uint64_t LinuxDriverInterface::MemoryMap(void *memory, uint32_t memorySize)
{
    struct gna_userptr userptr;

    userptr.user_address = reinterpret_cast<uint64_t>(memory);
    userptr.user_size = memorySize;

    if(ioctl(gnaFileDescriptor, GNA_IOCTL_USERPTR, &userptr) != 0)
    {
        throw GnaException {Gna2StatusDeviceOutgoingCommunicationError};
    }

    return userptr.memory_id;
}

void LinuxDriverInterface::MemoryUnmap(uint64_t memoryId)
{
    if(ioctl(gnaFileDescriptor, GNA_IOCTL_FREE, memoryId) != 0)
    {
        throw GnaException {Gna2StatusDeviceOutgoingCommunicationError};
    }
}

RequestResult LinuxDriverInterface::Submit(HardwareRequest& hardwareRequest,
                                        RequestProfiler * const profiler) const
{
    RequestResult result = { };
    int ret;

    // TODO:kj:3: add working optimization mechanism to reduce recalculation for same request config
    createRequestDescriptor(hardwareRequest);

    auto scoreConfig = reinterpret_cast<struct gna_score_cfg *>(hardwareRequest.CalculationData.get());

    scoreConfig->ctrl_flags.gna_mode = hardwareRequest.Mode == xNN ? 1 : 0;
    scoreConfig->layer_count = hardwareRequest.LayerCount;

    if(xNN == hardwareRequest.Mode)
    {
        scoreConfig->layer_base = hardwareRequest.LayerBase;
    }
    else if(GMM == hardwareRequest.Mode)
    {
        scoreConfig->layer_base = hardwareRequest.GmmOffset;
        scoreConfig->ctrl_flags.active_list_on = hardwareRequest.GmmModeActiveListOn ? 1 : 0;
    }
    else
    {
        throw GnaException { Gna2StatusXnnErrorLyrCfg };
    }

    profiler->Measure(Gna2InstrumentationPointLibDeviceRequestReady);

    ret = ioctl(gnaFileDescriptor, GNA_IOCTL_SCORE, scoreConfig);
    if (ret == -1)
    {
        throw GnaException { Gna2StatusDeviceOutgoingCommunicationError };
    }

    gna_wait wait_data = {};
    wait_data.request_id = scoreConfig->request_id;
    wait_data.timeout = (driverCapabilities.recoveryTimeout + 1) * 1000;

    profiler->Measure(Gna2InstrumentationPointLibDeviceRequestSent);
    ret = ioctl(gnaFileDescriptor, GNA_IOCTL_WAIT, &wait_data);
    profiler->Measure(Gna2InstrumentationPointLibDeviceRequestCompleted);
    if(ret == 0)
    {
        result.status = ((wait_data.hw_status & GNA_STS_SATURATE) != 0)
            ? Gna2StatusWarningArithmeticSaturation
            : Gna2StatusSuccess;
        /*result.driverPerf.startHW = wait_data.drv_perf.start_hw;
        result.driverPerf.scoreHW = wait_data.drv_perf.score_hw;
        result.driverPerf.intProc = wait_data.drv_perf.intr_proc;*/
        result.hardwarePerf.total = wait_data.hw_perf.total;
        result.hardwarePerf.stall = wait_data.hw_perf.stall;
    }
    else
    {
        switch(errno)
        {
        case EIO:
            result.status = parseHwStatus(static_cast<uint32_t>(wait_data.hw_status));
            break;
        case EBUSY:
            result.status = Gna2StatusWarningDeviceBusy;
            break;
        case ETIME:
            result.status = Gna2StatusDeviceCriticalFailure;
            break;
        default:
            result.status = Gna2StatusDeviceIngoingCommunicationError;
            break;
        }
    }

    return result;
}

void LinuxDriverInterface::createRequestDescriptor(HardwareRequest& hardwareRequest) const
{
    auto& scoreConfigSize = hardwareRequest.CalculationSize;
    scoreConfigSize = sizeof(struct gna_score_cfg);

    for (const auto &buffer : hardwareRequest.DriverMemoryObjects)
    {
        scoreConfigSize += sizeof(struct gna_buffer) +
            buffer.Patches.size() * sizeof(struct gna_patch);
    }

    scoreConfigSize = RoundUp(scoreConfigSize, sizeof(uint64_t));
    hardwareRequest.CalculationData.reset(new uint8_t[scoreConfigSize]);

    uint8_t *calculationData = static_cast<uint8_t *>(hardwareRequest.CalculationData.get());
    auto scoreConfig = reinterpret_cast<struct gna_score_cfg *>(
                        hardwareRequest.CalculationData.get());
    memset(scoreConfig, 0, scoreConfigSize);
    scoreConfig->ctrl_flags.hw_perf_encoding = hardwareRequest.HwPerfEncoding;

    scoreConfig->buffers_ptr = reinterpret_cast<uintptr_t>(
                                calculationData + sizeof(struct gna_score_cfg));
    scoreConfig->buffer_count = hardwareRequest.DriverMemoryObjects.size();

    auto buffer = reinterpret_cast<struct gna_buffer *>(scoreConfig->buffers_ptr);
    auto patch = reinterpret_cast<struct gna_patch *>(scoreConfig->buffers_ptr +
                    scoreConfig->buffer_count * sizeof(struct gna_buffer));

    for (const auto &driverBuffer : hardwareRequest.DriverMemoryObjects)
    {
        buffer->memory_id = driverBuffer.Buffer.GetId();
        buffer->offset = 0;
        buffer->size = driverBuffer.Buffer.GetSize();
        buffer->patches_ptr = reinterpret_cast<uintptr_t>(patch);
        buffer->patch_count = driverBuffer.Patches.size();

        for (const auto &driverPatch : driverBuffer.Patches)
        {
            patch->offset = driverPatch.Offset;
            patch->size = driverPatch.Size;
            patch->value = driverPatch.Value;
            patch++;
        }

        buffer++;
    }

    hardwareRequest.SubmitReady = true;
}

Gna2Status LinuxDriverInterface::parseHwStatus(uint32_t hwStatus) const
{
    if ((hwStatus & GNA_STS_PCI_MMU_ERR) != 0)
    {
        return Gna2StatusDeviceMmuRequestError;
    }
    if ((hwStatus & GNA_STS_PCI_DMA_ERR) != 0)
    {
        return Gna2StatusDeviceDmaRequestError;
    }
    if ((hwStatus & GNA_STS_PCI_UNEXCOMPL_ERR) != 0)
    {
        return Gna2StatusDeviceUnexpectedCompletion;
    }
    if ((hwStatus & GNA_STS_VA_OOR) != 0)
    {
        return Gna2StatusDeviceVaOutOfRange;
    }
    if ((hwStatus & GNA_STS_PARAM_OOR) != 0)
    {
        return Gna2StatusDeviceParameterOutOfRange;
    }

    return Gna2StatusDeviceCriticalFailure;
}

int LinuxDriverInterface::discoverDevice(uint32_t deviceIndex, gna_getparam *params, size_t paramsNum)
{
    int fd = -1;
    uint32_t found = 0;
    for (uint8_t i = 0; i < MAX_GNA_DEVICES; i++)
    {
        char name[12];
        sprintf(name, "/dev/gna%hhu", i);
        fd = open(name, O_RDWR);
        if (-1 == fd)
        {
            continue;
        }

        bool paramsValid = true;
        for (size_t p = 0; p < paramsNum && paramsValid; p++)
        {
            paramsValid &= ioctl(fd, GNA_IOCTL_GETPARAM, &params[p]) == 0;
        }
        if (paramsValid && found++ == deviceIndex)
        {
            return fd;
        }

        close(fd);
        fd = -1;
    }
    return -1;
}

#endif // not defined WIN32