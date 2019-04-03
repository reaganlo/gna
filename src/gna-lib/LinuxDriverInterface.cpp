/*
 INTEL CONFIDENTIAL
 Copyright 2018 Intel Corporation.

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

#include "LinuxDriverInterface.h"

#include "GnaException.h"
#include "HardwareRequest.h"
#include "Logger.h"
#include "Memory.h"
#include "gna2-common-impl.h"

#include <errno.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>

using namespace GNA;

using std::unique_ptr;

void LinuxDriverInterface::OpenDevice()
{
    int found = 0;
    int fd;
    struct gna_getparam params[3] =
    {
        { .param = GNA_PARAM_DEVICE_ID },
        { .param = GNA_PARAM_IBUFFS },
        { .param = GNA_PARAM_RECOVERY_TIMEOUT },
    };

    for(uint8_t i = 0; i < 16; i++)
    {
        char name[12];
        sprintf(name, "/dev/gna%hhu", i);
        fd = open(name, O_RDWR);
        if(-1 == fd)
        {
            continue;
        }

        if(!ioctl(fd, GNA_IOCTL_GETPARAM, &params[0])
            && !ioctl(fd, GNA_IOCTL_GETPARAM, &params[1])
            && !ioctl(fd, GNA_IOCTL_GETPARAM, &params[2]))
        {
            found = 1;
            break;
        }
    }

    if(!found)
        throw GnaException {GNA_DEVNOTFOUND};

    gnaFileDescriptor = fd;

    try
    {
        driverCapabilities.hwId = static_cast<DeviceVersion>(params[0].value);
    }
    catch(std::out_of_range &e)
    {
        throw GnaException { GNA_DEVNOTFOUND };
    }
    driverCapabilities.hwInBuffSize = params[1].value;
    driverCapabilities.recoveryTimeout = params[2].value;

    opened = true;
}

LinuxDriverInterface::~LinuxDriverInterface()
{
    if (gnaFileDescriptor != -1)
    {
        close(gnaFileDescriptor);
    }
}

uint64_t LinuxDriverInterface::MemoryMap(void *memory, size_t memorySize)
{
    struct gna_userptr userptr;

    userptr.user_address = reinterpret_cast<uint64_t>(memory);
    userptr.user_size = memorySize;

    if(ioctl(gnaFileDescriptor, GNA_IOCTL_USERPTR, &userptr))
    {
        throw GnaException {GNA_IOCTLSENDERR};
    }

    return userptr.memory_id;
}

void LinuxDriverInterface::MemoryUnmap(uint64_t memoryId)
{
    if(ioctl(gnaFileDescriptor, GNA_IOCTL_FREE, memoryId))
    {
        throw GnaException {GNA_IOCTLSENDERR};
    }
}

DriverCapabilities LinuxDriverInterface::GetCapabilities() const
{
    return driverCapabilities;
}

void LinuxDriverInterface::IoctlSend(const GnaIoctlCommand command,
                                    void * const inbuf, const uint32_t inlen,
                                    void * const outbuf, const uint32_t outlen)
{
    throw GnaException {GNA_IOCTLSENDERR};
}

RequestResult LinuxDriverInterface::Submit(HardwareRequest& hardwareRequest,
                                        RequestProfiler * const profiler) const
{
    RequestResult result = { 0 };
    int ret;

    if(!hardwareRequest.SubmitReady)
    {
        createRequestDescriptor(hardwareRequest);
    }

    auto scoreConfig = reinterpret_cast<struct gna_score_cfg *>(hardwareRequest.CalculationData.get());

    scoreConfig->ctrl_flags.active_list_on = hardwareRequest.ActiveListOn;
    scoreConfig->ctrl_flags.gna_mode = hardwareRequest.Mode == xNN ? 1 : 0;
    scoreConfig->layer_count = hardwareRequest.LayerCount;

    if(xNN == hardwareRequest.Mode)
    {
        scoreConfig->layer_base = hardwareRequest.LayerBase;
    }
    else if(GMM == hardwareRequest.Mode)
    {
        scoreConfig->layer_base = hardwareRequest.GmmOffset;
    }
    else
    {
        throw GnaException { XNN_ERR_LYR_CFG };
    }

    profilerTscStart(&profiler->ioctlSubmit);
    ret = ioctl(gnaFileDescriptor, GNA_IOCTL_SCORE, scoreConfig);
    profilerTscStop(&profiler->ioctlSubmit);
    if (ret)
    {
        throw GnaException { GNA_IOCTLSENDERR };
    }

    gna_wait wait_data = {};
    wait_data.request_id = scoreConfig->request_id;
    wait_data.timeout = GNA_REQUEST_TIMEOUT_MAX;

    profilerTscStart(&profiler->ioctlWaitOn);
    ret = ioctl(gnaFileDescriptor, GNA_IOCTL_WAIT, &wait_data);
    profilerTscStop(&profiler->ioctlWaitOn);
    if(!ret)
    {
        result.status = (wait_data.hw_status & GNA_STS_SATURATE) ? GNA_SSATURATE : GNA_SUCCESS;
        result.driverPerf.startHW = wait_data.drv_perf.start_hw;
        result.driverPerf.scoreHW = wait_data.drv_perf.score_hw;
        result.driverPerf.intProc = wait_data.drv_perf.intr_proc;
        result.hardwarePerf.total = wait_data.hw_perf.total;
        result.hardwarePerf.stall = wait_data.hw_perf.stall;
    }
    else switch(errno)
    {
        case EIO:
            result.status = parseHwStatus(wait_data.hw_status);
            break;
        case EBUSY:
            result.status = GNA_DEVICEBUSY;
            break;
        case ETIME:
            result.status = GNA_ERR_DEV_FAILURE;
            break;
        default:
            result.status = GNA_IOCTLRESERR;
            break;
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

    scoreConfigSize = ALIGN(scoreConfigSize, sizeof(uint64_t));
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
        buffer->memory_id = driverBuffer.Buffer->GetId();
        buffer->offset = 0;
        buffer->size = driverBuffer.Buffer->GetSize();
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

status_t LinuxDriverInterface::parseHwStatus(__u32 hwStatus) const
{
    if (hwStatus & GNA_STS_PCI_MMU_ERR)
    {
        return GNA_MMUREQERR;
    }
    if (hwStatus & GNA_STS_PCI_DMA_ERR)
    {
        return GNA_DMAREQERR;
    }
    if (hwStatus & GNA_STS_PCI_UNEXCOMPL_ERR)
    {
        return GNA_UNEXPCOMPL;
    }
    if (hwStatus & GNA_STS_VA_OOR)
    {
        return GNA_VAOUTOFRANGE;
    }
    if (hwStatus & GNA_STS_PARAM_OOR)
    {
        return GNA_PARAMETEROUTOFRANGE;
    }

    return GNA_ERR_DEV_FAILURE;
}

