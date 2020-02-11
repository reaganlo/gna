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

//TODO extract common part from library and util into separate file to omit redundancy.
#ifndef WIN32
#include "LinuxDriverInterface.h"

#include "gna-api.h"
#include "gna-api-status.h"

#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

bool LinuxDriverInterface::OpenDevice(uint32_t deviceIndex)
{
    struct gna_getparam params[] =
    {
        { GNA_PARAM_DEVICE_TYPE, 0 },
        { GNA_PARAM_INPUT_BUFFER_S, 0 },
        { GNA_PARAM_RECOVERY_TIMEOUT, 0 },
    };

    constexpr size_t paramsNum = sizeof(params)/sizeof(params[0]);

    const auto found = discover(deviceIndex, params, paramsNum);
    if (found == -1)
    {
        return false;
    }
    gnaFileDescriptor = found;

    try
    {
        driverCapabilities.deviceVersion = static_cast<gna_device_version>(params[0].value);
        driverCapabilities.recoveryTimeout = static_cast<uint32_t>(params[2].value);
        driverCapabilities.hwInBuffSize = static_cast<uint32_t>(params[1].value);
        std::cout << "hwInBuffSize: " << driverCapabilities.hwInBuffSize << std::endl;
        std::cout << "deviceVersion: " << driverCapabilities.deviceVersion << std::endl;
        std::cout << "recoveryTimeout: " << driverCapabilities.recoveryTimeout << std::endl;
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
    userptr.memory_id = 0; //all struct members should be initialized

    if(ioctl(gnaFileDescriptor, GNA_IOCTL_USERPTR, &userptr) != 0)
    {
        throw GNA_IOCTLSENDERR;
    }

    return userptr.memory_id;
}

void LinuxDriverInterface::MemoryUnmap(uint64_t memoryId)
{
    if(ioctl(gnaFileDescriptor, GNA_IOCTL_FREE, memoryId) != 0)
    {
        throw GNA_IOCTLSENDERR;
    }
}

void LinuxDriverInterface::createRequestDescriptor(HardwareRequest& hardwareRequest,
    const GnaUtilConfig& utilConfig) const
{
    auto scoreConfigSize = utilConfig.inferenceConfigSize;
    auto bufferOffset = utilConfig.modelConfig.bufferOffset;
    auto patchesOffset = utilConfig.modelConfig.patchOffset;

    hardwareRequest.CalculationData.reset(new uint8_t[scoreConfigSize]);

    auto scoreConfig = reinterpret_cast<struct gna_score_cfg *>(
                        hardwareRequest.CalculationData.get());

    memset(scoreConfig, 0, scoreConfigSize);

    memcpy(scoreConfig, utilConfig.modelConfig.inference, utilConfig.inferenceConfigSize);

    scoreConfig->buffers_ptr = reinterpret_cast<uintptr_t>(&utilConfig.modelConfig.inference[bufferOffset]);
    auto buffer = reinterpret_cast<struct gna_buffer *>(scoreConfig->buffers_ptr);
    buffer->patches_ptr = reinterpret_cast<uintptr_t>(&utilConfig.modelConfig.inference[patchesOffset]);
}

RequestResult LinuxDriverInterface::Submit(HardwareRequest& hardwareRequest,
    const GnaUtilConfig& utilConfig) const
{
    RequestResult result = { };
    int ret;

    createRequestDescriptor(hardwareRequest, utilConfig);

    auto scoreConfig = reinterpret_cast<struct gna_score_cfg *>(hardwareRequest.CalculationData.get());

    ret = ioctl(gnaFileDescriptor, GNA_IOCTL_SCORE, scoreConfig);
    if (ret == -1)
    {
        throw GNA_IOCTLSENDERR;
    }

    gna_wait wait_data = {};
    wait_data.request_id = scoreConfig->request_id;
    wait_data.timeout = (driverCapabilities.recoveryTimeout + 1) * 1000;

    ret = ioctl(gnaFileDescriptor, GNA_IOCTL_WAIT, &wait_data);

    if(ret == 0)
    {
        result.status = ((wait_data.hw_status & GNA_STS_SATURATE) != 0)
            ? GNA_SSATURATE
            : GNA_SUCCESS;
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
            result.status = GNA_DEVICEBUSY;
            break;
        case ETIME:
            result.status = GNA_ERR_DEV_FAILURE;
            break;
        default:
            result.status = GNA_IOCTLRESERR;
            break;
        }
    }

    return result;
}

gna_status_t LinuxDriverInterface::parseHwStatus(uint32_t hwStatus) const
{
    if ((hwStatus & GNA_STS_PCI_MMU_ERR) != 0)
    {
        return GNA_MMUREQERR;
    }
    if ((hwStatus & GNA_STS_PCI_DMA_ERR) != 0)
    {
        return GNA_DMAREQERR;
    }
    if ((hwStatus & GNA_STS_PCI_UNEXCOMPL_ERR) != 0)
    {
        return GNA_UNEXPCOMPL;
    }
    if ((hwStatus & GNA_STS_VA_OOR) != 0)
    {
        return GNA_VAOUTOFRANGE;
    }
    if ((hwStatus & GNA_STS_PARAM_OOR) != 0)
    {
        return GNA_PARAMETEROUTOFRANGE;
    }

    return GNA_ERR_DEV_FAILURE;
}

int LinuxDriverInterface::discover(uint32_t deviceIndex, gna_getparam *params, size_t paramsNum)
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

bool LinuxDriverInterface::discoverDevice(uint32_t deviceIndex)
{
    return OpenDevice(deviceIndex);
}
#endif
