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

#include "LinuxIoctlSender.h"

#include "GnaException.h"
#include "Logger.h"

#include <errno.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <stropts.h>

using namespace GNA;

using std::unique_ptr;

void LinuxIoctlSender::Open()
{
    struct gna_capabilities gnaCaps;
    int found = 0;
    int ret;
    int fd;

    for(int i = 0; i < 16; i++)
    {
        char name[12];
        sprintf(name, "/dev/gna%d", i);
        fd = open(name, O_RDWR);
        if(-1 == fd)
        {
            continue;
        }

        if(!ioctl(fd, GNA_CPBLTS, &gnaCaps))
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
        deviceCapabilities.deviceKind = deviceTypeMap.at(gnaCaps.device_type);
    }
    catch(std::out_of_range &e)
    {
        throw GnaException { GNA_DEVNOTFOUND };
    }
    deviceCapabilities.hwInBuffSize = gnaCaps.in_buff_size;
    deviceCapabilities.recoveryTimeout = gnaCaps.recovery_timeout;
}

LinuxIoctlSender::~LinuxIoctlSender()
{
    if (gnaFileDescriptor != -1)
    {
        close(gnaFileDescriptor);
    }
}

uint64_t LinuxIoctlSender::MemoryMap(void *memory, size_t memorySize)
{
    struct gna_usrptr usrptr;

    usrptr.padd = reinterpret_cast<uint64_t>(memory);
    usrptr.length = memorySize;
    usrptr.memory_id = 0;

    if(ioctl(gnaFileDescriptor, GNA_MAP_USRPTR, &usrptr))
    {
        throw GnaException {GNA_IOCTLSENDERR};
    }

    return usrptr.memory_id;
}

void LinuxIoctlSender::MemoryUnmap(uint64_t memoryId)
{
    struct gna_usrptr usrptr;
    usrptr.padd = NULL;
    usrptr.length = 0;
    usrptr.memory_id = memoryId;
    if(ioctl(gnaFileDescriptor, GNA_UNMAP_USRPTR, &usrptr))
    {
        throw GnaException {GNA_IOCTLSENDERR};
    }
}

GnaCapabilities LinuxIoctlSender::GetDeviceCapabilities() const
{
    return deviceCapabilities;
}

void LinuxIoctlSender::IoctlSend(const GnaIoctlCommand command, void * const inbuf, const uint32_t inlen,
    void * const outbuf, const uint32_t outlen)
{
    throw GnaException {GNA_IOCTLSENDERR};
}

RequestResult LinuxIoctlSender::Submit(HardwareRequest * const hardwareRequest, RequestProfiler * const profiler)
{
    RequestResult result = { 0 };
    int ret;

    if(!hardwareRequest->SubmitReady)
    {
        createRequestDescriptor(hardwareRequest);
    }

    auto scoreConfig = reinterpret_cast<struct gna_score_cfg *>(hardwareRequest->CalculationData.get());

    scoreConfig->flags.active_list_on = hardwareRequest->ActiveListOn;
    scoreConfig->flags.gna_mode = hardwareRequest->Mode == xNN ? 1 : 0;
    scoreConfig->flags.layer_count = hardwareRequest->LayerCount;

    if(xNN == hardwareRequest->Mode)
    {
        scoreConfig->flags.config_base = hardwareRequest->LayerBase;
    }
    else if(GMM == hardwareRequest->Mode)
    {
        scoreConfig->flags.config_base = hardwareRequest->GmmOffset;
    }
    else
    {
        throw GnaException { XNN_ERR_LYR_CFG };
    }

    profilerTscStart(&profiler->ioctlSubmit);
    ret = ioctl(gnaFileDescriptor, GNA_SCORE, scoreConfig);
    profilerTscStop(&profiler->ioctlSubmit);
    if (ret)
    {
        throw GnaException { GNA_IOCTLSENDERR };
    }

    gna_wait wait_data;
    wait_data.request_id = scoreConfig->request_id;
    wait_data.timeout = GNA_REQUEST_TIMEOUT_MAX;

    profilerTscStart(&profiler->ioctlWaitOn);
    ret = ioctl(gnaFileDescriptor, GNA_WAIT, &wait_data);
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

void LinuxIoctlSender::createRequestDescriptor(HardwareRequest *hardwareRequest)
{
    auto& scoreConfigSize = hardwareRequest->CalculationSize;
    scoreConfigSize = sizeof(struct gna_score_cfg);
    auto ioBuffersCount = hardwareRequest->IoBuffers.size();
    auto ioBuffersSize = ioBuffersCount * sizeof(hardwareRequest->IoBuffers[0]);
    auto nnopTypesCount = hardwareRequest->NnopTypes.size();
    auto nnopTypesSize = nnopTypesCount * sizeof(hardwareRequest->NnopTypes[0]);
    auto xnnActiveListsCount = hardwareRequest->XnnActiveLists.size();
    auto xnnActiveListsSize = xnnActiveListsCount * sizeof(hardwareRequest->XnnActiveLists[0]);
    auto gmmActiveListsCount = hardwareRequest->GmmActiveLists.size();
    auto gmmActiveListsSize = gmmActiveListsCount * sizeof(hardwareRequest->GmmActiveLists[0]);

    scoreConfigSize += ioBuffersSize +  nnopTypesSize +  xnnActiveListsSize +  gmmActiveListsSize;
    scoreConfigSize = ALIGN(scoreConfigSize, sizeof(uint64_t));
    hardwareRequest->CalculationData.reset(new uint8_t[scoreConfigSize]);

    auto scoreConfig = reinterpret_cast<struct gna_score_cfg *>(hardwareRequest->CalculationData.get());
    memset(scoreConfig, 0, scoreConfigSize);
    scoreConfig->memory_id = hardwareRequest->MemoryId;
    scoreConfig->hw_perf_encoding = hardwareRequest->HwPerfEncoding;
    scoreConfig->req_cfg_desc.model_id = hardwareRequest->ModelId;
    scoreConfig->req_cfg_desc.request_cfg_id = hardwareRequest->RequestConfigId;
    scoreConfig->req_cfg_desc.buffer_count = ioBuffersCount;
    scoreConfig->req_cfg_desc.xnn_al_count = xnnActiveListsCount;
    scoreConfig->req_cfg_desc.gmm_al_count = gmmActiveListsCount;
    scoreConfig->req_cfg_desc.nnop_type_count = nnopTypesCount;

    uint8_t *requestData = reinterpret_cast<uint8_t*>(scoreConfig) + sizeof(gna_score_cfg);
    uint8_t *calculationEnd = requestData + scoreConfigSize;
    memcpy_s(requestData, calculationEnd - requestData, hardwareRequest->IoBuffers.data(), ioBuffersSize);
    requestData += ioBuffersSize;
    memcpy_s(requestData, calculationEnd - requestData, hardwareRequest->NnopTypes.data(), nnopTypesSize);
    requestData += nnopTypesSize;
    memcpy_s(requestData, calculationEnd - requestData, hardwareRequest->XnnActiveLists.data(), xnnActiveListsSize);
    requestData += xnnActiveListsCount;
    memcpy_s(requestData, calculationEnd - requestData, hardwareRequest->GmmActiveLists.data(), gmmActiveListsSize);

    hardwareRequest->SubmitReady = true;
}

status_t LinuxIoctlSender::parseHwStatus(__u32 hwStatus) const
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

