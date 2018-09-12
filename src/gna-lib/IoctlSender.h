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

#pragma once

#include "gna-api-dumper.h"

#include <map>

#include "common.h"
#include "Request.h"
#include "Validator.h"

namespace GNA
{

class HardwareRequest;

struct RequestResult
{
    perf_hw_t hardwarePerf;
    perf_drv_t driverPerf;
    status_t status;
};

enum GnaIoctlCommand
{
    GNA_COMMAND_MAP,
    GNA_COMMAND_UNMAP,
    GNA_COMMAND_SCORE,
    GNA_COMMAND_CAPABILITIES,
#if HW_VERBOSE == 1
    GNA_COMMAND_READ_PGDIR,
    GNA_COMMAND_READ_REG,
    GNA_COMMAND_WRITE_REG
#endif
};

struct GnaCapabilities
{
    uint32_t hwInBuffSize;
    uint32_t recoveryTimeout;
    gna_device_kind deviceKind;
};

class IoctlSender
{
public:
    virtual void IoctlSend(const GnaIoctlCommand command, void * const inbuf, const uint32_t inlen,
        void * const outbuf, const uint32_t outlen) = 0;

    virtual void Open() = 0;

    virtual ~IoctlSender() = default;

    virtual GnaCapabilities GetDeviceCapabilities() const = 0;

    virtual uint64_t MemoryMap(void *memory, size_t memorySize) = 0;

    virtual void MemoryUnmap(uint64_t memoryId) = 0;

    virtual RequestResult Submit(HardwareRequest * const hardwareRequest, RequestProfiler * const profiler) = 0;

protected:
    IoctlSender() = default;
    IoctlSender(const IoctlSender &) = delete;
    IoctlSender& operator=(const IoctlSender&) = delete;

    static const std::map<uint32_t, gna_device_kind> deviceTypeMap;
};

}
