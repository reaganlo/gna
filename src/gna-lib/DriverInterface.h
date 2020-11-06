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

#include "Request.h"
#include "Expect.h"

#include "gna2-common-impl.h"

#include <map>

namespace GNA
{

class HardwareRequest;

struct HardwarePerfResults
{
    uint64_t total; // # of total cycles spent on scoring in hw
    uint64_t stall; // # of stall cycles spent in hw (since scoring)
};

struct DriverPerfResults
{
    /**
     Request preprocessing start
     */
    uint64_t Preprocessing;

    /**
     Request processing started by hardware
     */
    uint64_t Processing;

    /**
     Request completed interrupt triggered by hardware
     */
    uint64_t DeviceRequestCompleted;

    /**
     Driver completed interrupt and request handling.
     */
    uint64_t Completion;
};

struct RequestResult
{
    HardwarePerfResults hardwarePerf;
    DriverPerfResults driverPerf;
    Gna2Status status;
};

enum GnaIoctlCommand
{
    GNA_COMMAND_MAP,
    GNA_COMMAND_UNMAP,
    GNA_COMMAND_SCORE,
    GNA_COMMAND_GET_PARAM,
#if HW_VERBOSE == 1
    GNA_COMMAND_READ_PGDIR,
    GNA_COMMAND_READ_REG,
    GNA_COMMAND_WRITE_REG
#endif
};

struct DriverCapabilities
{
    uint32_t hwInBuffSize;
    uint32_t recoveryTimeout;
    DeviceVersion deviceVersion;

    /**
     Number of ticks of driver performance counter per second.
     */
    uint64_t perfCounterFrequency;
};

class DriverInterface
{
public:
    static constexpr uint8_t MAX_GNA_DEVICES = 16;

    virtual bool OpenDevice(uint32_t deviceIndex) = 0;

    virtual ~DriverInterface() = default;

    const DriverCapabilities& GetCapabilities() const;

    virtual uint64_t MemoryMap(void *memory, uint32_t memorySize) = 0;

    virtual void MemoryUnmap(uint64_t memoryId) = 0;

    virtual RequestResult Submit(
        HardwareRequest& hardwareRequest, RequestProfiler & profiler) const = 0;

protected:
    DriverInterface() = default;
    DriverInterface(const DriverInterface &) = delete;
    DriverInterface& operator=(const DriverInterface&) = delete;

    virtual void createRequestDescriptor(HardwareRequest& hardwareRequest) const = 0;

    virtual Gna2Status parseHwStatus(uint32_t hwStatus) const = 0;

    void convertPerfResultUnit(DriverPerfResults & driverPerf,
        Gna2InstrumentationUnit targetUnit) const;

    static void convertPerfResultUnit(DriverPerfResults & driverPerf,
        uint64_t frequency, uint64_t multiplier);

    DriverCapabilities driverCapabilities;
};

}
