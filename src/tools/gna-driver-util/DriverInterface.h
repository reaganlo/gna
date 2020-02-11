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
#pragma once
#include "GnaUtilConfig.h"
#include "gna-api.h"
#include "Memory.h"

#include <stdint.h>
#include <vector>

#if _WIN32
#include <basetsd.h>
#include "GnaDrvApi.h"
#endif

//TODO: change in case of new generation
constexpr int descriptorSize = 128;

struct DriverCapabilities
{
    uint32_t hwInBuffSize;
    uint32_t recoveryTimeout;
    gna_device_version deviceVersion;
};

typedef struct
{
    uint64_t            total;      // # of total cycles spent on scoring in hw
    uint64_t            stall;      // # of stall cycles spent in hw (since scoring)
} HardwarePerfResults;

typedef struct
{
    uint64_t            startHW;    // time of setting up and issuing HW scoring
    uint64_t            scoreHW;    // time between HW scoring start and scoring complete interrupt
    uint64_t            intProc;    // time of processing scoring complete interrupt
} DriverPerfResults;

struct RequestResult
{
    HardwarePerfResults hardwarePerf;
    DriverPerfResults driverPerf;
    gna_status_t status;
};

enum class GnaOperationMode : uint8_t
{
    GMM = 0,
    xNN = 1
};

class DriverBuffer
{
public:
    DriverBuffer(Memory const& memoryIn) :
        Buffer(memoryIn)
    {}
    Memory const& Buffer;
};

class HardwareRequest
{
public:
    std::unique_ptr<uint8_t[]> CalculationData;
    std::vector<DriverBuffer> DriverMemoryObjects;
};

class DriverInterface
{
public:
    static constexpr uint8_t MAX_GNA_DEVICES = 16;

    virtual bool discoverDevice(uint32_t device) = 0;

    virtual ~DriverInterface() = default;

    virtual uint64_t MemoryMap(void* memory, uint32_t memorySize) = 0;

    virtual void MemoryUnmap(uint64_t memoryId) = 0;

    virtual RequestResult Submit(
        HardwareRequest & hardwareRequest, const GnaUtilConfig& file) const = 0;
};
