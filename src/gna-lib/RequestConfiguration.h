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

#include "LayerConfiguration.h"

#include "gna-api.h"
#include "gna-api-instrumentation.h"
#include "gna-api-types-xnn.h"
#include "gna2-inference-api.h"
#include "gna2-inference-impl.h"

#include "gna2-common-impl.h"

#include <map>
#include <memory>
#include <vector>
#include <cstdint>

namespace GNA
{
class CompiledModel;
class Memory;
struct ActiveList;

/*
** RequestConfiguration is a bunch of request buffers
** sent to GNA kernel driver as part of WRITE request
**
 */
class RequestConfiguration
{
public:
    RequestConfiguration(CompiledModel& model, gna_request_cfg_id configId, DeviceVersion consistentDeviceIn);

    void AddBuffer(GnaComponentType type, uint32_t layerIndex, void *address);
    void AddActiveList(uint32_t layerIndex, const ActiveList& activeList);
    void SetHardwareConsistency(DeviceVersion consistentDeviceIn);
    void EnforceAcceleration(Gna2AccelerationMode accelMode)
    {
        Acceleration.SetMode(accelMode);
    }

    bool HasConsistencyMode() const
    {
        return Acceleration.GetHwConsistency();
    }
    DeviceVersion GetConsistentDevice() const;

    CompiledModel& Model;

    const gna_request_cfg_id Id;

    gna_hw_perf_encoding HwPerfEncoding = PERF_COUNT_DISABLED;
    gna_perf_t * PerfResults = nullptr;

    std::map<uint32_t, std::unique_ptr<LayerConfiguration>> LayerConfigurations;

    std::vector<Memory *> MemoryList;

    uint32_t ActiveListCount = 0;

    // Number of elements in buffer per input precision and per grouping
    uint32_t BufferElementCount[2 * XNN_N_GROUP_MAX];

    AccelerationMode Acceleration = Gna2AccelerationModeAuto;

private:
    DeviceVersion consistentDevice;

    void addMemoryObject(void *buffer, uint32_t bufferSize);
};
}
