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

#pragma once

#include "LayerConfiguration.h"
#include "HardwareLayer.h"
#include "MemoryContainer.h"
#include "ProfilerConfiguration.h"
#include "Tensor.h"

#include "gna-api.h"
#include "gna2-common-impl.h"
#include "gna2-inference-api.h"
#include "gna2-inference-impl.h"

#include <map>
#include <memory>
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
    RequestConfiguration(CompiledModel& model, uint32_t configId, DeviceVersion consistentDeviceIn);

    ~RequestConfiguration() = default;

    void AddBuffer(uint32_t operandIndex, uint32_t layerIndex, void *address);

    void AddActiveList(uint32_t layerIndex, const ActiveList& activeList);

    void SetHardwareConsistency(DeviceVersion consistentDeviceIn);

    void EnforceAcceleration(Gna2AccelerationMode accelMode);

    bool HasConsistencyMode() const
    {
        return Acceleration.GetHwConsistency();
    }
    DeviceVersion GetConsistentDevice() const;

    void AssignProfilerConfig(ProfilerConfiguration* config)
    {
        profilerConfiguration = config;
    }

    ProfilerConfiguration* GetProfilerConfiguration() const
    {
        return profilerConfiguration;
    }

    uint8_t GetHwInstrumentationMode() const;

    MemoryContainer const & GetAllocations() const
    {
        return allocations;
    }

    CompiledModel & Model;

    const uint32_t Id;

    std::map<uint32_t, std::unique_ptr<LayerConfiguration>> LayerConfigurations;


    uint32_t ActiveListCount = 0;

    // Number of elements in buffer per input precision and per grouping
    uint32_t const * BufferElementCount = nullptr;
    uint32_t const * BufferElementCountForAdl = nullptr;

    AccelerationMode Acceleration = Gna2AccelerationModeAuto;

private:
    struct AddBufferContext
    {
        AddBufferContext(CompiledModel & model, uint32_t operandIndex, uint32_t layerIndex, void *address);

        Layer const * SoftwareLayer;
        Tensor const * Operand;
        uint32_t OperandIndex;
        uint32_t LayerIndex;
        void * Address;
        uint32_t Size;
    };

    void storeAllocationIfNew(void const * buffer, uint32_t bufferSize);

    void applyBufferForSingleLayer(AddBufferContext & context);

    void addBufferForMultipleLayers(AddBufferContext & context);

    void addBufferForSingleLayer(AddBufferContext & context);

    LayerConfiguration & getLayerConfiguration(uint32_t layerIndex);

    ProfilerConfiguration* profilerConfiguration = nullptr;

    DeviceVersion consistentDevice;

    MemoryContainer allocations;
};

}
