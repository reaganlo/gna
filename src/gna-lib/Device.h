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

#pragma once

#include "AccelerationDetector.h"
#include "CompiledModel.h"
#include "DriverInterface.h"
#include "HardwareCapabilities.h"
#include "Memory.h"
#include "RequestBuilder.h"
#include "RequestHandler.h"

#include "gna-api.h"
#include "gna-api-dumper.h"
#include "gna2-instrumentation-api.h"

#include <cstdint>
#include <map>
#include <memory>
#include <vector>

namespace GNA
{

class Device
{
public:
    Device(uint32_t deviceIndex, uint32_t threadCount = 1);
    Device(const Device &) = delete;
    Device& operator=(const Device&) = delete;
    virtual ~Device() = default;

    DeviceVersion GetVersion() const;

    uint32_t GetNumberOfThreads() const;

    void SetNumberOfThreads(uint32_t threadCount);

    template<class T>
    uint32_t LoadModel(const T& model)
    {
        auto compiledModel = std::make_unique<CompiledModel>(
            model, accelerationDetector, hardwareCapabilities);

        if (!compiledModel)
        {
            throw GnaException(Gna2StatusResourceAllocationError);
        }

        auto modelId = modelIdSequence++;

        compiledModel->BuildHardwareModel(*driverInterface);
        models.emplace(modelId, std::move(compiledModel));
        return modelId;
    }

    void ReleaseModel(gna_model_id const modelId);

    void AttachBuffer(gna_request_cfg_id configId, uint32_t operandIndex, uint32_t layerIndex, void *address);

    void CreateConfiguration(gna_model_id modelId, gna_request_cfg_id *configId);

    void ReleaseConfiguration(gna_request_cfg_id configId);

    void EnableHardwareConsistency(gna_request_cfg_id configId, DeviceVersion deviceVersion);

    void EnforceAcceleration(gna_request_cfg_id configId, Gna2AccelerationMode accel);

    void AttachActiveList(gna_request_cfg_id configId, uint32_t layerIndex, uint32_t indicesCount, const uint32_t* const indices);

    void PropagateRequest(gna_request_cfg_id configId, uint32_t *requestId);

    Gna2Status WaitForRequest(gna_request_id requestId, gna_timeout milliseconds);

    void Stop();

    void* Dump(gna_model_id modelId, intel_gna_model_header* modelHeader, Gna2Status* status,
            intel_gna_alloc_cb customAlloc);

    void DumpLdNoMMu(gna_model_id modelId, intel_gna_alloc_cb customAlloc,
        void *& exportData, uint32_t & exportDataSize);

    void CreateProfilerConfiguration(uint32_t* configId, uint32_t numberOfInstrumentationPoints, Gna2InstrumentationPoint* selectedInstrumentationPoints, uint64_t* results);

    void ReleaseProfilerConfiguration(uint32_t configId);

    void AssignProfilerConfigToRequestConfig(uint32_t instrumentationConfigId, uint32_t requestConfigId);

    void SetInstrumentationUnit(gna_request_cfg_id configId, Gna2InstrumentationUnit instrumentationUnit);

    void SetHardwareInstrumentation(gna_request_cfg_id configId, enum Gna2InstrumentationMode instrumentationMode);

    bool HasModel(uint32_t modelId) const;

    bool HasMemory(void * memory) const;

    bool HasRequestConfigId(uint32_t requestConfigId) const;

    bool HasRequestId(uint32_t requestId) const;

    void MapMemory(Memory& memoryObject);

    void UnMapMemory(Memory & memoryObject);

protected:
    static const std::map<const gna_device_generation, const DeviceVersion> deviceDictionary;

    gna_device_id id;

    static uint32_t modelIdSequence;

    std::unique_ptr<DriverInterface> driverInterface;

    HardwareCapabilities hardwareCapabilities;

    AccelerationDetector accelerationDetector;

    std::map<gna_model_id, std::unique_ptr<CompiledModel>> models;

    RequestBuilder requestBuilder;

    RequestHandler requestHandler;

private:
    uint32_t numberOfThreads;
};
}
