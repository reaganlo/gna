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

#ifndef _DEVICE_H
#define _DEVICE_H

#include "gna-api-dumper.h"

#include "common.h"

#include "AccelerationDetector.h"
#include "RequestBuilder.h"
#include "RequestHandler.h"

namespace GNA
{

class DeviceManager
{
public:
    static uint32_t GetDeviceCount();

    static gna_device_version GetDeviceVersion(gna_device_id deviceId);

    static void SetThreadCount(gna_device_id deviceId, uint32_t threadCount);

    static uint32_t GetThreadCount(gna_device_id deviceId);

    static void VerifyDeviceIndex(gna_device_id deviceId);


private:
    static std::map<uint32_t, uint32_t> threadCountMap;
};


class Memory;
class Device
{
public:
    Device(gna_device_id deviceId, uint32_t threadCount = 1);
    Device(const Device &) = delete;
    Device& operator=(const Device&) = delete;

    void ValidateSession(gna_device_id deviceId) const;

    status_t AllocateMemory(uint32_t requestedSize, uint32_t * sizeGranted, void **memoryAddress);

    void FreeMemory(void *const buffer);

    void LoadModel(gna_model_id *modelId, const gna_model *userModel);

    void ReleaseModel(gna_model_id const modelId);

    void AttachBuffer(gna_request_cfg_id configId, GnaComponentType type, uint32_t layerIndex, void *address);

    void CreateConfiguration(gna_model_id modelId, gna_request_cfg_id *configId);

    void ReleaseConfiguration(gna_request_cfg_id configId);

    void SetHardwareConsistency(gna_request_cfg_id configId, gna_device_version hardwareVersion);

    void EnforceAcceleration(gna_request_cfg_id configId, AccelerationMode accel);

    void AttachActiveList(gna_request_cfg_id configId, uint32_t layerIndex, uint32_t indicesCount, const uint32_t* const indices);

    void EnableProfiling(gna_request_cfg_id configId, gna_hw_perf_encoding hwPerfEncoding, gna_perf_t * perfResults);

    void PropagateRequest(gna_request_cfg_id configId, gna_request_id *requestId);

    status_t WaitForRequest(gna_request_id requestId, gna_timeout milliseconds);

    void Stop();

    void* Dump(gna_model_id modelId, gna_device_generation generation, intel_gna_model_header* modelHeader, intel_gna_status_t* status, intel_gna_alloc_cb customAlloc);

protected:
    virtual std::unique_ptr<Memory> createMemoryObject( const uint32_t requestedSize);

    static const std::map<const gna_device_generation, const gna_device_version> deviceDictionary;

    gna_device_id id;

    uint32_t modelIdSequence = 0;

    std::unique_ptr<DriverInterface> driverInterface;

    HardwareCapabilities hardwareCapabilities;

    AccelerationDetector accelerationDetector;

    std::map<gna_model_id, std::unique_ptr<CompiledModel>> models;

    std::vector<std::unique_ptr<Memory>> memoryObjects;

    RequestBuilder requestBuilder;

    RequestHandler requestHandler;
};
}
#endif
