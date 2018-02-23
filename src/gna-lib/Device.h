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

#include "common.h"

#include "AccelerationDetector.h"
#include "RequestBuilder.h"
#include "RequestHandler.h"

namespace GNA
{
class Memory;
class Device
{
public:
    Device(gna_device_id *deviceId, uint8_t threadCount = 1);
    virtual ~Device();
    Device(const Device &) = delete;
    Device& operator=(const Device&) = delete;

    void ValidateSession(gna_device_id deviceId) const;

    void * AllocateMemory(uint32_t requestedSize, const uint16_t layerCount, uint16_t gmmCount, uint32_t * sizeGranted);

    void FreeMemory();

    void LoadModel(gna_model_id *modelId, const gna_model *model);

    void AttachBuffer(gna_request_cfg_id configId, gna_buffer_type type, uint16_t layerIndex, void *address);

    void CreateConfiguration(gna_model_id modelId, gna_request_cfg_id *configId);

    void AttachActiveList(gna_request_cfg_id configId, uint16_t layerIndex, uint32_t indicesCount, const uint32_t* const indices);

    void EnableProfiling(gna_request_cfg_id configId, gna_hw_perf_encoding hwPerfEncoding, gna_perf_t * perfResults);

    void PropagateRequest(gna_request_cfg_id configId, acceleration accel, gna_request_id *requestId);

    status_t WaitForRequest(gna_request_id requestId, gna_timeout milliseconds);

    void* Dump(gna_model_id modelId, gna_device_kind deviceKind, intel_gna_model_header* modelHeader, intel_gna_status_t* status, intel_gna_alloc_cb customAlloc);

protected:
    virtual std::unique_ptr<Memory> createMemoryObject(const uint64_t memoryId, const uint32_t requestedSize,
        const uint16_t layerCount, const uint16_t gmmCount);

    static const std::map<const gna_device_kind, const GnaDeviceType> deviceDictionary;

    gna_device_id id = GNA_DEVICE_INVALID;

    RequestHandler requestHandler;

    RequestBuilder requestBuilder;

    std::vector<std::unique_ptr<Memory>> memoryObjects;

    std::unique_ptr<IoctlSender> ioctlSender;

    AccelerationDetector accelerationDetector;

    uint32_t modelIdSequence = 0;
};
}
#endif
