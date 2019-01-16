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
class Memory;
class Device
{
public:
    Device(gna_device_id *deviceId, uint8_t threadCount = 1);
    Device(const Device &) = delete;
    Device& operator=(const Device&) = delete;

    void ValidateSession(gna_device_id deviceId) const;

    void * AllocateMemory(uint32_t requestedSize, const uint16_t layerCount, uint16_t gmmCount, uint32_t * sizeGranted);

    void FreeMemory(void * const buffer);

    void FreeMemory();

    void LoadModel(gna_model_id *modelId, const gna_model *rawModel);

    void AttachBuffer(gna_request_cfg_id configId, GnaComponentType type, uint32_t layerIndex, void *address);

    void CreateConfiguration(gna_model_id modelId, gna_request_cfg_id *configId);

    void AttachActiveList(gna_request_cfg_id configId, uint32_t layerIndex, uint32_t indicesCount, const uint32_t* const indices);

    void EnableProfiling(gna_request_cfg_id configId, gna_hw_perf_encoding hwPerfEncoding, gna_perf_t * perfResults);

    void PropagateRequest(gna_request_cfg_id configId, acceleration accel, gna_request_id *requestId);

    status_t WaitForRequest(gna_request_id requestId, gna_timeout milliseconds);

    void Stop();

    void* Dump(gna_model_id modelId, gna_device_generation generation, intel_gna_model_header* modelHeader, intel_gna_status_t* status, intel_gna_alloc_cb customAlloc);

protected:
    virtual std::unique_ptr<Memory> createMemoryObject(const uint32_t requestedSize,
        const uint16_t layerCount, const uint16_t gmmCount);

    static const std::map<const gna_device_generation, const gna_device_version> deviceDictionary;
    inline void* getMemoryId(gna_model_id modelId) const
    {
        try
        {
            return modelMemoryMap.at(modelId);
        }
        catch (std::out_of_range&)
        {
            throw GnaException(GNA_INVALID_MODEL);
        }
    }

    inline Memory* getMemory(void* memoryId) const
    {
        if (nullptr == memoryId)
        {
            auto memIter = memoryObjects.begin(); // TODO:3: CUrrent API does not allow memory buffer selection thus using first and the only
            if (memoryObjects.end() == memIter)
            {
                throw GnaException(GNA_ERR_MEMORY_NOT_ALLOCATED);
            }
            memoryId = memIter->first;
        }
        try
        {
             return memoryObjects.at(memoryId).get();
        }
        catch (std::out_of_range&)
        {
            throw GnaException(GNA_ERR_MEMORY_NOT_ALLOCATED);
        }
    }

    inline std::unique_ptr<Memory>& getMemoryObj(void* memoryId)
    {
        try
        {
             return memoryObjects.at(memoryId);
        }
        catch (std::out_of_range&)
        {
            throw GnaException(GNA_ERR_MEMORY_NOT_ALLOCATED);
        }
    }

    gna_device_id id = GNA_DEVICE_INVALID;

    uint32_t modelIdSequence = 0;

    std::unique_ptr<IoctlSender> ioctlSender;

    AccelerationDetector accelerationDetector;

    std::map<void*, std::unique_ptr<Memory>> memoryObjects;

    std::map<gna_model_id, void*> modelMemoryMap;

    RequestBuilder requestBuilder;

    RequestHandler requestHandler;
};
}
#endif
