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

#ifndef _DEVICE_H
#define _DEVICE_H

#include "AcceleratorController.h"
#include "ModelCompiler.h"
#include "ModelContainer.h"
#include "RequestBuilder.h"
#include "RequestHandler.h"

namespace GNA
{
/**
 * Accelerator Manager for opening and closing accelerator device
 */
class Device
{
public:
    Device(gna_device_id *deviceId, uint8_t threadCount = 1);
    ~Device();

    void ValidateSession(gna_device_id deviceId) const;

    const size_t AllocateMemory(const size_t requestedSize, void **buffer);

    void FreeMemory();

    /**
     ** Allocated and compiles raw model
     **
     ** @intelNnet      (in)
     */
    void LoadModel(gna_model_id *modelId, const gna_model *model);

    /**
     ** Deallocates model
     **
     ** @modelId      (in)
     */
    void ReleaseModel(gna_model_id modelId);

    void AttachBuffer(gna_request_cfg_id configId, gna_buffer_type type, uint16_t layerIndex, void *address);

    void CreateConfiguration(gna_model_id modelId, gna_request_cfg_id *configId);

    void AttachActiveList(gna_request_cfg_id configId, uint16_t layerIndex, uint32_t indicesCount, uint32_t *indices);

    void EnableProfiling(gna_request_cfg_id configId, gna_hw_perf_encoding hwPerfEncoding, gna_perf_t * perfResults);

    /**
     ** Propagates request for execution
     *
     * @modelId         (in)
     * @requestId       (in)
     */
    void PropagateRequest(gna_request_cfg_id configId, acceleration accel, gna_request_id *requestId);

    /**
     ** Propagates request for execution
     *
     * @requestId       (in)
     */
    status_t WaitForRequest(gna_request_id requestId, gna_timeout milliseconds);


    /**
     * Deleted functions to prevent from being defined or called
     * @see: https://msdn.microsoft.com/en-us/library/dn457344.aspx
     */
    Device() = delete;
    Device(const Device &) = delete;
    Device& operator=(const Device&) = delete;

private:
    /*
    * Unique device id
    */
    gna_device_id id = GNA_DEVICE_INVALID;

    std::unique_ptr<Memory> totalMemory;
    RequestHandler requestHandler;

    // !!! ORDER_DEPENDENCY !!!
    // accelerationController depends on accelerationDetector
    AccelerationDetector accelerationDetector;
    AcceleratorController acceleratorController;

    ModelContainer modelContainer;
    ModelCompiler modelCompiler;
    RequestBuilder requestBuilder;
};
}
#endif
