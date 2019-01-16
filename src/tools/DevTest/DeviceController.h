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

#include "gna-api.h"
#include "gna-api-verbose.h"

class DeviceController
{
public:
    DeviceController();
    ~DeviceController();

    uint8_t * Alloc(uint32_t sizeRequested, uint16_t layerCount, uint16_t gmmCount, uint32_t * sizeGranted);

    void Free();

    void ModelCreate(const gna_model *, gna_model_id *);

    gna_request_cfg_id ConfigAdd(gna_model_id);

    void BufferAdd(gna_request_cfg_id, GnaComponentType, uint32_t layerIndex, void * address);

    void RequestEnqueue(gna_request_cfg_id, gna_acceleration, gna_request_id *);
    void RequestWait(gna_request_id);

    void ActiveListAdd(gna_request_cfg_id configId, uint32_t layerIndex, uint32_t indicesCount, uint32_t* indices);

#if HW_VERBOSE == 1
    void AfterscoreDebug(gna_model_id modelId, uint32_t nActions, dbg_action *actions);

    void PrescoreDebug(gna_model_id modelId, uint32_t nActions, dbg_action *actions);
#endif

private:
    gna_device_id gnaHandle;
};
