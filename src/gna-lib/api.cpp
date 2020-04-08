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

#include "DeviceManager.h"
#include "Expect.h"
#include "GnaException.h"
#include "Logger.h"

#include "gna2-common-impl.h"
#include "gna-api-dumper.h"
#include "ModelWrapper.h"

#include <cstddef>
#include <memory>

using namespace GNA;

static intel_gna_status_t HandleUnknownException(const std::exception& e)
{
    Log->Error("Unknown exception: %s.", e.what());
    return GNA_UNKNOWN_ERROR;
}

/******************************************************************************
 *
 * API routines implementation
 *
 *****************************************************************************/

GNAAPI gna_status_t GnaModelCreate(
    gna_device_id deviceId,
    gna_model const *model,
    gna_model_id *modelId)
{
    try
    {
        Expect::NotNull(modelId);
        Expect::NotNull(model);
        auto& device = DeviceManager::Get().GetDevice(deviceId);
        *modelId = device.LoadModel(*model);
        return GNA_SUCCESS;
    }
    catch (const GnaModelException &e)
    {
        return e.GetLegacyStatus();
    }
    catch (const GnaException &e)
    {
        return e.GetLegacyStatus();
    }
    catch (const std::exception& e)
    {
        return HandleUnknownException(e);
    }
}
