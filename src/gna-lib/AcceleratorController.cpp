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

#include "AcceleratorController.h"
#include "AcceleratorHw.h"
#include "AcceleratorHwVerbose.h"
#include "AcceleratorSw.h"
#include "CompiledModel.h"
#include "GnaException.h"
#include "RequestConfiguration.h"

using std::make_shared;
using std::shared_ptr;

using namespace GNA;

void AcceleratorController::CreateAccelerators(bool isHardwarePresent, acceleration fastestMode)
{
    hardwarePresent = isHardwarePresent;

    // attempt to create Hardware Accelerator
    if(hardwarePresent)
        accelerators[GNA_HW] = make_shared<AcceleratorHw>(GNA_HW);

    for (uint32_t mode = GNA_AUTO_SAT; mode < NUM_GNA_ACCEL_MODES; mode++)
    {
        accelerators[static_cast<acceleration>(mode)] = nullptr;
    }
    accelerators[GNA_CNL_SAT] = nullptr;
    accelerators[GNA_CNL_FAST] = nullptr;
    for (uint32_t mode = GNA_AUTO_SAT; mode < NUM_GNA_ACCEL_MODES; mode++)
    {
        accelerators[static_cast<acceleration>(mode)] = nullptr;
    }
    accelerators[GNA_CNL_SAT] = nullptr;
    accelerators[GNA_CNL_FAST] = nullptr;

    for (uint32_t mode = fastestMode; mode >= GNA_GEN_SAT; mode--)
    {
        acceleration gnaAcc = static_cast<acceleration>(mode);
        try
        {
            accelerators[gnaAcc] = make_shared<AcceleratorSw>(gnaAcc);
        }
        catch (GnaException &e)
        {
            accelerators[gnaAcc] = nullptr;
            ERR("Creating accelerator with acc mode %d failed with status: %d", GNA_GEN_FAST, e.getStatus());
        }
    }

    accelerators[GNA_AUTO_FAST] = accelerators[fastestMode];
    accelerators[GNA_SW_FAST] = accelerators[fastestMode];

    accelerators[GNA_AUTO_SAT] = accelerators[static_cast<acceleration>(fastestMode - 1)];
    accelerators[GNA_SW_SAT] = accelerators[static_cast<acceleration>(fastestMode - 1)];
}

void AcceleratorController::ClearAccelerators()
{
    uint32_t i;
    AccMapIter a;
    uint32_t accMapSize = accelerators.size();
    for (i = 0; i < accMapSize; i++)
    {
        a = accelerators.begin();
        if (accelerators.end() != a)
        {
            accelerators.erase(a);
        }
    }
    accelerators.clear();
}

ScoreMethod AcceleratorController::getScoreMethod(CompiledModel &model, acceleration accel) const
{
    // acceleration mode validation
    if ((accel >= NUM_GNA_ACCEL_MODES || accel < 2) && accel != GNA_HW)
    {
        throw GnaException(GNA_CPUTYPENOTSUPPORTED);
    }

    // hardware requested, but no hardware exists
    if (GNA_HW == accel && !hardwarePresent)
        throw GnaException(GNA_DEVNOTFOUND);

    // software acceleration
    if (accel != GNA_AUTO_FAST && accel != GNA_AUTO_SAT && accel != GNA_HW)
    {
        return SoftwareOnly;
    }

    // hardware or auto acceleration
    const auto& submodels = model.GetSubmodels();

    // there is only one submodel, which means 
    if (submodels.size() == 1)
    {
        switch(submodels.front()->Type)
        {
        // hardware does not support model
        case Software:
            return SoftwareOnly;
        // hardware can handle whole model
        case Hardware:
            return HardwareOnly;
        default:
            throw GnaException(GNA_ERR_UNKNOWN);
        }
    }

    // otherwise hardware can handle particular layers
    // the rest should be scored by software
    return Mixed;
}

status_t AcceleratorController::ScoreModel(
    CompiledModel& model, 
    RequestConfiguration& config,
    acceleration accel,
    req_profiler *profiler,
    aligned_fv_bufs *buffers)
{
    auto status = GNA_SUCCESS;
    auto scoreMethod = getScoreMethod(model, accel);
    switch (scoreMethod)
    {
    case SoftwareOnly:
    case HardwareOnly:
        return accelerators[accel]->Score(model, config, profiler, buffers);
    case Mixed:
    {
        auto& acceleratorHw = accelerators[GNA_HW];
        auto& acceleratorSw = accelerators[accel];
        for (const auto& submodel : model.GetSubmodels())
        {
            uint32_t layerIndex = submodel->GetLayerIndex();
            uint32_t layerCount = submodel->GetLayerCount();
            switch (submodel->Type)
            {
            case Software:
                status = acceleratorSw->Score(model, *submodel.get(), config, profiler, buffers);
                if (status != GNA_SUCCESS && status != GNA_SSATURATE)
                    return status;
                break;
            case Hardware:
                status = acceleratorHw->Score(model, *submodel.get(), config, profiler, buffers);
                if (status != GNA_SUCCESS && status != GNA_SSATURATE)
                    return status;
                break;
            case GMMHardware:
                throw GnaException(GNA_CPUTYPENOTSUPPORTED);
            }
        }
        break;
    }
    }

    return status;
}
