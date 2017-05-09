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

#include "AccelerationDetector.h"
#include "AcceleratorHw.h"
#include "AcceleratorHwVerbose.h"
#include "AcceleratorSw.h"
#include "GnaException.h"
#include "Logger.h"

using std::make_shared;
using std::shared_ptr;

using namespace GNA;

AcceleratorController::AcceleratorController(AccelerationDetector& detector) :
    isHardwarePresent{detector.IsHardwarePresent()}
{
    // attempt to create Hardware Accelerator
    if (isHardwarePresent)
    {
        accelerators[GNA_HW] = make_shared<AcceleratorHw>(GNA_HW);
    }

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

    auto fastestMode = detector.GetFastestAcceleration();
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
            Log->Error(e.getStatus(), "Creating accelerator with acc mode %d failed.\n", GNA_GEN_FAST);
        }
    }

    accelerators[GNA_SW_FAST] = accelerators[fastestMode];
    accelerators[GNA_SW_SAT] = accelerators[static_cast<acceleration>(fastestMode - 1)];

    if (isHardwarePresent)
    {
        accelerators[GNA_AUTO_FAST] = accelerators[GNA_HW];
        accelerators[GNA_AUTO_SAT] = accelerators[GNA_HW];
    }
    else
    {
        accelerators[GNA_AUTO_FAST] = accelerators[fastestMode];
        accelerators[GNA_AUTO_SAT] = accelerators[static_cast<acceleration>(fastestMode - 1)];
    }
}

ScoreMethod AcceleratorController::getScoreMethod(const std::vector<std::unique_ptr<SubModel>>& subModels,
    acceleration accel) const
{
    // acceleration mode validation
    if ((accel >= NUM_GNA_ACCEL_MODES || accel < 2) && accel != GNA_HW)
    {
        throw GnaException(GNA_CPUTYPENOTSUPPORTED);
    }

    // hardware requested, but no hardware exists
    if (GNA_HW == accel && !isHardwarePresent)
        throw GnaException(GNA_DEVNOTFOUND);

    // software acceleration
    if (accel != GNA_AUTO_FAST && accel != GNA_AUTO_SAT && accel != GNA_HW)
    {
        return SoftwareOnly;
    }

    // hardware or auto acceleration
    // there is only one submodel, which means
    if (subModels.size() == 1)
    {
        switch(subModels.front()->Type)
        {
        // hardware does not support model
        case Software:
            return SoftwareOnly;
        // hardware can handle whole model
        case Hardware:
            return HardwareOnly;
        default:
            throw GnaException(GNA_UNKNOWN_ERROR);
        }
    }

    // otherwise hardware can handle particular layers
    // the rest should be scored by software
    return Mixed;
}

status_t AcceleratorController::ScoreModel(
    RequestConfiguration& config,
    acceleration accel,
    RequestProfiler *profiler,
    KernelBuffers *buffers) const
{
    auto status = GNA_SUCCESS;
    auto& subModels = config.Model.GetSubmodels();
    auto scoreMethod = getScoreMethod(subModels, accel);
    switch (scoreMethod)
    {
    case SoftwareOnly:
    case HardwareOnly:
        return accelerators.at(accel)->Score(0, config.Model.LayerCount, config, profiler, buffers);
    case Mixed:
    {
        auto& acceleratorHw = accelerators.at(GNA_HW);
        auto& acceleratorSw = accelerators.at(accel);
        for (const auto& submodel : subModels)
        {
            uint32_t layerIndex = submodel->LayerIndex;
            uint32_t layerCount = submodel->GetLayerCount();
            switch (submodel->Type)
            {
            case Software:
                status = acceleratorSw->Score(layerIndex, layerCount, config, profiler, buffers);
                if (status != GNA_SUCCESS && status != GNA_SSATURATE)
                    return status;
                break;
            case Hardware:
                status = acceleratorHw->Score(layerIndex, layerCount, config, profiler, buffers);
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
