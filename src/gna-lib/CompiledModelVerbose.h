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

#include "CompiledModel.h"
#include "HardwareModelVerbose.h"

namespace GNA
{

class CompiledModelVerbose : public CompiledModel
{
public:
    CompiledModelVerbose(const gna_model& userModel,
        const AccelerationDetector& detectorIn,
        const HardwareCapabilities& hwCapabilitiesIn,
        std::vector<std::unique_ptr<Memory>>& memoryObjects) :
        CompiledModel(userModel, detectorIn, hwCapabilitiesIn, memoryObjects)
    {
    };

    void CompiledModelVerbose::SetPrescoreScenario(uint32_t nActions, dbg_action *actions)
    {
        if (hardwareModel)
        {
            auto& hardwareModelVerbose = static_cast<HardwareModelVerbose&>(*hardwareModel);
            hardwareModelVerbose.SetPrescoreScenario(nActions, actions);
        }
    }

    void CompiledModelVerbose::SetAfterscoreScenario(uint32_t nActions, dbg_action *actions)
    {
        if (hardwareModel)
        {
            auto& hardwareModelVerbose = static_cast<HardwareModelVerbose&>(*hardwareModel);
            hardwareModelVerbose.SetAfterscoreScenario(nActions, actions);
        }
    }
};
}


