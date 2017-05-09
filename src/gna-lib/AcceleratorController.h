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

#include <cstdint>
#include <map>

#include "common.h"
#include "IAccelerator.h"
#include "Request.h"


namespace GNA
{
typedef enum _ScoreMethod
{
    SoftwareOnly,
    HardwareOnly,
    Mixed
} ScoreMethod;

class AcceleratorController
{
public:
    AcceleratorController(AccelerationDetector &detector);
    ~AcceleratorController() = default;

    status_t ScoreModel(
        RequestConfiguration& config,
        acceleration accel,
        RequestProfiler *profiler,
        KernelBuffers *buffers) const;

private:
    std::map<acceleration, std::shared_ptr<IAccelerator>> accelerators;
    bool isHardwarePresent;

    ScoreMethod getScoreMethod(const std::vector<std::unique_ptr<SubModel>>& subModels, acceleration accel) const;

    AcceleratorController(const AcceleratorController&) = delete;
    AcceleratorController& operator=(const AcceleratorController&) = delete;
};

}
