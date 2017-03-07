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
#include "Request.h"
#include "RequestConfiguration.h"
#include "SubModel.h"

namespace GNA
{

/**
 * Accelerator Interface
 */
class IAccelerator
{
public:

    IAccelerator(acceleration acceleration_mode) : accel(acceleration_mode) {}
    /**
     * Scores the whole xNN model
     */
    virtual status_t Score(
        const CompiledModel&        model,
        const RequestConfiguration& config,
              RequestProfiler *profiler,
              KernelBuffers *buffers) = 0;

    /**
     * Scores part of xNN model described by submodel
     */
    virtual status_t Score(
        const CompiledModel&        model,
        const SubModel&             submodel,
        const RequestConfiguration& config,
              RequestProfiler *profiler,
              KernelBuffers *buffers) = 0;

    virtual ~IAccelerator() = default;
    
protected:
    acceleration accel;
};

}
