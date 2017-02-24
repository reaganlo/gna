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

#include <map>

#include "common.h"
#include "Request.h"
#include "RequestConfiguration.h"

namespace GNA
{
    class CompiledModel;

    class Request;
class IAccelerator;

/**
 * Accelerator Interface
 */
class IAccelerator
{
public:

    /**
     * Submits Xnn request for calculation
     *
     * @network     Neural network model or nullptr
     * @actIndices  active indices data
     * @nActIndices active indices number
     * @reqId       (out)(optional) id of submitted request
     * @return  status submission
     */
    virtual status_t Score(
        const CompiledModel&        model,
        const RequestConfiguration& config,
        const uint32_t              layerIndex,
        const uint32_t              layerCount) = 0;

    /**
     * Waits for completion of Scoring IOCTL by GMM device driver
     * retrieves results and profiling info
     * dequeues and releases request if not in processing anymore
     *
     * @r       scoring request to complete
     * @timeout time [ms] after that wait fails if not completed
     * @perfResults buffer to save performance results to, or nullptr
     * @return  status of request processing, not scoring status
     */
    virtual status_t Wait(
        Request*        r,
        uint32_t        timeout,
        perf_t*         perfResults) = 0;


    virtual ~IAccelerator() {};
};

}
