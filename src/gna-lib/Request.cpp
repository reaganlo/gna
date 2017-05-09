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

#include "Request.h"

#include "AcceleratorController.h"
#include "RequestConfiguration.h"

using std::function;
using std::future;
using std::move;
using std::unique_ptr;

using namespace GNA;

Request::Request(RequestConfiguration& config, std::unique_ptr<RequestProfiler> profiler, acceleration accel,
        const AcceleratorController& acceleratorController) :
    Configuration(config),
    Accel{accel},
    Profiler{move(profiler)},
    PerfResults{config.PerfResults}
{
    if (PerfResults)
    {
        memset(PerfResults, 0, sizeof(gna_perf_t));
    }
    auto callback = [&, accel](KernelBuffers *buffers, RequestProfiler *profilerPtr)
    {
        return acceleratorController.ScoreModel(Configuration, Accel, profilerPtr, buffers);
    };
    scoreTask = std::packaged_task<status_t(KernelBuffers *buffers, RequestProfiler *profiler)>(callback);
}

future<status_t> Request::GetFuture()
{
    return scoreTask.get_future();
}