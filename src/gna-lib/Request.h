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

#include <functional>
#include <future>

#include "profiler.h"

#include "GnaException.h"
#include "KernelArguments.h"

namespace GNA
{

class RequestConfiguration;

/**
 * Library level request processing profiler
 */
struct RequestProfiler
{
    profiler_tsc submit;         // score request submit profiler
    profiler_tsc preprocess;     // preprocessing score request profiler
    profiler_tsc process;        // processing score request profiler (includes GNAWait)
    profiler_tsc scoring;        // profiler for computing scores in software mode
    profiler_tsc total;          // profiler for total processing time (does not include GNAWait)
    profiler_tsc ioctlSubmit;    // profiler for issuing "start scoring IOCTL"
    profiler_tsc ioctlWaitOn;    // profiler for waiting for "start scoring IOCTL" completion
}; // Library level request processing profiler

/**
 * Calculation request for single scoring or propagate forward operation
 */
class Request
{
public:
    Request(RequestConfiguration& config, std::unique_ptr<RequestProfiler> profiler, acceleration accel);
    ~Request() = default;
    Request() = delete;
    Request(const Request &) = delete;
    Request& operator=(const Request&) = delete;

    std::future<status_t> GetFuture();

    void operator()(KernelBuffers *buffers)
    {
        scoreTask(buffers, Profiler.get());
    }

    // External id (0-GNA_REQUEST_WAIT_ANY)
    gna_request_id Id = 0;
    RequestConfiguration& Configuration;
    acceleration Accel;
    std::unique_ptr<RequestProfiler> Profiler;
    gna_perf_t *PerfResults;

private:
    std::packaged_task<status_t(KernelBuffers *buffers, RequestProfiler *profiler)> scoreTask;

    status_t scoreStatus = GNA_DEVICEBUSY;
};

}
