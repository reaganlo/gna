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

#include "GnaException.h"
#include "SwHw.h"

namespace GNA
{
#ifdef PROFILE

/**
 * Library level request processing profiler
 */
typedef struct _request_profiler
{
    prof_tsc_t      submit;         // score request submit profiler
    prof_tsc_t      preprocess;     // preprocessing score request profiler
    prof_tsc_t      process;        // processing score request profiler (includes GNAWait)
    prof_tsc_t      scoring;        // profiler for computing scores in software mode
    prof_tsc_t      total;          // profiler for total processing time (does not include GNAWait)
    prof_tsc_t      ioctlSubmit;    // profiler for issuing "start scoring IOCTL"
    prof_tsc_t      ioctlWaitOn;    // profiler for waiting for "start scoring IOCTL" completion

} req_profiler;                     // Library level request processing profiler

#endif // PROFILE

using RequestFunctor = std::function<status_t(aligned_fv_bufs*, req_profiler*)>;

/**
 * Calculation request for single scoring or propagate forward operation
 */
class Request
{
public:
    /**
     * Creates empty request
     */
    Request(
        RequestFunctor callback,
        std::unique_ptr<req_profiler>&& profiler);

    /**
     * Destroys request resources if any
     */
    ~Request() {}

    void operator()(aligned_fv_bufs *buffers)
    {
        scoreTask(buffers, profiler.get());
    }

    /**************************************************************************
     * Properties
     *************************************************************************/

    /**
     * External id (0-GNA_REQUEST_WAIT_ANY)
     */
    gna_request_id id = 0;

#ifdef PROFILE
    
    /**
     * performance profiler
     */
    unique_ptr<req_profiler> profiler;

#endif

    std::future<status_t> GetFuture();

    // should be used only by RequestHandler
    void SetId(gna_request_id requestId);

private:
    std::packaged_task<status_t(aligned_fv_bufs *buffers, req_profiler *profiler)> scoreTask;

    status_t scoreStatus = GNA_DEVICEBUSY;

    /**
     * Deleted functions to prevent from being defined or called
     * @see: https://msdn.microsoft.com/en-us/library/dn457344.aspx
     */
    Request() = delete;
    Request(const Request &) = delete;
    Request& operator=(const Request&) = delete;
};

}
