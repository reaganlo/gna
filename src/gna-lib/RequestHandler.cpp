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

#include "Device.h"
#include "GnaException.h"
#include "RequestHandler.h"

using std::unique_ptr;
using std::pair;
using std::mutex;

using namespace GNA;

void RequestHandler::Enqueue(
    gna_request_id *requestId,
    unique_ptr<Request> request)
{
    {
        std::lock_guard<std::mutex> lockGuard(lock);

        if (requests.size() >= GNA_REQUEST_QUEUE_LENGTH)
        {
            throw GnaException(GNA_ERR_QUEUE);
        }

        *requestId = nRequests;
        auto insert = requests.try_emplace(*requestId, move(request));

        if (!insert.second)
        {
            throw GnaException(GNA_ERR_RESOURCES);
        }

        requests.at(*requestId)->SetId(*requestId);

        nRequests = (++nRequests) % GNA_REQUEST_WAIT_ANY;
    }

    threadPool.Enqueue(requests.at(*requestId).get());
}

status_t RequestHandler::removeRequest(gna_request_id requestId)
{
    std::lock_guard<std::mutex> lockGuard(lock);
    auto r = requests.find(requestId);
    if (requests.end() != r)
    {
        requests.erase(requestId);
        return GNA_SUCCESS;
    }
    return GNA_BADREQID;
}

void RequestHandler::initRequestMap()
{
    nRequests = 0;
}

void RequestHandler::clearRequestMap()
{
    std::lock_guard<std::mutex> lockGuard(lock);
    requests.clear();
}

void RequestHandler::Init(uint8_t threadCount)
{
    initRequestMap();
    threadPool.Init(threadCount);
}

void RequestHandler::ClearRequests()
{
    threadPool.Stop();
    clearRequestMap();
}

status_t RequestHandler::WaitFor(const gna_request_id requestId, const gna_timeout milliseconds)
{
    auto request = requests.at(requestId).get();
    auto future = request->GetFuture();

    auto future_status = future.wait_for(std::chrono::milliseconds(milliseconds));
    switch (future_status)
    {
    case std::future_status::ready:
    {
        auto score_status = future.get();
        auto status = removeRequest(requestId);
        Expect::True(GNA_SUCCESS == status, status);
        return score_status;
    }
    default:
        return GNA_DEVICEBUSY;
    }
}
