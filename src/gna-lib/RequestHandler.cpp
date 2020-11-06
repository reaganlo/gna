/*
 INTEL CONFIDENTIAL
 Copyright 2017-2020 Intel Corporation.

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


#include "Expect.h"
#include "GnaException.h"
#include "Request.h"
#include "RequestHandler.h"
#include "RequestConfiguration.h"

#include <chrono>
#include <future>
#include <utility>

using namespace GNA;

// TODO:4:KJ: refactor to have ThreadPool per device and RequestHandler per model
RequestHandler::RequestHandler(uint32_t threadCount) : threadPool(threadCount)
{}

RequestHandler::~RequestHandler()
{
    clearRequestMap();
}

uint32_t RequestHandler::GetNumberOfThreads() const
{
    return threadPool.GetNumberOfThreads();
}

void RequestHandler::ChangeNumberOfThreads(uint32_t threadCount)
{
    threadPool.SetNumberOfThreads(threadCount);
}

void RequestHandler::Enqueue(
    uint32_t *requestId,
    std::unique_ptr<Request> request)
{
    Expect::NotNull(requestId);
    auto r = request.get();
    {
        std::lock_guard<std::mutex> lockGuard(lock);

        if (requests.size() >= QueueLengthMax)
        {
            throw GnaException(Gna2StatusDeviceQueueError);
        }

        *requestId = assignRequestId();
        r->Id = *requestId;
        addRequest(std::move(request));
    }
    r->Profiler->Measure(Gna2InstrumentationPointLibSubmission);

    threadPool.Enqueue(r);
}

void RequestHandler::addRequest(std::unique_ptr<Request> request)
{
    auto insert = requests.emplace(request->Id, move(request));
    if (!insert.second)
    {
        throw GnaException(Gna2StatusResourceAllocationError);
    }
}

Gna2Status RequestHandler::WaitFor(const uint32_t requestId, const uint32_t milliseconds)
{
    auto request = extractRequestLocked(requestId);

    auto const status = request->WaitFor(milliseconds);
    if (Gna2StatusWarningDeviceBusy == status)
    {
        addRequestLocked(std::move(request));
    }
    return status;
}

void RequestHandler::StopRequests()
{
    threadPool.StopAndJoin();
}

bool RequestHandler::HasRequest(uint32_t requestId) const
{
    return requests.count(requestId) > 0;
}

std::unique_ptr<Request> RequestHandler::extractRequestLocked(const uint32_t requestId)
{
    std::lock_guard<std::mutex> lockGuard(lock);
    auto found = requests.find(requestId);
    if(found == requests.end())
    {
        throw GnaException(Gna2StatusIdentifierInvalid);
    }
    auto extracted = std::move(found->second);
    requests.erase(found);
    return extracted;
}

void RequestHandler::addRequestLocked(std::unique_ptr<Request> request)
{
    std::lock_guard<std::mutex> lockGuard(lock);
    addRequest(std::move(request));
}

uint32_t RequestHandler::assignRequestId()
{
    static uint32_t id;
    return id++;
}

void RequestHandler::clearRequestMap()
{
    std::lock_guard<std::mutex> lockGuard(lock);
    requests.clear();
}
