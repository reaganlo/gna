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

#include "GnaException.h"
#include "ThreadPool.h"

#include "gna-api.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace GNA
{
class Request;

class RequestHandler
{
public:

    explicit RequestHandler(uint32_t threadCount);

    ~RequestHandler();

    uint32_t GetNumberOfThreads() const;

    void ChangeNumberOfThreads(uint32_t threadCount);

    void Enqueue(
        gna_request_id *requestId,
        std::unique_ptr<Request> request);

    Gna2Status WaitFor(const gna_request_id requestId, const gna_timeout milliseconds);

    void StopRequests();

    bool HasRequest(uint32_t requestId) const;

private:

    void clearRequestMap();

    void removeRequest(uint32_t requestId);

    Request * get(const uint32_t requestId)
    {
        try
        {
            auto& request = requests.at(requestId);
            return request.get();
        }
        catch (const std::out_of_range&)
        {
            throw GnaException(Gna2StatusIdentifierInvalid);
        }
    }

    static uint32_t assignRequestId();

    std::unordered_map<uint32_t, std::unique_ptr<Request>> requests;
    std::mutex lock;
    ThreadPool threadPool;
};

}
