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
#include <mutex>

#include "Request.h"

using std::map;
using std::mutex;

namespace GNA
{

class RequestManager {
    /**
     * Requests map container
     */
    static map<uint32_t, Request*> requests;

    /**
     * iterator of submitted requests (during session life [open-close])
     */
    static uint32_t nRequests;

    /**
     * mutex for synchronizing request map operations
     */
    static mutex*  lock;

public:
    /**
     * Inserts and submits request for calculation
     *
     * @reqId   (out)(optional) id of submitted request
     */
    static status_t insertRequest(Request *r, const acceleration nProcessorType);

    /**
     * Retrieves request from container
     *
     * @id  id of the request to be found
     * @return  request found or NULL
     */
    static Request * getRequest(const uint32_t id);

    /**
     * Removes request from container
     *
     * @iter    request to be removed
     * @return  status of removal
     */
    static status_t removeRequest(Request* r);

    /**
     * Releases request
     *
     * @r    request to be deleted
     */
    static void deleteRequest(Request** r);

    static void init();

    static status_t clear();
};

}
