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

#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <map>
#include <mutex>
#include <deque>
#include <thread>
#include <vector>

#include "common.h"
#include "Request.h"

namespace GNA
{
class ThreadPool {
public:
    ThreadPool(uint8_t nThreads);
    ~ThreadPool();
    ThreadPool(const ThreadPool &) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    void CancelTasks(const gna_model_id modelId);
    void Enqueue(Request *request);
    void Stop();

private:
    std::vector<KernelBuffers> buffers; // NOTE: order is important, buffers have to be destroyed last
    std::mutex tpMutex;
    std::deque<Request*> tasks;
    bool stopped = false;
    std::condition_variable condition;
    std::vector<std::thread> workers;
};

}
