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

#include "KernelArguments.h"

#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <deque>
#include <thread>
#include <vector>

namespace GNA
{
class Request;

class ThreadPool {
public:
    explicit ThreadPool(uint32_t threadCount);
    ~ThreadPool();
    ThreadPool(const ThreadPool &) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    uint32_t GetNumberOfThreads() const;

    void SetNumberOfThreads(uint32_t threadCount);

    void Enqueue(Request *request);
    void StopAndJoin();

private:
    void employWorkers();

    // NOTE: order is important, buffers have to be destroyed last
    std::vector<KernelBuffers> buffers;
    std::mutex tpMutex;
    std::deque<Request*> tasks;
    bool stopped = false;
    std::condition_variable condition;
    std::vector<std::thread> workers;
    uint32_t numberOfThreads;
};

}
