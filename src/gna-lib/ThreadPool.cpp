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

#include "ThreadPool.h"

#include <cstring>

#include "common.h"
#include "RequestConfiguration.h"
#include "Validator.h"
#include "KernelArguments.h"

using namespace GNA;

KernelBuffers::KernelBuffers()
{
    // TODO: use one allocation for inputs and pool buffer
    d0 = (int16_t*)_gna_malloc(8 * (UINT16_MAX + 1) * sizeof(int16_t));
    if (nullptr == d0)
    {
        throw GnaException(GNA_ERR_RESOURCES);
    }
    d1 = d0 + UINT16_MAX + 1;
    d2 = d1 + UINT16_MAX + 1;
    d3 = d2 + UINT16_MAX + 1;
    d4 = d3 + UINT16_MAX + 1;
    d5 = d4 + UINT16_MAX + 1;
    d6 = d5 + UINT16_MAX + 1;
    d7 = d6 + UINT16_MAX + 1;

    pool = (int64_t*)_kernel_malloc(CNN_POOL_SIZE_MAX * CNN_N_FLT_MAX * sizeof(int64_t));
    if (nullptr == pool)
    {
        this->~KernelBuffers();
        throw GnaException(GNA_ERR_RESOURCES);
    }
}

KernelBuffers::~KernelBuffers()
{
    if (nullptr != d0)
        _gna_free(d0);
    if (nullptr != pool)
        _gna_free(pool);
    memset(this, 0, sizeof(KernelBuffers));
}

ThreadPool::ThreadPool(uint8_t nThreads) :
    buffers{nThreads}
{
    Expect::InRange(nThreads, 1, 127, GNA_ERR_INVALID_THREAD_COUNT);
    for (uint8_t i = 0; i < nThreads; i++)
    {
        KernelBuffers* buff = &buffers.at(i);
        this->workers.emplace_back([&, buff]() {
            while (true)
            {
                std::unique_lock<std::mutex> lock(tpMutex);
                condition.wait(lock, [&]() { return stopped || !tasks.empty(); });
                if (stopped)
                {
                    return;
                }
                if (!tasks.empty())
                {
                    auto request_task = tasks.front();
                    tasks.pop_front();
                    lock.unlock();
                    request_task->operator()(buff);
                }
            }
        });
    }
}

ThreadPool::~ThreadPool()
{
#if defined(_WIN32)
    for (auto& w : workers)
    {
	w.detach();
    }
#else
    Stop();
#endif
}

void ThreadPool::Stop()
{
    {
        std::unique_lock<std::mutex> lock(tpMutex);
        stopped = true;
        condition.notify_all();
    }

    for (auto &worker : workers)
    {
        worker.join();
    }

    workers.clear();
}

void ThreadPool::Enqueue(Request *request)
{
    std::lock_guard<std::mutex> lock(tpMutex);
    tasks.emplace_back(request);
    condition.notify_one();
}

void ThreadPool::CancelTasks(const gna_model_id modelId)
{
    std::lock_guard<std::mutex> lock(tpMutex);
    for (auto it = tasks.begin(); it != tasks.end(); )
    {
        Request *request = *it;
        if (request->Configuration.Model.Id == modelId)
            it = tasks.erase(it);
        else ++it;
    }
}

