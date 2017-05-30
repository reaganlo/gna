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

#include "common.h"
#include "GnaException.h"

using std::condition_variable;
using std::function;
using std::future;
using std::make_shared;
using std::map;
using std::mutex;
using std::packaged_task;
using std::queue;
using std::thread;
using std::unique_lock;
using std::vector;

using namespace GNA;

#define _kernel_malloc(a) _aligned_malloc(a, INTRIN_ALIGN)


ThreadPool::ThreadPool() :
    stopped{true}
{}

ThreadPool::~ThreadPool()
{
    {
        unique_lock<mutex> lock(tp_mutex);
        if (stopped)
        {
            return;
        }
    }
    Stop();
}

void allocateFvBuffers(KernelBuffers * buffers)
{
    // TODO: use one allocation for inputs and pool buffer
    buffers->d0 = (int16_t*)_gna_malloc(8 * (UINT16_MAX + 1) * sizeof(int16_t));
    if (nullptr == buffers->d0)
    {
        throw GnaException(GNA_ERR_RESOURCES);
    }
    buffers->d1 = buffers->d0 + UINT16_MAX + 1;
    buffers->d2 = buffers->d1 + UINT16_MAX + 1;
    buffers->d3 = buffers->d2 + UINT16_MAX + 1;
    buffers->d4 = buffers->d3 + UINT16_MAX + 1;
    buffers->d5 = buffers->d4 + UINT16_MAX + 1;
    buffers->d6 = buffers->d5 + UINT16_MAX + 1;
    buffers->d7 = buffers->d6 + UINT16_MAX + 1;

    buffers->pool = (int64_t*)_kernel_malloc(CNN_POOL_SIZE_MAX * CNN_N_FLT_MAX * sizeof(int64_t));
    if (nullptr == buffers->pool)
    {
        throw GnaException(GNA_ERR_RESOURCES);
    }
}

void deallocateFvBuffers(KernelBuffers *buffers)
{
    if (nullptr != buffers->d0)
    {
        _gna_free(buffers->d0);
        buffers->d0 = nullptr;
    }
    if (nullptr != buffers->pool)
    {
        _gna_free(buffers->pool);
        buffers->pool = nullptr;
    }
}

void ThreadPool::Init(uint8_t n_threads) 
{
    {
        unique_lock<mutex> lock(tp_mutex);
        if (!stopped)
        {
            throw GnaException(GNA_ERR_QUEUE);
        }

        stopped = false;
    }
    for (uint8_t i = 0; i < n_threads; i++)
    {
        this->workers.emplace_back([&]() {
            thread_local KernelBuffers buffers;
            allocateFvBuffers(&buffers);
            while (true)
            {
                {
                    unique_lock<mutex> lock(tp_mutex);
                    condition.wait(lock, [&]() { return stopped || !tasks.empty(); });
                    if (stopped)
                    {
                        deallocateFvBuffers(&buffers);
                        return;
                    }
                    if (!tasks.empty()) {
                        auto& request_task = tasks.front();
                        tasks.pop();
                        (*request_task)(&buffers);
                    } 
                }
            }
        });
    }
}

void ThreadPool::Enqueue(Request *request)
{
    {
        unique_lock<mutex> lock(tp_mutex);
        if (stopped) 
        {
            throw GnaException(GNA_UNKNOWN_ERROR);
        }
        tasks.emplace(request);
    }
    condition.notify_one();
}

void ThreadPool::Stop() 
{
    {
        unique_lock<mutex> lock(tp_mutex);
        if (stopped)
        {
            throw GnaException(GNA_ERR_QUEUE);
        }
        stopped = true;
    }

    condition.notify_all();
    for (auto &worker : workers) 
    {
        worker.join();
    }
    // release resources if any left
    this->workers.erase(workers.begin(), workers.end());
}