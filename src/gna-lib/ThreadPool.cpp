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
#include "pwl-types.h"
#include "GnaException.h"

using std::unique_lock;
using std::packaged_task;
using std::make_shared;
using std::condition_variable;
using std::function;
using std::future;
using std::map;
using std::mutex;
using std::thread;
using std::vector;
using std::queue;

using namespace GNA;

ThreadPool::ThreadPool() : stopped(true) {}

ThreadPool::~ThreadPool()
{
    {
        unique_lock<mutex> lock(tp_mutex);
        if (true == stopped)
        {
            return;
        }
    }
    Stop();
}


void ThreadPool::Init(uint8_t n_threads) 
{
    {
        unique_lock<mutex> lock(tp_mutex);
        if (false == stopped)
        {
            throw GnaException(GNA_ERR_QUEUE);
        }

        stopped = false;
    }
    for (uint8_t i = 0; i < n_threads; i++)
    {
        this->workers.emplace_back([&]() {
            while (true) 
            {
                {
                    unique_lock<mutex> lock(tp_mutex);
                    condition.wait(lock, [&]() { return stopped || !tasks.empty(); });
                    if (true == stopped) return;
                    if (false == tasks.empty()) {
                        auto& request_task = tasks.front();
                        tasks.pop();
                        (*request_task)();
                    } 
                }
            }
        });
        threadMap[workers.at(i).get_id()] = i;
    }
}

void ThreadPool::Enqueue(Request *request)
{
    {
        unique_lock<mutex> lock(tp_mutex);
        if (true == stopped) 
        {
            throw GnaException(GNA_ERR_UNKNOWN);
        }
        tasks.emplace(request);
    }
    condition.notify_one();
}

void ThreadPool::Stop() 
{
    {
        unique_lock<mutex> lock(tp_mutex);
        if (true == stopped)
        {
            throw GnaException(GNA_ERR_QUEUE);
        }
        stopped = true;
    }

    condition.notify_all();
    for (thread &worker : workers) 
    {
        worker.join();
    }
    // release resources if any left
    this->workers.erase(workers.begin(), workers.end());
    this->threadMap.erase(threadMap.begin(), threadMap.end());
    for (size_t i = 0; i < tasks.size(); i++)
    {
        tasks.pop();
    }

    auto iterator = threadBuffers.begin();
    for (; iterator != threadBuffers.end(); iterator++)
    {
        aligned_fv_bufs buffers = iterator->second;
        if (nullptr != buffers.d0)
        {
            _gna_free(buffers.d0);
            buffers.d0 = nullptr;
        }
        if (nullptr != buffers.pool)
        {
            _gna_free(buffers.pool);
            buffers.pool = nullptr;
        }
        if (nullptr != buffers.pwl)
        {
            _gna_free(buffers.pwl);
            buffers.pwl = nullptr;
        }
        if (nullptr != buffers.lookup)
        {
            _gna_free(buffers.lookup);
            buffers.lookup = nullptr;
        }
        if (nullptr != buffers.xBase)
        {
            _gna_free(buffers.xBase);
            buffers.xBase = nullptr;
        }
    }
    threadBuffers.clear();
}

uint8_t ThreadPool::GetThreadNumber()
{
    thread::id threadId = std::this_thread::get_id();
    uint8_t threadNumber = threadMap[threadId];
    return threadNumber;
}

aligned_fv_bufs* ThreadPool::GetThreadBuffer()
{
    uint8_t threadNumber = GetThreadNumber();
    return &threadBuffers[threadNumber];
}

status_t ThreadPool::AllocThreadBuffers(uint32_t n_threads)
{
    for (uint8_t i = 0; i < n_threads; i++)
    {
        threadBuffers[i].d0 = (int16_t*)_gna_malloc(8 * (UINT16_MAX + 1) * sizeof(int16_t));
        if (nullptr == threadBuffers[i].d0)
        {
            return GNA_ERR_RESOURCES;
        }
        threadBuffers[i].d1 = threadBuffers[i].d0 + UINT16_MAX + 1;
        threadBuffers[i].d2 = threadBuffers[i].d1 + UINT16_MAX + 1;
        threadBuffers[i].d3 = threadBuffers[i].d2 + UINT16_MAX + 1;
        threadBuffers[i].d4 = threadBuffers[i].d3 + UINT16_MAX + 1;
        threadBuffers[i].d5 = threadBuffers[i].d4 + UINT16_MAX + 1;
        threadBuffers[i].d6 = threadBuffers[i].d5 + UINT16_MAX + 1;
        threadBuffers[i].d7 = threadBuffers[i].d6 + UINT16_MAX + 1;

        threadBuffers[i].pool = (int64_t*)_kernel_malloc(CNN_POOL_SIZE_MAX * CNN_N_FLT_MAX * sizeof(int64_t));
        if (nullptr == threadBuffers[i].pool)
        {
            return GNA_ERR_RESOURCES;
        }

        threadBuffers[i].lookup = _gna_malloc(PWL_LOOKUP_SIZE);
        if (nullptr == threadBuffers[i].lookup)
        {
            return GNA_ERR_RESOURCES;
        }

        threadBuffers[i].xBase = _gna_malloc(PWL_X_BUFFER_SIZE + PWL_Y_BUFFER_SIZE);
        if (nullptr == threadBuffers[i].xBase)
        {
            return GNA_ERR_RESOURCES;
        }

        threadBuffers[i].ySeg = ((int8_t*)(threadBuffers[i].xBase) + PWL_X_BUFFER_SIZE);

        threadBuffers[i].pwl = _kernel_malloc(PWL_PARAMS_BUFFER_SIZE);
        if (nullptr == threadBuffers[i].pwl)
        {
            return GNA_ERR_RESOURCES;
        }

        memset(threadBuffers[i].pwl,    0,      PWL_PARAMS_BUFFER_SIZE);
        memset(threadBuffers[i].lookup, 0xff,   PWL_LOOKUP_SIZE);
        memset(threadBuffers[i].xBase,  0,      PWL_X_BUFFER_SIZE);
        memset(threadBuffers[i].ySeg,   0,      PWL_Y_BUFFER_SIZE);

        ((pwl_params*)threadBuffers[i].pwl)->lookup = (pwl_u_t*)threadBuffers[i].lookup;
        ((pwl_params*)threadBuffers[i].pwl)->xBase  = (pwl_x_t*)threadBuffers[i].xBase;
        ((pwl_params*)threadBuffers[i].pwl)->ySeg   = (pwl_y_t*)threadBuffers[i].ySeg;
    }

    return GNA_SUCCESS;
}