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

#include "common.h"
#include "GnaException.h"
#include "pwl-types.h"
#include "ThreadPool.h"

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

void allocateFvBuffers(aligned_fv_bufs * buffers)
{
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

    buffers->lookup = _gna_malloc(PWL_LOOKUP_SIZE);
    if (nullptr == buffers->lookup)
    {
        throw GnaException(GNA_ERR_RESOURCES);
    }

    buffers->xBase = _gna_malloc(PWL_X_BUFFER_SIZE + PWL_Y_BUFFER_SIZE);
    if (nullptr == buffers->xBase)
    {
        throw GnaException(GNA_ERR_RESOURCES);
    }

    buffers->ySeg = ((int8_t*)(buffers->xBase) + PWL_X_BUFFER_SIZE);

    buffers->pwl = _kernel_malloc(PWL_PARAMS_BUFFER_SIZE);
    if (nullptr == buffers->pwl)
    {
        throw GnaException(GNA_ERR_RESOURCES);
    }

    memset(buffers->pwl,    0,      PWL_PARAMS_BUFFER_SIZE);
    memset(buffers->lookup, 0xff,   PWL_LOOKUP_SIZE);
    memset(buffers->xBase,  0,      PWL_X_BUFFER_SIZE);
    memset(buffers->ySeg,   0,      PWL_Y_BUFFER_SIZE);

    ((pwl_params*)buffers->pwl)->lookup = (pwl_u_t*)buffers->lookup;
    ((pwl_params*)buffers->pwl)->xBase  = (pwl_x_t*)buffers->xBase;
    ((pwl_params*)buffers->pwl)->ySeg   = (pwl_y_t*)buffers->ySeg;
}

void deallocateFvBuffers(aligned_fv_bufs *buffers)
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
    if (nullptr != buffers->pwl)
    {
        _gna_free(buffers->pwl);
        buffers->pwl = nullptr;
    }
    if (nullptr != buffers->lookup)
    {
        _gna_free(buffers->lookup);
        buffers->lookup = nullptr;
    }
    if (nullptr != buffers->xBase)
    {
        _gna_free(buffers->xBase);
        buffers->xBase = nullptr;
    }
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
            thread_local aligned_fv_bufs buffers;
            allocateFvBuffers(&buffers);
            while (true)
            {
                {
                    unique_lock<mutex> lock(tp_mutex);
                    condition.wait(lock, [&]() { return stopped || !tasks.empty(); });
                    if (true == stopped)
                    {
                        deallocateFvBuffers(&buffers);
                        return;
                    }
                    if (false == tasks.empty()) {
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
    for (auto &worker : workers) 
    {
        worker.join();
    }
    // release resources if any left
    this->workers.erase(workers.begin(), workers.end());
}