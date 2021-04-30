/*
 INTEL CONFIDENTIAL
 Copyright 2019 Intel Corporation.

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

#include "ConvolutionalLayer2DCapabilities.h"
#include "Expect.h"
#include "GnaException.h"
#include "Memory.h"
#include "Request.h"
#include "KernelArguments.h"

#include <cstring>
#include <cstdint>

using namespace GNA;
using CnnCaps = GNA::ConvolutionalLayer2DCapabilities;

// will set memory only in DEBUG configuration
template<typename M, typename S>
static void clearMemoryInDebug(M* memory, S size)
{
#if DEBUG == 1
    memset(memory, 0, size);
#else
    ((void)(memory));
    ((void)(size));
#endif
}

KernelBuffers::KernelBuffers()
{
    // TODO: use one allocation for inputs and pool buffer
    auto const size = 8 * (UINT16_MAX + 1) * sizeof(int16_t);
    d0 = static_cast<int16_t*>(_gna_malloc(size));
    if (nullptr == d0)
    {
        throw GnaException(Gna2StatusResourceAllocationError);
    }
    clearMemoryInDebug(d0, size);
    d1 = d0 + UINT16_MAX + 1;
    d2 = d1 + UINT16_MAX + 1;
    d3 = d2 + UINT16_MAX + 1;
    d4 = d3 + UINT16_MAX + 1;
    d5 = d4 + UINT16_MAX + 1;
    d6 = d5 + UINT16_MAX + 1;
    d7 = d6 + UINT16_MAX + 1;

    auto const poolSize = CnnCaps::PoolingWindowSizeMax * CnnCaps::Filter1DCountMax * sizeof(int64_t);
    pool = static_cast<int64_t *>(_kernel_malloc(poolSize));
    if (nullptr == pool)
    {
        this->~KernelBuffers();
        throw GnaException(Gna2StatusResourceAllocationError);
    }
    clearMemoryInDebug(pool, poolSize);

    auto const cnnScratchSize = 1 * 1024 * 1024;
    ReallocateCnnScratchPad(cnnScratchSize);
}

KernelBuffers::~KernelBuffers()
{
    if (nullptr != d0)
    {
        _gna_free(d0);
    }
    if (nullptr != pool)
    {
        _gna_free(pool);
    }
    if (nullptr != cnnFusedBuffer)
    {
        _gna_free(cnnFusedBuffer);
    }
    memset(this, 0, sizeof(*this));
}

void KernelBuffers::ReallocateCnnScratchPad(uint32_t cnnScratchSize)
{
    if(cnnScratchSize > cnnFusedBufferSize)
    {
        if (nullptr != cnnFusedBuffer)
        {
            _gna_free(cnnFusedBuffer);
            cnnFusedBuffer = nullptr;
            cnnFusedBufferSize = 0;
        }
        cnnFusedBuffer = static_cast<int8_t*>(_kernel_malloc(cnnScratchSize));
        if (nullptr == cnnFusedBuffer)
        {
            throw GnaException(Gna2StatusResourceAllocationError);
        }
        cnnFusedBufferSize = cnnScratchSize;

        clearMemoryInDebug(cnnFusedBuffer, cnnScratchSize);
    }
}

ThreadPool::ThreadPool(uint32_t threadCount) :
    buffers{ threadCount },
    numberOfThreads{ threadCount }
{
    Expect::InRange(threadCount, 1U, 127U, Gna2StatusDeviceNumberOfThreadsInvalid);
    employWorkers();
}

ThreadPool::~ThreadPool()
{
    StopAndJoin();
}

uint32_t ThreadPool::GetNumberOfThreads() const
{
    return numberOfThreads;
}

void ThreadPool::SetNumberOfThreads(uint32_t threadCount)
{
    Expect::InRange(threadCount, 1U, 127U, Gna2StatusDeviceNumberOfThreadsInvalid);

    if (threadCount == numberOfThreads)
    {
        return;
    }

    StopAndJoin();

    try
    {
        buffers.resize(threadCount);
    }
    catch (std::exception& e)
    {
        UNREFERENCED_PARAMETER(e);
        throw GnaException(Gna2StatusResourceAllocationError);
    }

    numberOfThreads = threadCount;
    employWorkers();
}



void ThreadPool::Enqueue(Request *request)
{
    std::lock_guard<std::mutex> lock(tpMutex);
    tasks.emplace_back(request);
    condition.notify_one();
}

void ThreadPool::StopAndJoin()
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

void ThreadPool::employWorkers()
{
    stopped = false;
    for (uint32_t i = 0; i < numberOfThreads; i++)
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
                    request_task->operator()(buff);
                }
            }
        });
    }
}
