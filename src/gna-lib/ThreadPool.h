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

#include <condition_variable>
#include <functional>
#include <future>
#include <map>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "common.h"
#include "Request.h"

namespace GNA
{
class ThreadPool {
public:

    /**
    * Creates worker threads
    * Fills thread map based on thread ids
    * 
    * @n_threads - number of threads to spawn
    */
    void Init(uint8_t n_threads);

    /**
     * Stops thread pool and joins worker threads
     */
    void Stop();

    /**
     * Adds request (as Accelerator function call) to the queue
     * and notifies one thread
     *
     * @requestFn   request function call wrapper
     * @return      future 
     */
    void Enqueue(Request *request);

    /**
     * Constructor
     */
    ThreadPool();

    /**
     * Destructor, calls stop method
     */
    ~ThreadPool();

private:

    std::vector<std::thread> workers;

    // calculation function queue
    std::queue<Request*> tasks;

    std::mutex tp_mutex;

    std::condition_variable condition;

    bool stopped;

    /**
     * Deleted functions to prevent from being defined or called
     * @see: https://msdn.microsoft.com/en-us/library/dn457344.aspx
     */
    ThreadPool(const ThreadPool &) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
};

}
