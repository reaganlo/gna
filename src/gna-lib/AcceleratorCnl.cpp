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

#include "AcceleratorCnl.h"

#include <future>

#include "AcceleratorManager.h"
#include "RequestManager.h"
#include "ThreadPool.h"

using std::future;
using namespace GNA;

AcceleratorCnl::AcceleratorCnl(acceleration swAccMode) 
{
    auto swAccPtr = AcceleratorManager::accelerators[swAccMode];
    if (swAccPtr)
        swAcc = (AcceleratorSw*)swAccPtr.get();
    else throw ApiException(GNA_CPUTYPENOTSUPPORTED);

    auto hwAccPtr = AcceleratorManager::accelerators[GNA_HW];
    if (hwAccPtr)
        hwAcc = (AcceleratorHw*)hwAccPtr.get();
    else throw ApiException(GNA_CPUTYPENOTSUPPORTED);

    // GNA_HW mode is what AcceleratorCnl should be viewed outside
    accMode = GNA_HW;
}

status_t AcceleratorCnl::submit(Request *r) 
{
    memset(&drv_res, 0, sizeof(perf_drv_t));
    memset(&hw_res, 0, sizeof(perf_hw_t));

    r->handle = new Sw();
    Sw* sw = r->handle;
    sw->handle = ThreadPool::Get().enqueue([=]() { return swingScore(r->model, &r->profiler); });

    return GNA_SUCCESS;
}

status_t AcceleratorCnl::swingScore(SoftwareModel *model, req_profiler *profiler) 
{
    status_t status = GNA_SUCCESS;
    
    uint32_t i, start;

    // origin values to retain immutability of model structure
    uint32_t nLayers = model->nLayers;
    nn_layer *layers = model->layers;
    BaseLayer **bLayers = model->bLayers;

    for (i = 0, start = 0; i < nLayers; i++) 
    {
        if (i != nLayers - 1)
        {
            if ((layers[i].nLayerKind == INTEL_CONVOLUTIONAL && layers[i + 1].nLayerKind == INTEL_CONVOLUTIONAL) 
                || (layers[i].nLayerKind != INTEL_CONVOLUTIONAL && layers[i + 1].nLayerKind != INTEL_CONVOLUTIONAL))
                continue;
        }

        uint32_t rewind = i - start + 1;
        model->nLayers = rewind;

        if (INTEL_CONVOLUTIONAL != model->layers->nLayerKind) 
        {
            // process non-cnn layers in hardware accelerator
            Hw* hw = new Hw(AcceleratorManager::buffer, hwAcc->hwInBuffSize);
            if (NULL == hw)
            {
                status = GNA_ERR_RESOURCES;
                ERR("FAILED with status %d\n", status);
                break;
            }

            hw->Fill(model);
            if (GNA_SUCCESS != status)
            {
                delete hw;
                ERR("FAILED with status %d\n", status);
                break;
            }

            hwAcc->SetRegister("setregister0.txt");
            hwAcc->SetDescriptor("setdescriptor.txt", hw->layersDesc, hw->inData);
            hwAcc->SetConfig("setconfig.txt", hw->inData);

            status = hwAcc->Submit(hw->inData, hw->dataSize, &profiler->ioctlSubmit, &hw->io_handle);
            if (GNA_SUCCESS != status)
            {
                delete hw;
                ERR("FAILED with status %d\n", status);
                break;
            }

            // accumulate hw profiler passed time
            profilerDTscAStart(&profiler->ioctlWaitOn);
            status = hwAcc->IoctlWait(&hw->io_handle, GNA_REQUEST_TIMEOUT_MAX);
            profilerDTscAStop(&profiler->ioctlWaitOn);

            // accumulate drv & hw performance results
            drv_res.intProc += hw->inData->drvPerf.intProc;
            drv_res.scoreHW += hw->inData->drvPerf.scoreHW;
            drv_res.startHW += hw->inData->drvPerf.startHW;
            hw_res.stall    += hw->inData->hwPerf.stall;
            hw_res.total    += hw->inData->hwPerf.total;

            delete hw;
        }
        // process cnn layers in software accelerator
        else 
        {
            Sw* sw = new Sw();
            if (NULL == sw)
            {
                status = GNA_ERR_RESOURCES;
                ERR("FAILED with status %d\n", status);
                break;
            }
            delete sw;
            status = swAcc->xnnSoftwareKernel(model, profiler);
        }

        // allow saturate warning
        ERRCHECKB(GNA_SUCCESS != status && GNA_SSATURATE != status, status);

        start = i + 1;
        model->layers += rewind;
        model->bLayers += rewind;
    }

    // restore model structure original state
    model->nLayers = nLayers;
    model->layers = layers;
    model->bLayers = bLayers;

    profilerDTscAStop(&profiler->total);
    return status;
}

status_t AcceleratorCnl::Wait(Request *r, uint32_t timeout, perf_t* perfResults)
{
    status_t    status = GNA_SUCCESS;
    status_t    status2 = GNA_SUCCESS;
    Sw*         sw = (Sw*)r->handle;

    // wait for threaded request r to finish with timeout
    auto future_status = sw->handle.wait_for(std::chrono::milliseconds(timeout));
    if (future_status == std::future_status::deferred || future_status == std::future_status::timeout)
    {
        status = GNA_DEVICEBUSY;
    }
    else if (future_status == std::future_status::ready)
    {
        status = sw->handle.get();
        // finish profiling
        profilerDTscAStop(&r->profiler.process);

        // save profiling results to app provided space
#ifdef PROFILE
        if (NULL != perfResults)
        {
            perfResults->lib.preprocess = r->profiler.preprocess.passed;
            perfResults->lib.process = r->profiler.process.passed;
            perfResults->lib.submit = r->profiler.submit.passed;
            perfResults->lib.scoring = r->profiler.scoring.passed;
            perfResults->lib.total = r->profiler.total.passed;

            perfResults->total.start = r->profiler.submit.start;
            perfResults->total.stop = r->profiler.process.stop;
            memcpy_s(&perfResults->drv, sizeof(perf_drv_t), &drv_res, sizeof(perf_drv_t));
            memcpy_s(&perfResults->hw, sizeof(perf_hw_t), &hw_res, sizeof(perf_hw_t));

            perfResults->lib.ioctlSubmit = r->profiler.ioctlSubmit.passed;
            perfResults->lib.ioctlWaitOn = r->profiler.ioctlWaitOn.passed;
        }
#endif // PROFILE

        // remove request from queue
        status2 = RequestManager::removeRequest(r);
        if (GNA_SUCCESS != status2) status = status2; // notify if dequeue error occurred
    }

    return status;
}

status_t AcceleratorCnl::init() 
{
    return GNA_SUCCESS;
}

// it can do everything and more
uint32_t AcceleratorCnl::getCapabilities() 
{
    return 0;
}
