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

#include "AcceleratorHw.h"

#define PHDUMP

#include <string>

using std::string;
using namespace GNA;

status_t AcceleratorHw::Score(
    const CompiledModel& model,
    const RequestConfiguration& requestConfiguration,
          RequestProfiler *profiler,
          KernelBuffers *buffers)
{
    UNREFERENCED_PARAMETER(buffers);
    auto status = GNA_SUCCESS;

    io_handle_t ioHandle;
    memset(&ioHandle, 0, sizeof(ioHandle));
    ioHandle.hEvent = CreateEvent(nullptr, false, false, nullptr);

    GNA_CALC_IN calculationData;
    auto found = std::find_if(
        requestConfiguration.LayerConfigurations.cbegin(),
        requestConfiguration.LayerConfigurations.cend(),
        [](const auto& iter) { return nullptr != iter.second->ActiveList; });

    calculationData.ctrlFlags.activeListOn = found != requestConfiguration.LayerConfigurations.cend();
    calculationData.ctrlFlags.gnaMode = 1; // xnn by default
    calculationData.ctrlFlags.layerCount = model.GetLayerCount(); // xnn by default
    calculationData.ctrlFlags.layerNo = 0;
    calculationData.modelId = model.GetModelId();

    status = Submit(&calculationData, sizeof(calculationData), &profiler->ioctlSubmit, &ioHandle);
    ERRCHECKR(GNA_SUCCESS != status, status);

    profilerDTscStart(&profiler->ioctlWaitOn);
    status = IoctlWait(&ioHandle, GNA_REQUEST_TIMEOUT_MAX);
    ERRCHECKR(GNA_SUCCESS != status, status);
    profilerDTscStop(&profiler->ioctlWaitOn);

    return GNA_SUCCESS;
}

status_t AcceleratorHw::Score(
    const CompiledModel& model,
    const SubModel& submodel,
    const RequestConfiguration& requestConfiguration,
          RequestProfiler *profiler,
          KernelBuffers *buffers)
{
    return GNA_SUCCESS;
}
//status_t AcceleratorHw::Score(Hw *hw, RequestProfiler* p)
//{
//    // TODO: with already compiled model, this won't be necessary
//    //r->handle = (Sw*)new Hw(Device::buffer, hwInBuffSize);
//    // TODO: use GnaException
//    //
//    //hw = (Hw*)r->handle;
//    //if (r->xnn)
//    //{
//    //    status = hw->Fill(r->xnn);
//    //}
//    //else
//    //{
//    //    status = hw->Fill(r->gmm);
//    //}
//    // TODO: use GnaException
//
//    SetRegister("setregister0.txt");
//    SetDescriptor("setdescriptor.txt", hw->xnnLayerDescriptors, hw->inData);
//    SetConfig("setconfig.txt", hw->inData);
//    status_t status = Submit(hw->inData, hw->dataSize, &p->ioctlSubmit, &hw->io_handle);
//    // TODO: use GnaException
//
//    // wait for
//    profilerDTscStart(&p->ioctlWaitOn);
//    status = IoctlWait(&hw->io_handle, GNA_REQUEST_TIMEOUT_MAX);
//    profilerDTscStop(&p->ioctlWaitOn);
//
//    profilerDTscAStop(&p->total);
//    return status;
//}

//status_t AcceleratorHw::Wait(Request *r, uint32_t timeout, perf_t* perfResults)
//{
//    auto    status  = GNA_SUCCESS;
//    //status_t    status2 = GNA_SUCCESS;
//    // FIXME: do something about the handle
//    Hw* hw = nullptr; // (Hw*)r->handle;
//
//    // get submit status
//    auto future_status = hw->handle.wait_for(std::chrono::milliseconds(timeout));
//    if (future_status == std::future_status::deferred || future_status == std::future_status::timeout)
//    {
//        status = GNA_DEVICEBUSY;
//    }
//    else if (future_status == std::future_status::ready)
//    {
//        status = hw->handle.get();
//    }
//
//    if(GNA_SUCCESS == status)
//    {
//        // finish profiling
//        profilerDTscStop(&r->profiler.process);
//        // save profiling results to app provided space
//#ifdef PROFILE
//        if (nullptr != perfResults)
//        {
//            memcpy_s(&perfResults->drv, sizeof(perf_drv_t), &hw->inData->drvPerf, sizeof(perf_drv_t));
//            memcpy_s(&perfResults->hw, sizeof(perf_hw_t), &hw->inData->hwPerf, sizeof(perf_hw_t));
//            perfResults->lib.preprocess  = r->profiler.preprocess.passed;
//            perfResults->lib.process     = r->profiler.process.passed;
//            perfResults->lib.submit      = r->profiler.submit.passed;
//            perfResults->lib.total       = r->profiler.total.passed;
//            perfResults->lib.ioctlSubmit = r->profiler.ioctlSubmit.passed;
//            perfResults->lib.ioctlWaitOn = r->profiler.ioctlWaitOn.passed;
//            perfResults->total.start     = r->profiler.submit.start;
//            perfResults->total.stop      = r->profiler.process.stop;
//        }
//#endif // PROFILE
//
//        //HwVerifier(r);
//        // return scoring status
//        status = hw->inData->status;
//        // remove request from queue
//        //status2 = RequestHandler::RemoveRequest(r->id);
//        //if(GNA_SUCCESS != status2) status = status2; // notify if dequeue error occurred
//    }
//    else if(GNA_DEVICEBUSY != status) // GNA_IOCTLRESERR
//    {
//        return status;
//        //RequestHandler::RemoveRequest(r->id);          // ignore error, as already in error state
//    }
//
//    return status;
//}

/**
 * Empty virtual hw verification methods implemented in HW VERBOSE version only
 */
void AcceleratorHw::HwVerifier(Request* r){};
void AcceleratorHw::HwVerifier(SoftwareModel *model, status_t scoring_status){};
bool AcceleratorHw::SetConfig(string path, hw_calc_in_t* inData){ return true; };
bool AcceleratorHw::SetDescriptor(string path, XNN_LYR* buff, hw_calc_in_t* inData){ return true; };
bool AcceleratorHw::SetRegister(string path){ return true; };