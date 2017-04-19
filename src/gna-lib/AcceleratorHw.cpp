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

#include <string>

#include "Validator.h"

#define PHDUMP

using std::string;

using namespace GNA;

status_t AcceleratorHw::Score(
    uint32_t layerIndex,
    uint32_t layerCount,
    const RequestConfiguration& requestConfiguration,
    RequestProfiler *profiler,
    KernelBuffers *buffers)
{
    UNREFERENCED_PARAMETER(buffers);

    void* data;
    size_t size;
    requestConfiguration.GetHwConfigData(data, size, layerIndex, layerCount);

    sender.Submit(data, size, profiler);

    auto response = reinterpret_cast<PGNA_CALC_IN>(data);
    auto status = response->status;
    Expect::True(GNA_SUCCESS == status || GNA_SSATURATE == status, status);

    return status;
}

/**
 * Empty virtual hw verification methods implemented in HW VERBOSE version only
 */
void AcceleratorHw::HwVerifier(Request*) {};
void AcceleratorHw::HwVerifier(SoftwareModel*, status_t) {};
bool AcceleratorHw::SetConfig(string, hw_calc_in_t*) { return true; };
bool AcceleratorHw::SetDescriptor(string, XNN_LYR*, hw_calc_in_t*) { return true; };
bool AcceleratorHw::SetRegister(string) { return true; };
