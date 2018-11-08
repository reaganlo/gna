//*****************************************************************************
//
// INTEL CONFIDENTIAL
// Copyright 2018 Intel Corporation
//
// The source code contained or described herein and all documents related
// to the source code ("Material") are owned by Intel Corporation or its suppliers
// or licensors. Title to the Material remains with Intel Corporation or its suppliers
// and licensors. The Material contains trade secrets and proprietary
// and confidential information of Intel or its suppliers and licensors.
// The Material is protected by worldwide copyright and trade secret laws and treaty
// provisions. No part of the Material may be used, copied, reproduced, modified,
// published, uploaded, posted, transmitted, distributed, or disclosed in any way
// without Intel's prior express written permission.
//
// No license under any patent, copyright, trade secret or other intellectual
// property right is granted to or conferred upon you by disclosure or delivery
// of the Materials, either expressly, by implication, inducement, estoppel
// or otherwise. Any license under such intellectual property rights must
// be express and approved by Intel in writing.
//*****************************************************************************

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <malloc.h>

#include "gna-api.h"
#include "SelfTest.h"
#include "MultiOsHardwareSelfTest.h"

int main(int argc, char *argv[])
{
    PressEnterToContinue();
    LOG("Starting GNA device self-test\n");
    //getchar();
    MultiOsGnaSelfTestHardwareStatus hwDrvStatus;
    hwDrvStatus.Initialize();

    if (!hwDrvStatus.IsOK()) {
        hwDrvStatus.Print();
        GnaSelfTestIssue::HARDWARE_OR_DRIVER_NOT_INITIALIZED_PROPERLY.Handle();
    }
    else
        PressEnterToContinue();

    //open the default GNA device
    SelfTestDevice gnaDevice;

    LOG("Performing basic functionality test...\n");
    PressEnterToContinue();

    LOG("Sample model creation...\n");
    gnaDevice.SampleModelCreate();

    LOG("Request initialization...\n");
    gnaDevice.BuildSampleRequest();

    LOG("Request configuration...\n");
    gnaDevice.ConfigRequestBuffer();

    // calculate on GNA HW (blocking call)
    // wait for HW calculations (blocks until the results are ready)
    // after this call, outputs can be inspected under nnet.pLayers->pOutputs
    LOG("Sending request...\n");
    gnaDevice.RequestAndWait();

    LOG("Comparing results...\n");
    gnaDevice.CompareResults();
    LOG("GNA device self-test has beed finished\n");
    PressEnterToContinue();
    return 0;
}
