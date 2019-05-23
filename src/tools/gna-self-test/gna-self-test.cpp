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

GnaSelfTestLogger logger;

int main(int argc, char *argv[])
{
    auto config = GnaSelfTestConfig::ReadConfigFromCmdLine(argc, argv);
    logger.SetVerbose(config.VerboseMode());
    GnaSelfTest gnaSelfTest{ config };
    gnaSelfTest.StartTest();
    return 0;
}

void GnaSelfTest::StartTest()
{
    GnaSelfTestLogger::Log("Starting GNA device self test\n");
    GnaSelfTestLogger::Log("=============================\n");
    PrintSystemInfo();
    SampleModelForGnaSelfTest sampleNetwork = SampleModelForGnaSelfTest::GetDefault();
    MultiOsGnaSelfTestHardwareStatus hwDrvStatus{*this};
    hwDrvStatus.Initialize();

    if (!hwDrvStatus.IsOK())
    {
        hwDrvStatus.Print();
        Handle(GSTIT_HARDWARE_OR_DRIVER_NOT_INITIALIZED_PROPERLY);
    }
    else
    {
        GnaSelfTestLogger::Log("GNA device and driver are OK\n");
        GnaSelfTestLogger::Log("============================\n");
    }

    //open the default GNA device
    SelfTestDevice gnaDevice(*this);

    GnaSelfTestLogger::Log("Performing basic functionality test...\n");

    logger.Verbose("Sample model creation...\n");
    gnaDevice.SampleModelCreate(sampleNetwork);

    logger.Verbose("Request initialization...\n");
    gnaDevice.BuildSampleRequest();

    logger.Verbose("Request configuration...\n");
    gnaDevice.ConfigRequestBuffer();

    // calculate on GNA HW (blocking call)
    // wait for HW calculations (blocks until the results are ready)
    // after this call, outputs can be inspected under nnet.pLayers->pOutputs
    logger.Verbose("Sending request...\n");
    gnaDevice.RequestAndWait();

    logger.Verbose("Comparing results...\n");
    gnaDevice.CompareResults(sampleNetwork);
    GnaSelfTestLogger::Log("GNA device self-test has beed finished\n");
}

GnaSelfTestConfig::GnaSelfTestConfig(int argc, const char *const argv[])
{
    for (int i = 1; i < argc; i++)
    {
        if (strncmp("-v", argv[i], 2) == 0)
        {
            verboseMode = true;
        }
        else if (strncmp("-c", argv[i], 2) == 0)
        {
            continueAfterError = true;
        }
        else if (strncmp("-p", argv[i], 2) == 0)
        {
            continueAfterError = true;
            pauseAfterError = true;
        }
        else
        {
            GnaSelfTestLogger::Log("gna-self-test [-v] [-c] [-p] [-h]\n");
            GnaSelfTestLogger::Log("   -v   verbose mode\n");
            GnaSelfTestLogger::Log("   -c   ignore errors and continue execution\n");
            GnaSelfTestLogger::Log("   -p   after the error, ask the user whether to continue\n");
            GnaSelfTestLogger::Log("   -h   display help\n");
            exit(0);
        }
    }
}
GnaSelfTestConfig GnaSelfTestConfig::ReadConfigFromCmdLine(int argc, const char *const argv[])
{
    return GnaSelfTestConfig(argc, argv);
}
