//*****************************************************************************
//
// INTEL CONFIDENTIAL
// Copyright 2018-2020 Intel Corporation
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

#include "SelfTest.h"
#include "MultiOsHardwareSelfTest.h"

#include "gna2-capability-api.h"

#include <cstdlib>
#include <cstring>

GnaSelfTestLogger logger;

int main(int argc, char *argv[])
try
{
    MultiOs::Init();
    auto config = GnaSelfTestConfig::ReadConfigFromCmdLine(argc, argv);
    logger.SetVerbose(config.VerboseMode());
    GnaSelfTest gnaSelfTest{ config };
    gnaSelfTest.StartTest();
    return 0;
}
catch (const GnaSelfTestException& e)
{
    return e.GetExitCode();
}
catch (...)
{
    GnaSelfTestLogger::Error("Unknown exception in main()\n");
    return GSTIT_UNHANDLED_EXCEPTION;
}

void GnaSelfTest::StartTest()
{
    GnaSelfTestLogger::Log("Starting GNA device self test\n");
    GnaSelfTestLogger::Log("=============================\n");
    PrintLibraryVersion();
    PrintSystemInfo();
    MultiOsGnaSelfTestHardwareStatus hwDrvStatus{ *this };
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

    for (int iteration = 0; iteration < config.GetRepeatCount(); iteration++)
    {
        DoIteration();
    }
}

void GnaSelfTest::DoIteration()
{
    SampleModelForGnaSelfTest sampleNetwork{ config };

    //open the default GNA device
    SelfTestDevice gnaDevice(*this);

    GnaSelfTestLogger::Log("Performing basic functionality test...\n");

    logger.Verbose("Sample model creation...\n");
    gnaDevice.SampleModelCreate(sampleNetwork);

    logger.Verbose("Request initialization...\n");
    gnaDevice.BuildSampleRequest();

    logger.Verbose("Request configuration...\n");
    gnaDevice.ConfigRequestBuffer();


    for (int i = 0; i < config.GetRequestRepeatCount();i++)
    {
        // calculate on GNA HW (blocking call)
        // wait for HW calculations (blocks until the results are ready)
        // after this call, outputs can be inspected under nnet.pLayers->pOutputs
        logger.Verbose("Sending request...\n");
        gnaDevice.RequestAndWait(config.GetRequestWaitMs());

        logger.Verbose("Comparing results...\n");
        gnaDevice.CompareResults(sampleNetwork);
    }
    GnaSelfTestLogger::Log("GNA device self-test has been finished\n");
}

std::string GnaSelfTest::GetBuildTimeLibraryVersionString()
{
#ifdef GNA_LIBRARY_VERSION_STRING
    static const char versionString[] = GNA_LIBRARY_VERSION_STRING;
#else
    static const char versionString[] = "Unknown build time version";
#endif
    return versionString;
}

void GnaSelfTest::PrintLibraryVersion()
{
    char buffer[32];
    auto const status = Gna2GetLibraryVersion(buffer, 32);
    if(Gna2StatusIsSuccessful(status))
    {
        logger.Log("Detected GNA Library version: %s\n", buffer);
    }
    else
    {
        logger.Error("GNA Library version: UNKNOWN\n");
    }
    logger.Log("Build time GNA library version: %s\n", GetBuildTimeLibraryVersionString().c_str());
}

int GnaSelfTestConfig::GetRequestRepeatCount()
{
    return requestRepeatCount;
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
        else if (strncmp("-r", argv[i], 2) == 0)
        {
            if (i + 1 < argc)
            {
                repeatCount = atoi(argv[i + 1]);
                i++;
            }
        }
        else if (strncmp("-e", argv[i], 2) == 0)
        {
            if (i + 1 < argc)
            {
                requestRepeatCount = atoi(argv[i + 1]);
                i++;
            }
        }
        else if (strncmp("-w", argv[i], 2) == 0)
        {
            if (i + 1 < argc)
            {
                requestWaitMs = static_cast<uint32_t>(atoi(argv[i + 1]));
                i++;
            }
        }
        else if (strncmp("-m", argv[i], 2) == 0)
        {
            if (i + 4 < argc)
            {
                defaultFC = false;
                modelCustomFCNumberOfOperations = static_cast<uint32_t>(atoi(argv[i + 1]));
                modelCustomFCInputSize = static_cast<uint32_t>(atoi(argv[i + 2]));
                modelCustomFCGrouping = static_cast<uint32_t>(atoi(argv[i + 3]));
                modelCustomFCOutputSize = static_cast<uint32_t>(atoi(argv[i + 4]));
                i += 4;
            }
        }
        else
        {
            GnaSelfTestLogger::Log("gna-self-test [-v] [-c] [-p] [-e K] [-r N ] [-m L I G O] [-w M] [-h]\n");
            GnaSelfTestLogger::Log("   -v          verbose mode\n");
            GnaSelfTestLogger::Log("   -c          ignore errors and continue execution\n");
            GnaSelfTestLogger::Log("   -p          after the error, ask the user whether to continue\n");
            GnaSelfTestLogger::Log("   -e K        send request K times\n");
            GnaSelfTestLogger::Log("   -r N        repeat the flow N times\n");
            GnaSelfTestLogger::Log("   -m L I G O  use model of L FC layers with I inputs G groups and O outputs\n");
            GnaSelfTestLogger::Log("   -w M        set M milliseconds as wait timeout\n");
            GnaSelfTestLogger::Log("   -h          display help\n");
            throw GnaSelfTestIssue{ GSTIT_PRINT_HELP_ONLY };
        }
    }
}

GnaSelfTestConfig GnaSelfTestConfig::ReadConfigFromCmdLine(int argc, const char *const argv[])
{
    return GnaSelfTestConfig(argc, argv);
}
