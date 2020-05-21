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
#pragma once

#include "SampleModelForGnaSelfTest.h"

#include "gna2-common-api.h"
#include "gna2-model-api.h"

#include <cstdio>
#include <exception>
#include <string>
#include <vector>

class GnaSelfTestLogger
{
public:
    //Logs to stdout only if the verbose mode is enabled
    template <typename... Args>
    void Verbose(const char* fmt, Args... args) const
    {
        if (verboseLogging)
            Log(fmt, args...);
    }
    //Logs to stdout
    template <typename... Args>
    static void Log(const char* fmt, Args... args)
    {
        logToFile(stdout, fmt, args...);
    }
    //Logs to stderr
    template <typename... Args>
    static void Error(const char* fmt, Args... args)
    {
        logToFile(stderr, fmt, args...);
    }
    //Enables or disables verbose mode
    void SetVerbose(bool verbose) { verboseLogging = verbose; }
private:
    bool verboseLogging = false;
    static void logToFile(FILE* out, const char* txt)
    {
        logToFile(out, "%s", txt);
    }
    template <typename Arg, typename... Args>
    static void logToFile(FILE* out, const char* fmt, Arg arg, Args... args)
    {
        fputs("[GNA-SELF-TEST] ", out);
        fprintf(out, fmt, arg, args...);
    }
};
extern GnaSelfTestLogger logger;

class GnaSelfTestConfig : public ModelConfig
{
public:
    static GnaSelfTestConfig ReadConfigFromCmdLine(int argc, const char *const argv[]);
    bool VerboseMode() const { return verboseMode; }
    bool PauseMode() const { return pauseAfterError; }
    bool ContinueMode() const { return continueAfterError; }
    int GetRepeatCount() const { return repeatCount; }
    uint32_t GetRequestWaitMs() const { return  requestWaitMs; }
    int GetRequestRepeatCount();
private:
    GnaSelfTestConfig(int argc, const char * const argv[]);
    bool verboseMode = false;
    bool continueAfterError = false;
    bool pauseAfterError = false;
    int repeatCount = 1;
    int requestRepeatCount = 1;
    uint32_t requestWaitMs = 10000;
};

class GnaSelfTestIssue;
class GnaSelfTest {
public:
    GnaSelfTest(GnaSelfTestConfig configIn) :config{configIn}{}
    void StartTest();
    void Handle(const GnaSelfTestIssue& issue) const;
    void HandleGnaStatus(const Gna2Status &status, const char *what) const;
private:
    static bool userWantsGoFurther();
    GnaSelfTestConfig config;
    static std::string GetBuildTimeLibraryVersionString();
    static void PrintLibraryVersion();
    void DoIteration();
};

#define DEFAULT_SELFTEST_TIMEOUT_MS 1000

class SelfTestDevice
{
public:
    SelfTestDevice(const GnaSelfTest& gst);
    // obtains pinned memory shared with the device
    uint8_t * Alloc(const uint32_t bytesRequested);

    void SampleModelCreate(SampleModelForGnaSelfTest &model);

    void BuildSampleRequest();
    void ConfigRequestBuffer();

    void RequestAndWait(uint32_t requestWaitMs);

    void CompareResults(const SampleModelForGnaSelfTest &model);
    ~SelfTestDevice();
private:
    uint32_t deviceId = 0;
    bool deviceOpened = false;
    Gna2DeviceVersion deviceVersion = Gna2DeviceVersionSoftwareEmulation;
    uint32_t sampleModelId;
    uint32_t configId;
    Gna2Model gnaModel; // main neural network container
    int16_t *pinned_outputs;
    int16_t *pinned_inputs;
    uint32_t requestId;
    const GnaSelfTest& gnaSelfTest;
    std::vector<void*> allocated_mems;
};

// Types and exit codes of GnaSelfTestIssue
enum GSTIT
{
    GSTIT_GENERAL_GNA_NO_SUCCESS = 1,
    GSTIT_DRV_1_INSTEAD_2 = 2,
    GSTIT_NO_HARDWARE = 3,
    GSTIT_NUL_DRIVER = 4,
    GSTIT_DEVICE_OPEN_NO_SUCCESS = 5,
    GSTIT_HARDWARE_OR_DRIVER_NOT_INITIALIZED_PROPERLY = 6,
    GSTIT_GNAALLOC_MEM_ALLOC_FAILED = 7,
    GSTIT_UNKNOWN_DRIVER = 8,
    GSTIT_MALLOC_FAILED = 9,
    GSTIT_SETUPDI_ERROR = 10,
    GSTIT_NO_DRIVER = 11,
    GSTIT_ERRORS_IN_SCORES = 12,
    GSTIT_NO_DEVICE_DETECTED_BY_GNA_LIB = 13,
    GSTIT_UNHANDLED_EXCEPTION = 14,
    GSTIT_UNHANDLED_SIGNAL = 15,
    GSTIT_PRINT_HELP_ONLY = 16
};

class GnaSelfTestException : public std::exception
{
public:
    GnaSelfTestException(int codeIn) : code{codeIn}
    {
    }
    int GetExitCode() const
    {
        return code;
    }
private:
    int code;
};

class GnaSelfTestIssue
{
public:
    GnaSelfTestIssue(GSTIT);

    int GetExitCode() const
    {
        return issueType;
    }

    void Log() const;

private:
    std::string GetDescription() const;

    GSTIT issueType;
    std::string symbol;
};

class MultiOs
{
public:
    static void Init();
};
