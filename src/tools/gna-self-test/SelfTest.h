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
#include <cstdio>
#include <string>
#include "gna-api.h"
#include "SampleModelForGnaSelfTest.h"

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

class GnaSelfTestConfig
{
public:
    static GnaSelfTestConfig ReadConfigFromCmdLine(int argc, const char *const argv[]);
    bool VerboseMode() const { return verboseMode; }
    bool PauseMode() const { return pauseAfterError; }
    bool ContinueMode() const { return continueAfterError; }
private:
    GnaSelfTestConfig(int argc, const char * const argv[]);
    bool verboseMode = false;
    bool continueAfterError = false;
    bool pauseAfterError = false;
};

class GnaSelfTestIssue;
class GnaSelfTest {
public:
    GnaSelfTest(GnaSelfTestConfig configIn) :config{configIn}{}
    void StartTest();
    void Handle(const GnaSelfTestIssue& issue) const;
    void HandleGnaStatus(const intel_gna_status_t &status, const char *what = "") const;
private:
    void askToContinueOrExit(int exitCode) const;
    GnaSelfTestConfig config;
};

#define DEFAULT_SELFTEST_TIMEOUT_MS 10000

class SelfTestDevice
{
public:
    SelfTestDevice(const GnaSelfTest& gst);
    // obtains pinned memory shared with the device
    void Alloc(const uint32_t bytesRequested, const uint16_t layerCount, const uint16_t gmmCount);

    void SampleModelCreate(const SampleModelForGnaSelfTest &model);

    void BuildSampleRequest();
    void ConfigRequestBuffer();

    void RequestAndWait();

    void CompareResults(const SampleModelForGnaSelfTest &model);
    ~SelfTestDevice();
private:
    gna_device_id deviceId = 0;
    uint8_t *pinned_mem_ptr = nullptr;
    gna_model_id sampleModelId;
    gna_request_cfg_id configId;
    intel_nnet_type_t nnet; // main neural network container
    int16_t *pinned_outputs;
    int16_t *pinned_inputs;
    gna_request_id requestId;
    const GnaSelfTest& gnaSelfTest;
};

// Types and exit codes of GnaSelfTestIssue
enum GSTIT
{
    GSTIT_GENERAL_GNA_NO_SUCCESS = 1,
    GSTIT_DRV_1_INSTEAD_2,
    GSTIT_NO_HARDWARE,
    GSTIT_NUL_DRIVER,
    GSTIT_DEVICE_OPEN_NO_SUCCESS,
    GSTIT_HARDWARE_OR_DRIVER_NOT_INITIALIZED_PROPERLY,
    GSTIT_GNAALLOC_MEM_ALLOC_FAILED,
    GSTIT_UNKNOWN_DRIVER,
    GSTIT_MALLOC_FAILED,
    GSTIT_SETUPDI_ERROR,
    GSTIT_NO_DRIVER,
    GSTIT_ERRORS_IN_SCORES
};

class GnaSelfTestIssue
{
public:
    GnaSelfTestIssue(GSTIT);
    std::string GetDescription() const;
    int ExitCode() const
    {
        return issueType;
    }
private:
    GSTIT issueType;
    std::string symbol;
};
