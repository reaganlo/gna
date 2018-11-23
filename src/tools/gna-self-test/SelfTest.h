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
#include "gna-api.h"
#define LOG(...) { fputs("[GNA-SELF-TEST] ",stdout);\
                   fprintf(stdout,__VA_ARGS__);\
                 }

#define DEFAULT_SELFTEST_TIMEOUT_MS 10000

void PressEnterToContinue();

class SelfTestDevice {
    gna_device_id deviceId;
    uint8_t* pinned_mem_ptr = nullptr;
    gna_model_id sampleModelId;
    gna_request_cfg_id configId;
    intel_nnet_type_t nnet;  // main neural network container
    int16_t *pinned_outputs;
    int16_t *pinned_inputs;

public:
    SelfTestDevice();
    // obtains pinned memory shared with the device
    void Alloc(const uint32_t bytesRequested, const uint16_t layerCount, const uint16_t gmmCount);
    //TODO refactor into factory
    void SampleModelCreate();

    void BuildSampleRequest();
    void ConfigRequestBuffer();
    gna_request_id requestId;
    void RequestAndWait();

    void CompareResults();
    ~SelfTestDevice();
};

#define DECL_GNASELFTESTISSUE(NAME) static const GnaSelfTestIssue NAME
class GnaSelfTestIssue {
    const char* info;
public:
    GnaSelfTestIssue(const char* info) : info{ info } {}
    void Handle() const;
    DECL_GNASELFTESTISSUE(GENERAL_GNA_NO_SUCCESS);
    DECL_GNASELFTESTISSUE(DRV_1_INSTEAD_2);
    DECL_GNASELFTESTISSUE(NO_HARDWARE);
    DECL_GNASELFTESTISSUE(NUL_DRIVER);
    DECL_GNASELFTESTISSUE(DEVICE_OPEN_NO_SUCCESS);
    DECL_GNASELFTESTISSUE(HARDWARE_OR_DRIVER_NOT_INITIALIZED_PROPERLY);
    DECL_GNASELFTESTISSUE(GNAALLOC_MEM_ALLOC_FAILED);
    DECL_GNASELFTESTISSUE(UNKNOWN_DRIVER);
    DECL_GNASELFTESTISSUE(MALLOC_FAILED);
    DECL_GNASELFTESTISSUE(SETUPDI_ERROR);
    DECL_GNASELFTESTISSUE(NO_DRIVER);
};
