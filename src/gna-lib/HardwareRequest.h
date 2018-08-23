/*
 INTEL CONFIDENTIAL
 Copyright 2018 Intel Corporation.

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

#include "gna-api.h"
#include "gna-api-instrumentation.h"

#include<map>
#include<memory>
#include<tuple>
#include<vector>

#include "GnaConfig.h"
#include "GnaTypes.h"
#include "common.h"
#include "profiler.h"

namespace GNA
{

class HardwareModel;
class RequestConfiguration;
class LayerConfiguration;

enum GnaOperationMode : uint8_t
{
    GMM = 0,
    xNN = 1
};

struct IoBufferPatch
{
    uint32_t bufferLdOffset;
    uint32_t bufferOffset;
};

struct NnopTypePatch
{
    uint32_t nnopTypeLdOffset;
    NN_OP_TYPE nnopType;
};

struct XnnAlPatch
{
    uint32_t xnnAlBufferLdOffset;
    uint32_t xnnAlBufferOffset;

    uint32_t xnnAlIndicesLdOffset;
    uint16_t xnnAlIndices;
};

struct GmmAlPatch
{
    uint32_t gmmAlBufferLdOffset;
    ASLADDR gmmAlBufferOffset;

    uint32_t gmmAlIndicesLdOffset;
    ASTLISTLEN gmmAlIndices;

    uint32_t gmmScrlenLdOffset;
    GMMSCRLEN gmmSrclen;
};

class HardwareRequest
{
public:
    HardwareRequest(uint64_t memoryId, const HardwareModel& hwModelIn,
                    const RequestConfiguration& requestConfigurationIn);

    void Invalidate();
    void Update(uint32_t layerIndex, uint32_t layerCount, GnaOperationMode operationMode);

    /* these fields will not change between request executions */
    const uint64_t MemoryId;
    const gna_model_id ModelId;
    const gna_hw_perf_encoding HwPerfEncoding;
    const gna_request_cfg_id RequestConfigId;

    /* these fields can change on each request execution */
    GnaOperationMode Mode;

    /* xNN fields */
    uint32_t LayerBase;
    uint32_t LayerCount;

    /* GMM fields */
    uint32_t GmmOffset;
    bool ActiveListOn;

    std::vector<IoBufferPatch> IoBuffers;
    std::vector<NnopTypePatch> NnopTypes;
    std::vector<GmmAlPatch> GmmActiveLists;
    std::vector<XnnAlPatch> XnnActiveLists;

    /* Hardware request ready for driver submition indicator */
    bool SubmitReady = false;

private:

    const RequestConfiguration& requestConfiguration;
    const HardwareModel& hwModel;

    std::map<uint32_t, bool> activeLists;

    void updateActiveLists(uint32_t layerIndex, uint32_t layerCount);

};

};
