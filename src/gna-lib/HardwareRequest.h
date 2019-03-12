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

#include "common.h"
#include "profiler.h"

#include "GnaConfig.h"
#include "GnaTypes.h"
#include "HardwareLayer.h"
#include "Layer.h"

namespace GNA
{

class HardwareModelScorable;
class RequestConfiguration;
struct LayerConfiguration;

enum GnaOperationMode : uint8_t
{
    GMM = 0,
    xNN = 1
};

/*struct IoBufferPatch
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
*/

struct DriverPatch
{
    DriverPatch(uint32_t offset, uint32_t value, uint32_t size) :
        Offset(offset), Value(value), Size(size)
    {}

    uint32_t Offset;
    uint32_t Value;
    uint32_t Size;
};

struct DriverBuffer
{
    explicit DriverBuffer(Memory *memoryIn) :
        Buffer (memoryIn)
    {}

    Memory *Buffer;
    std::vector<DriverPatch> Patches = {};
};

class HardwareRequest
{
public:
    // TODO:3: ldMemory should be on modelMemoryObjects
    HardwareRequest(const HardwareModelScorable& hwModelIn,
        const RequestConfiguration& requestConfigurationIn,
        Memory *ldMemory, const std::vector<Memory *>& modelMemoryObjectsIn);

    void Invalidate();
    void Update(uint32_t layerIndex, uint32_t layerCount, GnaOperationMode operationMode);

    /* these fields will not change between request executions */
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

    std::vector<DriverBuffer> DriverMemoryObjects;

    /* Driver specific request data*/
    std::unique_ptr<uint8_t[]> CalculationData = nullptr;
    size_t CalculationSize;

    /* Hardware request ready for driver submition indicator */
    bool SubmitReady = false;

private:

    const RequestConfiguration& requestConfiguration;
    const HardwareModelScorable& hwModel;

    std::map<uint32_t, bool> activeLists;

    void updateActiveLists(uint32_t layerIndex, uint32_t layerCount);

    void generateBufferPatches(const LayerConfiguration& layerConfiguration,
                               const Layer &layer, const HardwareLayer &hwLayer);

    Memory *ldMemory;
};

};
