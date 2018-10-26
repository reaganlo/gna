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

#include "IoctlSender.h"
#include "HardwareRequest.h"

#if defined(_WIN32)
#include "GnaDrvApi.h"
typedef GNA_CALC_IN GnaDriverRequest;
typedef GNA_MEMORY_PATCH GnaMemoryPatch;
#else
#include "string.h"
#include "gna.h"
typedef struct gna_score_cfg GnaDriverRequest;
typedef struct gna_memory_patch GnaMemoryPatch;
/*********** !!!WARNING!!! ************
 * use these variable names only when *
 * dereferencing driver structures    *
 **************************************/
#define memoryId memory_id
#define hwPerfEncoding hw_perf_encoding
#define configSize config_size
#define patchCount patch_count
#endif

using namespace GNA;

const std::map<uint32_t, gna_device_kind> IoctlSender::deviceTypeMap = {
    { 0x3190, GNA_GLK },
    { 0x5A11, GNA_CNL },
    { 0x8A11, GNA_ICL },
    { 0x9A11, GNA_TGL }
};

void IoctlSender::createRequestDescriptor(HardwareRequest *hardwareRequest)
{
    auto& scoreConfigSize = hardwareRequest->CalculationSize;
    scoreConfigSize = sizeof(GnaDriverRequest);
    auto ioBuffersCount = hardwareRequest->IoBuffers.size();
    auto ioBuffersSize = ioBuffersCount *
        (sizeof(uint32_t) + sizeof(GnaMemoryPatch));
    auto nnopTypesCount = hardwareRequest->NnopTypes.size();
    auto nnopTypesSize = nnopTypesCount *
        (sizeof(NN_OP_TYPE) + sizeof(GnaMemoryPatch));
    auto xnnActiveListsCount = hardwareRequest->XnnActiveLists.size();
    auto xnnActiveListsSize = xnnActiveListsCount *
        (sizeof(GnaMemoryPatch) + sizeof(uint32_t) +
         sizeof(GnaMemoryPatch) + sizeof(uint16_t));
    auto gmmActiveListsCount = hardwareRequest->GmmActiveLists.size();
    auto gmmActiveListsSize = gmmActiveListsCount *
        (sizeof(GnaMemoryPatch) + sizeof(ASLADDR) +
         sizeof(GnaMemoryPatch) + sizeof(ASTLISTLEN) +
         sizeof(GnaMemoryPatch) + sizeof(GMMSCRLEN));

    scoreConfigSize += ioBuffersSize +  nnopTypesSize +  xnnActiveListsSize +  gmmActiveListsSize;
    scoreConfigSize = ALIGN(scoreConfigSize, sizeof(uint64_t));
    hardwareRequest->CalculationData.reset(new uint8_t[scoreConfigSize]);

    auto scoreConfig = reinterpret_cast<GnaDriverRequest *>(hardwareRequest->CalculationData.get());
    memset(scoreConfig, 0, scoreConfigSize);
    scoreConfig->memoryId = hardwareRequest->MemoryId;
    scoreConfig->hwPerfEncoding = hardwareRequest->HwPerfEncoding;
    scoreConfig->configSize = scoreConfigSize;
    scoreConfig->patchCount = ioBuffersCount + nnopTypesCount + 2*xnnActiveListsCount + 3*gmmActiveListsCount;

    GnaMemoryPatch *memory_patch =
        reinterpret_cast<GnaMemoryPatch *>(scoreConfig->patches);
    for(const auto& buffer : hardwareRequest->IoBuffers)
    {
        memory_patch->offset = buffer.bufferLdOffset;
        memory_patch->size = sizeof(uint32_t);
        uint32_t *memory_value = reinterpret_cast<uint32_t *>(memory_patch->data);
        *memory_value = buffer.bufferOffset;
        memory_patch = (GnaMemoryPatch*)((uint8_t *)memory_patch +
            sizeof(GnaMemoryPatch) + memory_patch->size);
    }

    for(const auto& buffer : hardwareRequest->NnopTypes)
    {
        memory_patch->offset = buffer.nnopTypeLdOffset;
        memory_patch->size = sizeof(NN_OP_TYPE);
        NN_OP_TYPE *memory_value = reinterpret_cast<NN_OP_TYPE *>(memory_patch->data);
        *memory_value = buffer.nnopType;
        memory_patch = (GnaMemoryPatch*)((uint8_t *)memory_patch +
            sizeof(GnaMemoryPatch) + memory_patch->size);
    }

    for(const auto& buffer : hardwareRequest->XnnActiveLists)
    {
        memory_patch->offset = buffer.xnnAlBufferLdOffset;
        memory_patch->size = sizeof(uint32_t);
        uint32_t *buffer_value = reinterpret_cast<uint32_t *>(memory_patch->data);
        *buffer_value = buffer.xnnAlBufferOffset;
        memory_patch = (GnaMemoryPatch*)((uint8_t *)memory_patch +
            sizeof(GnaMemoryPatch) + memory_patch->size);

        memory_patch->offset = buffer.xnnAlIndicesLdOffset;
        memory_patch->size = sizeof(uint16_t);
        uint16_t *indices_value = reinterpret_cast<uint16_t *>(memory_patch->data);
        *indices_value = buffer.xnnAlIndices;
        memory_patch = (GnaMemoryPatch*)((uint8_t *)memory_patch +
            sizeof(GnaMemoryPatch) + memory_patch->size);
    }

    for(const auto& buffer : hardwareRequest->GmmActiveLists)
    {
        memory_patch->offset = buffer.gmmAlBufferLdOffset;
        memory_patch->size = sizeof(ASLADDR);
        ASLADDR *buffer_value = reinterpret_cast<ASLADDR *>(memory_patch->data);
        *buffer_value = buffer.gmmAlBufferOffset;
        memory_patch = (GnaMemoryPatch*)((uint8_t *)memory_patch +
            sizeof(GnaMemoryPatch) + memory_patch->size);

        memory_patch->offset = buffer.gmmAlIndicesLdOffset;
        memory_patch->size = sizeof(ASTLISTLEN);
        ASTLISTLEN *indices_value = reinterpret_cast<ASTLISTLEN *>(memory_patch->data);
        *indices_value = buffer.gmmAlIndices;
        memory_patch = (GnaMemoryPatch*)((uint8_t *)memory_patch +
            sizeof(GnaMemoryPatch) + memory_patch->size);

        memory_patch->offset = buffer.gmmScrlenLdOffset;
        memory_patch->size = sizeof(GMMSCRLEN);
        GMMSCRLEN *scrlen_value = reinterpret_cast<GMMSCRLEN *>(memory_patch->data);
        *scrlen_value = buffer.gmmSrclen;
        memory_patch = (GnaMemoryPatch*)((uint8_t *)memory_patch +
            sizeof(GnaMemoryPatch) + memory_patch->size);
    }

    hardwareRequest->SubmitReady = true;
}
