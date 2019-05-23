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

#include "Memory.h"
#include "DriverInterface.h"
#include "HardwareRequest.h"

#if defined(_WIN32)
#include "GnaDrvApi.h"
typedef GNA_CALC_IN GnaDriverRequest;
typedef GNA_MEMORY_PATCH GnaMemoryPatch;
#else
#include "gna.h"
#include <cstring>
typedef struct gna_score_cfg GnaDriverRequest;
typedef struct gna_buffer GnaDriverBuffer;
typedef struct gna_patch GnaDriverPatch;
/*********** !!!WARNING!!! ************
 * use these variable names only when *
 * dereferencing driver structures    *
 **************************************/
#define memoryId memory_id
#define ctrlFlags ctrl_flags
#define hwPerfEncoding hw_perf_encoding
#define buffersPtr buffers_ptr
#define bufferCount buffer_count
#define patchesPtr patches_ptr
#define patchCount patch_count
#endif

using namespace GNA;

void DriverInterface::createRequestDescriptor(HardwareRequest& hardwareRequest) const
{
    auto& scoreConfigSize = hardwareRequest.CalculationSize;
    scoreConfigSize = sizeof(GnaDriverRequest);

    for (const auto &buffer : hardwareRequest.DriverBuffers)
    {
        scoreConfigSize += sizeof(GnaDriverBuffer) +
            buffer.Patches.size() * sizeof(GnaDriverPatch);
    }

    scoreConfigSize = ALIGN(scoreConfigSize, sizeof(uint64_t));
    hardwareRequest.CalculationData.reset(new uint8_t[scoreConfigSize]);

    uint8_t *calculationData = static_cast<uint8_t *>(hardwareRequest.CalculationData.get());
    auto scoreConfig = reinterpret_cast<GnaDriverRequest *>(hardwareRequest.CalculationData.get());
    memset(scoreConfig, 0, scoreConfigSize);
    scoreConfig->ctrlFlags.hwPerfEncoding = hardwareRequest.HwPerfEncoding;
    scoreConfig->buffersPtr = reinterpret_cast<uintptr_t>(calculationData + sizeof(GnaDriverRequest));
    scoreConfig->bufferCount = hardwareRequest.DriverBuffers.size();

    // TODO: refactor according to new DDI
    auto buffer = reinterpret_cast<GnaDriverBuffer *>(scoreConfig->buffersPtr);
    auto patch = reinterpret_cast<GnaDriverPatch *>(scoreConfig->buffersPtr +
                                                scoreConfig->bufferCount * sizeof(GnaDriverBuffer));

    for (const auto &driverBuffer : hardwareRequest.DriverBuffers)
    {
        buffer->memoryId = driverBuffer.Buffer->GetId();
        buffer->offset = 0;
        buffer->size = driverBuffer.Buffer->GetSize();
        buffer->patchesPtr = reinterpret_cast<uintptr_t>(patch);
        buffer->patchCount = driverBuffer.Patches.size();

        for (const auto &driverPatch : driverBuffer.Patches)
        {
            patch->offset = driverPatch.Offset;
            patch->size = driverPatch.Size;
            patch->value = driverPatch.Value;
            patch++;
        }

        buffer++;
    }

    hardwareRequest.SubmitReady = true;
}

