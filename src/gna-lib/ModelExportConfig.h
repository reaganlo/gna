/*
 INTEL CONFIDENTIAL
 Copyright 2019 Intel Corporation.

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

#include "gna2-model-export-api.h"

#include "gna2-common-api.h"
#include "gna2-common-impl.h"

#include <map>

namespace GNA
{
class ModelExportConfig
{
public:
    explicit ModelExportConfig(Gna2UserAllocator userAllocatorIn);
    void SetSource(uint32_t deviceId, uint32_t modelId);
    void SetTarget(Gna2DeviceVersion version);
    void Export(enum Gna2ModelExportComponent componentType,
        void ** exportBuffer,
        uint32_t * exportBufferSize);

protected:
    void ValidateState() const;

private:
    Gna2UserAllocator userAllocator = nullptr;
    uint32_t sourceDeviceId = Gna2DisabledU32;
    uint32_t sourceModelId = Gna2DisabledU32;
    Gna2DeviceVersion targetDeviceVersion = Gna2DeviceVersionSoftwareEmulation;

    static void* privateAllocator(uint32_t size);
    static void privateDeAllocator(void * ptr);
};

class ModelExportManager
{
public:
    static ModelExportManager& Get();

    ModelExportManager(const ModelExportManager&) = delete;
    void operator=(const ModelExportManager&) = delete;

    uint32_t AddConfig(Gna2UserAllocator userAllocator);
    void RemoveConfig(uint32_t id);
    ModelExportConfig& GetConfig(uint32_t exportConfigId);

private:
    ModelExportManager() = default;
    uint32_t configCount = 0;
    std::map<uint32_t, ModelExportConfig> allConfigs;
};

}
