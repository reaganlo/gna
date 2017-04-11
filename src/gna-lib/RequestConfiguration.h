/*
 INTEL CONFIDENTIAL
 Copyright 2017 Intel Corporation.

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

#include <array>
#include <map>
#include <memory>
#include <vector>

#include "Address.h"
#include "ActiveList.h"
#include "common.h"
#include "CompiledModel.h"

namespace GNA
{
struct ConfigurationBuffer : public InOutBuffer
{
    ConfigurationBuffer(gna_buffer_type type, void *address);

    ConfigurationBuffer(ConfigurationBuffer &&) = default;

    ConfigurationBuffer() = delete;
    ConfigurationBuffer(const ConfigurationBuffer &) = delete;
    ConfigurationBuffer& operator=(const ConfigurationBuffer&) = delete;

    gna_buffer_type type;
};

struct LayerConfiguration
{
    std::unique_ptr<ActiveList> ActiveList;
    std::unique_ptr<ConfigurationBuffer> InputBuffer;
    std::unique_ptr<ConfigurationBuffer> OutputBuffer;
};

/*
** RequestConfiguration is a bunch of request buffers
** sent to GNA kernel driver as part of WRITE request
**
 */
class RequestConfiguration
{
public:
    RequestConfiguration(const CompiledModel& model, gna_request_cfg_id configId);

    void AddBuffer(gna_buffer_type type, uint32_t layerIndex, void *address);
    void AddActiveList(uint32_t layerIndex, uint32_t indicesCount, uint32_t *indices);

    void GetHwConfigData(void* &buffer, size_t &size) const;

    const CompiledModel& Model;

    const gna_request_cfg_id ConfigId;

    gna_hw_perf_encoding HwPerfEncoding = PERF_COUNT_DISABLED;
    gna_perf_t * PerfResults = nullptr;

    std::map<uint32_t, std::unique_ptr<LayerConfiguration>> LayerConfigurations;
    uint32_t InputBuffersCount = 0;
    uint32_t OutputBuffersCount = 0;
    uint32_t ActiveListCount = 0;

private:
    void invalidateHwConfigCache();
    void writeLayerConfigBuffersIntoHwConfigCache(PGNA_BUFFER_DESCR &lyrsCfg) const;
    void writeLayerConfigActiveListsIntoHwConfigCache(PGNA_ACTIVE_LIST_DESCR &actLstCfg) const;

    mutable std::unique_ptr<uint8_t[]> hwConfigCache;
    mutable size_t hwConfigSize = 0;
};
}
