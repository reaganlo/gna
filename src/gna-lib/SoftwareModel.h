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

#include "HardwareRequest.h"
#include "IScorable.h"
#include "Layer.h"
#include "Logger.h"
#include "ModelError.h"


#include <cstdint>
#include <memory>
#include <vector>

namespace GNA
{

class BaseValidator;
class RequestConfiguration;
class RequestProfiler;

class SoftwareModel : public IScorable
{
public:
    static void LogAcceleration(AccelerationMode accel)
    {
        auto name = accel.GetName();
        Log->Message("Processing using %s acceleration\n", name);
    }
    static void LogOperationMode(GnaOperationMode mode)
    {
        if (mode == GMM)
        {
            Log->Message("Processing using GMM operation mode\n");
        }
        else if (mode == xNN)
        {
            Log->Message("Processing using xNN operation mode\n");
        }
    }
    
    SoftwareModel(const Gna2Model& model,
        BaseValidator validator,
        const std::vector<Gna2AccelerationMode>& supportedCpuAccelerations);

    SoftwareModel(const SoftwareModel &) = delete;
    SoftwareModel& operator=(const SoftwareModel&) = delete;
    virtual ~SoftwareModel() = default;

    virtual uint32_t Score(
        uint32_t layerIndex,
        uint32_t layerCountIn,
        RequestConfiguration const &requestConfiguration,
        RequestProfiler &profiler,
        KernelBuffers *fvBuffers) override;

    void validateConfiguration(const RequestConfiguration& configuration) const;

    uint32_t GetMaximumOperandSize(uint32_t operandIndex);

    Layer const& GetLayer(uint32_t layerIndex) const;

    std::vector<std::unique_ptr<Layer>> const& GetLayers() const
    {
        return layers;
    }

private:
    template<class T>
    void build(const T* const operations, const BaseValidator & validator)
    {
        maximumOperandSizes.emplace(ScratchpadOperandIndex, 0);
        maximumOperandSizes.emplace(SoftwareScratchpadOperandIndex, 0);

        for (auto i = uint32_t{ 0 }; i < layerCount; i++)
        {
            try
            {
                auto layer = Layer::Create(operations[i], validator);
                buildSingleLayer(layer);
            }
            catch (GnaModelErrorException& e)
            {
                e.SetLayerIndex(i);
                throw;
            }
            catch (const GnaException& e)
            {
                throw GnaModelErrorException(i, e.GetStatus());
            }
            catch (...)
            {
                throw GnaModelErrorException(i);
            }
        }
    }

    void buildSingleLayer(std::unique_ptr<Layer> & layer);

    void CheckModel(uint32_t declaredBatchSize, void * operationPointer) const;

    uint32_t FindMaximumOperandSize(uint32_t operandIndex) const;

    static void FindMaximumOperandSizeForSingleLayer(Layer const & layer, uint32_t operandIndex,
        uint32_t & maxSize);

    std::vector<std::unique_ptr<Layer>> layers;

    uint32_t const layerCount;

    const std::vector<Gna2AccelerationMode>& supportedCpuAccelerations;

    std::map<uint32_t /* operandIndex */, uint32_t> maximumOperandSizes;
};

struct InferenceConfig
{
    typedef ExecutionConfig& (InferenceConfig::*GetEffectiveMethod)(Layer const & layer) const;

    InferenceConfig(KernelBuffers *fvBuffers, RequestConfiguration const &requestConfiguration);

    ExecutionConfig& GetEffective(Layer& layer) const
    {
        return (this->*getEffective)(layer);
    }

    // scoring saturation counter
    uint32_t SaturationCount;

private:
    GetEffectiveMethod getEffective;

    ExecutionConfig& getNormal(Layer const & layer) const;
    ExecutionConfig& getForAdlFix(Layer const & layer) const;

    // if ADL consistency is active
    bool hasAdlConsistency = false;

    // config for usual inference request
    std::unique_ptr<ExecutionConfig> executionConfig;

    // config for inference request with ADL consistency workaround
    std::unique_ptr<ExecutionConfig> executionConfigAdl;
};

}
