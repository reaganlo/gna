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

#include <functional>
#include <memory>
#include <vector>

#include "AccelerationDetector.h"
#include "KernelArguments.h"
#include "IScorable.h"
#include "Tensor.h"
#include "Validator.h"

#include "common.h"
#include "profiler.h"

namespace GNA
{

class Layer;
struct LayerConfiguration;
class RequestConfiguration;
struct RequestProfiler;

class SoftwareModel : public IScorable
{
public:
    static void LogAcceleration(AccelerationMode accel)
    {
        Log->Message("Processing using %s acceleration\n",
            AccelerationDetector::AccelerationToString(accel));
    }

    SoftwareModel(const gna_model *const network,
                  BaseValidator validator,
                  AccelerationMode fastestAcceleration);

    SoftwareModel(const SoftwareModel &) = delete;
    SoftwareModel& operator=(const SoftwareModel&) = delete;
    ~SoftwareModel() = default;

    virtual uint32_t Score(
        uint32_t layerIndex,
        uint32_t layerCount,
        RequestConfiguration const &requestConfiguration,
        RequestProfiler *profiler,
        KernelBuffers *fvBuffers,
        GnaOperationMode operationMode = xNN) override;

    void validateConfiguration(const RequestConfiguration& configuration) const;

    std::vector<std::unique_ptr<Layer>> Layers;

private:
    void build(const nn_layer* layers, const BaseValidator& validator);

    AccelerationMode getEffectiveAccelerationMode(AccelerationMode requiredAccel) const;

    uint32_t const layerCount;
    AccelerationMode const fastestAccel;
};

}
