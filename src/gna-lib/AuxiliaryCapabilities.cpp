/*
 INTEL CONFIDENTIAL
 Copyright 2020 Intel Corporation.

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

#include "AuxiliaryCapabilities.h"

#include "Capabilities.h"
#include "DataMode.h"
#include "Tensor.h"


using namespace GNA;

namespace GNA
{

template<uint32_t operandIndex>
struct AuxComponentCaps : protected LayerCapabilities
{
    template<Gna2DeviceGeneration generation, Gna2DeviceGeneration modeGeneration = generation>
    static std::pair<const Gna2DeviceGeneration, std::shared_ptr<ComponentLimits>>
        Make()
    {
        //TODO:3:use variadic template Make
        return { generation,
            std::make_shared<TensorLimits>(TensorLimits{
                {GNA_TENSOR_HW},
                {{GNA_DIM_H, MakeLimits<InputGroupMax, operandIndex>()},
                    {GNA_DIM_W, MakeLimitsMulti<LegacyInputs, operandIndex>()}},
                GetCommonModes(operandIndex, modeGeneration)}) };
    }

    template<Gna2DeviceGeneration generation, Gna2DeviceGeneration modeGeneration = generation>
    static std::pair<const Gna2DeviceGeneration, std::shared_ptr<ComponentLimits>>
        MakeInterleaved()
    {
        //TODO:3:use variadic template Make
        return { generation,
            std::make_shared<TensorLimits>(TensorLimits{
                {GNA_TENSOR_HW},
                {{GNA_DIM_H, MakeLimitsMulti<LegacyInputs, operandIndex>()},
                    {GNA_DIM_W, MakeLimits<InputGroupMax, operandIndex>()}},
                GetCommonModes(operandIndex, modeGeneration)}) };
    }
};

using InputAuxCaps = AuxComponentCaps<InputOperandIndex>;
using OutputAuxCaps = AuxComponentCaps<OutputOperandIndex>;

const FullCapabilitiesMap& AuxiliaryCapabilities::GetOperands(uint32_t operandIndex)
{
    static const ComponentFullCapabilityMap operands =
    {
        {InputOperandIndex,{
            {INTEL_COPY, {
                InputAuxCaps::Make<Gna2DeviceGeneration0_9>(),
                InputAuxCaps::Make<Gna2DeviceGeneration3_0>(),
                InputAuxCaps::Make<Gna2DeviceGeneration3_5>(),
            }},
            {INTEL_INTERLEAVE, {
                InputAuxCaps::Make<Gna2DeviceGeneration0_9>(),
                InputAuxCaps::Make<Gna2DeviceGeneration3_0>(),
                InputAuxCaps::Make<Gna2DeviceGeneration3_5>(),
            }},
            {INTEL_DEINTERLEAVE, {
                InputAuxCaps::MakeInterleaved<Gna2DeviceGeneration0_9>(),
                InputAuxCaps::MakeInterleaved<Gna2DeviceGeneration3_0>(),
                InputAuxCaps::MakeInterleaved<Gna2DeviceGeneration3_5>(),
            }},
        }},
        {OutputOperandIndex,{
            {INTEL_COPY, {
                OutputAuxCaps::Make<Gna2DeviceGeneration0_9>(),
                OutputAuxCaps::Make<Gna2DeviceGeneration3_0>(),
                OutputAuxCaps::Make<Gna2DeviceGeneration3_5>(),
            }},
            {INTEL_DEINTERLEAVE, {
                OutputAuxCaps::Make<Gna2DeviceGeneration0_9>(),
                OutputAuxCaps::Make<Gna2DeviceGeneration3_0>(),
                OutputAuxCaps::Make<Gna2DeviceGeneration3_5>(),
            }},
            {INTEL_INTERLEAVE, {
                OutputAuxCaps::MakeInterleaved<Gna2DeviceGeneration0_9>(),
                OutputAuxCaps::MakeInterleaved<Gna2DeviceGeneration3_0>(),
                OutputAuxCaps::MakeInterleaved<Gna2DeviceGeneration3_5>(),
            }},
        }},
    };

    return operands.at(operandIndex);
}

}
