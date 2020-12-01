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

#include "Capabilities.h"
#include "DataMode.h"
#include "Tensor.h"

#include <map>

namespace GNA
{

using ComponentFullCapabilityMap = std::map<const uint32_t, FullCapabilitiesMap>;

template <nn_operation Operation>
struct OperationCaps
{};

template<nn_operation Operation>
FullCapabilitiesMap::value_type GetOperationCaps(uint32_t operandIndex)
{
    return { Operation, OperationCaps<Operation>::GetOperands(operandIndex).at(Operation) };
}

struct LayerCapabilities
{
    /** Total number of input elements constraint - must be multiple of */
    static constexpr uint32_t InputElementCountMultiplier = 8;

    // TODO:4: ensure single point of definition
    /** Number of input groups constraint for Copy layer 3.0- max */
    static constexpr uint32_t CopyRowsMax = 255;

    /** Total number of output elements constraint - must be multiple of */
    static constexpr uint32_t RecurrentOutputElementCountMultiplier = 32;

    /** Total number of input elements constraint - max elements */
    static constexpr uint32_t InputElementCountMax = UINT16_MAX;

    /** Weight elements size constraint - max size B */
    static constexpr uint32_t WeightElementSizeMax = 2;

    static const DataModeLimits & GetModes(uint32_t operandIndex, Gna2DeviceGeneration generation);

    static std::vector<DataMode> MakeDataModesCartesian(std::vector<Gna2DataType> types,
        std::vector<Gna2TensorMode> modes = { Gna2TensorModeDefault, Gna2TensorModeExternalBuffer })
    {
        auto cartesian = std::vector<DataMode>(types.size() * modes.size());
        for (auto && type : types)
        {
            for (auto && mode : modes)
            {
                cartesian.emplace_back(DataMode{ type, mode });
            }
        }
        return cartesian;
    }

    template<uint32_t min, uint32_t max, uint32_t mulitpliers, Gna2Status status>
    static const RangeLimits<>& MakeLimits()
    {
        static const RangeLimits<> limits =
        {
            min, max, mulitpliers, status
        };
        return limits;
    }

    template<Gna2Status status>
    static const RangeLimits<>& GetLimitsBasedOnInput()
    {
        return MakeLimits<1u, InputElementCountMax, 1u, status>();
    }

    template<Gna2Status status>
    static const RangeLimits<>&  GetLimitsBasedOnInputLegacy()
    {
        static const RangeLimits<> limits =
        {
            InputElementCountMultiplier, InputElementCountMax,
            MultiplierMap{
                {Gna2DataTypeInt8, 2 * InputElementCountMultiplier},
                {Gna2DataTypeInt16, 1 * InputElementCountMultiplier},
                {Gna2DataTypeInt32, InputElementCountMultiplier / 2},},
            status
        };
        return limits;
    }

    template<Gna2Status status>
    static const RangeLimits<>& GetLimitsBasedOnInputGroupsMax()
    {
        return MakeLimits<1u, BatchSizeMax, 1u, status>();
    }
};

template<uint32_t operandIndex>
struct ComponentCaps : protected LayerCapabilities
{};

struct LayerCaps : protected LayerCapabilities
{
    template<uint32_t operandIndex>
    static const std::shared_ptr<ComponentLimits>& GetComponentLimits(
        Gna2DeviceGeneration generation)
    {
        static const OperationCapabilityMap caps =
        {
            {ComponentCaps<operandIndex>::template Make<Gna2DeviceGeneration0_9>()},
            {ComponentCaps<operandIndex>::template Make<Gna2DeviceGeneration2_0, Gna2DeviceGeneration0_9>()},
            {ComponentCaps<operandIndex>::template Make<Gna2DeviceGeneration3_0>()},
            {ComponentCaps<operandIndex>::template Make<Gna2DeviceGeneration3_5>()},
        };
        return caps.at(generation);
    }

    template<Gna2DeviceGeneration generation, uint32_t operandIndex>
    static std::pair<const Gna2DeviceGeneration, const std::shared_ptr<ComponentLimits>>
    Make()
    {
        return { generation, GetComponentLimits<operandIndex>(generation) };
    }

    template<Gna2DeviceGeneration generation>
    static std::pair<const Gna2DeviceGeneration, const std::shared_ptr<ComponentLimits>>
    Make(const OrderLimits order, const ShapeLimits& dimensions, const DataModeLimits& modes)
    {
        return {
            generation,
            std::make_shared<TensorLimits>(order, dimensions, modes)
        };
    }

    template<Gna2DeviceGeneration generation, uint32_t operandIndex, Gna2DeviceGeneration modeGeneration = generation>
    static std::pair<const Gna2DeviceGeneration, const std::shared_ptr<ComponentLimits>>
    Make(const OrderLimits order, const ShapeLimits& dimensions)
    {
        return Make<generation>(order, dimensions, ComponentCaps<operandIndex>::GetModes(modeGeneration));
    }
};

template<>
struct ComponentCaps<InputOperandIndex> : protected LayerCapabilities
{
    template<Gna2DeviceGeneration generation, Gna2DeviceGeneration modeGeneration = generation>
    static std::pair<const Gna2DeviceGeneration, std::shared_ptr<ComponentLimits>>
    Make()
    {
        return { generation, std::make_shared<TensorLimits>(TensorLimits{
               {GNA_TENSOR_HW},
               {{GNA_DIM_H, GetLimitsBasedOnInputLegacy<Gna2StatusXnnErrorInputVolume>()},
               {GNA_DIM_W, GetLimitsBasedOnInputGroupsMax<Gna2StatusXnnErrorInputVolume>()}},
               LayerCapabilities::GetModes(InputOperandIndex, modeGeneration)}) };
    }

    static const DataModeLimits& GetModes(Gna2DeviceGeneration generation)
    {
        return LayerCapabilities::GetModes(InputOperandIndex, generation);
    }
};

template<>
struct ComponentCaps<OutputOperandIndex> : protected LayerCapabilities
{
    template<Gna2DeviceGeneration generation, Gna2DeviceGeneration modeGeneration = generation>
    static std::pair<const Gna2DeviceGeneration, std::shared_ptr<ComponentLimits>>
    Make()
    {
        return { generation, std::make_shared<TensorLimits>(TensorLimits{
           {GNA_TENSOR_HW},
           {{GNA_DIM_H, GetLimitsBasedOnInput<Gna2StatusXnnErrorOutputVolume>()},
           {GNA_DIM_W, GetLimitsBasedOnInputGroupsMax<Gna2StatusXnnErrorOutputVolume>()}},
           GetModes(modeGeneration)}) };
    }

    static const DataModeLimits& GetModes(Gna2DeviceGeneration generation)
    {
        return LayerCapabilities::GetModes(OutputOperandIndex, generation);
    }
};

}
