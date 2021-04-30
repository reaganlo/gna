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

#include "LayerCapabilities.h"

#include "AffineLayerCapabilities.h"
#include "AuxiliaryCapabilities.h"
#include "ConvolutionalLayer2DCapabilities.h"
#include "GmmLayerCapabilities.h"

using namespace GNA;

const OperationCapabilityMap& LayerCapabilities::GetOperands(nn_operation operation, uint32_t operandIndex)
{
    switch (operation)
    {
    case INTEL_AFFINE:
    case INTEL_AFFINE_DIAGONAL:
    case INTEL_AFFINE_MULTIBIAS:
    case INTEL_RECURRENT:
        return AffineLayerCapabilities::GetOperands(operandIndex).at(operation);
    case INTEL_COPY:
    case INTEL_INTERLEAVE:
    case INTEL_DEINTERLEAVE:
        return AuxiliaryCapabilities::GetOperands(operandIndex).at(operation);
    case INTEL_CONVOLUTIONAL:
    case INTEL_CONVOLUTIONAL_2D:
    case INTEL_CONVOLUTIONAL_1D:
        return ConvolutionalLayer2DCapabilities::GetOperands(operandIndex).at(operation);
    case INTEL_GMM:
        return GmmLayerCapabilities::GetOperands(operandIndex).at(operation);
    default:
        throw GnaException(Gna2StatusNotImplemented);
    }
}

const DataModeLimits& LayerCapabilities::GetCommonModes(uint32_t operandIndex, Gna2DeviceGeneration generation)
{
    static const std::map<uint32_t, std::map<Gna2DeviceGeneration, DataModeLimits>> modes =
    {
        {InputOperandIndex,
            {{Gna2DeviceGeneration0_9, {{Gna2DataTypeInt16}, Gna2StatusXnnErrorInputBytes}},
            {Gna2DeviceGeneration2_0, {{Gna2DataTypeInt16}, Gna2StatusXnnErrorInputBytes}},
            {Gna2DeviceGeneration3_0, {{Gna2DataTypeInt8, Gna2DataTypeInt16}, Gna2StatusXnnErrorInputBytes}},
            {Gna2DeviceGeneration3_1, {{Gna2DataTypeInt8, Gna2DataTypeInt16}, Gna2StatusXnnErrorInputBytes}},
            {Gna2DeviceGeneration3_5, {
                MakeDataModesCartesian({Gna2DataTypeInt8, Gna2DataTypeInt16}),
                Gna2StatusXnnErrorInputBytes}},}
        },
        {OutputOperandIndex,
            {{Gna2DeviceGeneration0_9, {{Gna2DataTypeInt16, Gna2DataTypeInt32}, Gna2StatusXnnErrorOutputBytes}},
            {Gna2DeviceGeneration2_0, {{Gna2DataTypeInt16, Gna2DataTypeInt32}, Gna2StatusXnnErrorOutputBytes}},
            {Gna2DeviceGeneration3_0, {{Gna2DataTypeInt8, Gna2DataTypeInt16, Gna2DataTypeInt32}, Gna2StatusXnnErrorOutputBytes}},
            {Gna2DeviceGeneration3_1, {{Gna2DataTypeInt8, Gna2DataTypeInt16, Gna2DataTypeInt32}, Gna2StatusXnnErrorOutputBytes}},
            {Gna2DeviceGeneration3_5, {
                MakeDataModesCartesian({Gna2DataTypeInt8, Gna2DataTypeInt16, Gna2DataTypeInt32}),
                Gna2StatusXnnErrorInputBytes}},}
        },
        {BiasOperandIndex, {
            MakeModes<Gna2DeviceGeneration0_9, BiasOperandIndex>
                (Gna2DataTypeInt32, Gna2DataTypeCompoundBias),
            MakeModes<Gna2DeviceGeneration2_0, BiasOperandIndex>
               (Gna2DataTypeInt32, Gna2DataTypeCompoundBias),
            MakeModes<Gna2DeviceGeneration3_0, BiasOperandIndex>
               (Gna2DataTypeInt8, Gna2DataTypeInt16, Gna2DataTypeInt32, Gna2DataTypeCompoundBias),
            MakeModes<Gna2DeviceGeneration3_1, BiasOperandIndex>
               (Gna2DataTypeInt8, Gna2DataTypeInt16, Gna2DataTypeInt32, Gna2DataTypeCompoundBias),
            MakeModes<Gna2DeviceGeneration3_5, BiasOperandIndex>
               (MakeDataModesCartesian(
                    {Gna2DataTypeInt8, Gna2DataTypeInt16, Gna2DataTypeInt32, Gna2DataTypeCompoundBias})),
        }},
    };
    return modes.at(operandIndex).at(generation);
}

constexpr StaticCaps LayerCapabilities::Input;
constexpr StaticCaps LayerCapabilities::InputGroupMax;
constexpr RangeLimits<uint32_t> LayerCapabilities::LegacyInputs;
constexpr StaticCaps LayerCapabilities::InputEqual1;
constexpr StaticCaps LayerCapabilities::Input1D;
constexpr StaticCaps LayerCapabilities::WeightMultiplier;
constexpr StaticCaps LayerCapabilities::OutputRnn;
