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

#include "GmmLayerCapabilities.h"

#include "Capabilities.h"
#include "DataMode.h"
#include "LayerCapabilities.h"
#include "Tensor.h"

using namespace GNA;

const FullCapabilitiesMap& GmmLayerCapabilities::GetOperands(uint32_t operandIndex)
{
    static const ComponentFullCapabilityMap operands =
    {
        {InputOperandIndex,{
            {INTEL_GMM, {
                {Gna2DeviceGenerationGmm, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_HW},
                    {{GNA_DIM_H, {1, BatchSizeMax, 1, Gna2StatusXnnErrorInputVolume}},
                    {GNA_DIM_W, {GMM_FV_ELEMENT_COUNT_MIN, GMM_FV_ELEMENT_COUNT_MAX, GMM_FV_ELEMENT_COUNT_MULTIPLE_OF, Gna2StatusBadFeatLength}}},
                    {{ Gna2DataTypeUint8}, Gna2StatusXnnErrorInputBytes }})}
                }},
        }},
        {OutputOperandIndex, {
            {INTEL_GMM, {
                {Gna2DeviceGenerationGmm, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_HW}, // H - GMM States, W - grouping
                    {{GNA_DIM_W, {1, BatchSizeMax, 1, Gna2StatusXnnErrorOutputVolume}},
                    {GNA_DIM_H, {1, GMM_STATES_COUNT_MAX, 1, Gna2StatusXnnErrorOutputVolume}}},
                    {{Gna2DataTypeUint32 }, Gna2StatusXnnErrorOutputBytes}})}
            }},
        }},
        {WeightOperandIndex, {
            {INTEL_GMM, {
                {Gna2DeviceGenerationGmm, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_HWD },                  // H - GMM states, W - #mixtures, D - #inputs
                    {{GNA_DIM_H, {1, GMM_STATES_COUNT_MAX, 1, Gna2StatusGmmBadNumGmm}},
                    {GNA_DIM_W, {1, GMM_MIXTURE_COMP_COUNT_MAX, 1, Gna2StatusGmmBadMixCnum}},
                    {GNA_DIM_D, {GMM_FV_ELEMENT_COUNT_MIN, GMM_FV_ELEMENT_COUNT_MAX, GMM_FV_ELEMENT_COUNT_MULTIPLE_OF, Gna2StatusBadFeatLength}}},
                    {{Gna2DataTypeUint8, Gna2DataTypeUint16}, Gna2StatusGmmBadMode},
                    {GMM_MEM_ALIGNMENT, Gna2StatusGmmBadVarsAlign}})}
            }},
        }},
        {BiasOperandIndex,{
            {INTEL_GMM, {
                {Gna2DeviceGenerationGmm, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_HW},                   // H - GMM states, W - #mixtures
                    {{GNA_DIM_W, {1, GMM_MIXTURE_COMP_COUNT_MAX, 2, Gna2StatusGmmBadMixCnum}},
                    {GNA_DIM_H, {1, GMM_STATES_COUNT_MAX, 1, Gna2StatusGmmBadNumGmm}}},
                    {{Gna2DataTypeUint32}, Gna2StatusGmmBadMode},
                    {GMM_MEM_ALIGNMENT, Gna2StatusGmmBadGconstAlign}})}
            }}
        }}
    };
    return operands.at(operandIndex);

}

