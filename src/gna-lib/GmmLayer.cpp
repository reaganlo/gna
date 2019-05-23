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

#include "GmmLayer.h"

#include "AccelerationDetector.h"
#include "Address.h"
#include "ActiveList.h"
#include "DataMode.h"
#include "Expect.h"
#include "LayerConfiguration.h"
#include "LayerInput.h"
#include "LayerOutput.h"
#include "Validator.h"

#include "gna-api-types-xnn.h"
#include "gna-api.h"

#include <algorithm>
#include <memory>

using namespace GNA;

GmmParams::GmmParams(const gna_gmm_config &config, const uint32_t inputElementCount)
{
    VarianceSize = (GNA_MAXMIX16 == config.mode)
        ? static_cast<uint32_t>(sizeof(uint16_t))
        : static_cast<uint32_t>(sizeof(uint8_t));

    MeanSetOffsetSize = config.mixtureComponentCount * inputElementCount * GMM_MEAN_VALUE_SIZE;
    VarSetOffsetSize = config.mixtureComponentCount * inputElementCount * VarianceSize;
    GaussConstSetOffsetSize = ALIGN(config.mixtureComponentCount, 2) * GMM_CONSTANTS_SIZE;
    if (GMM_LAYOUT_INTERLEAVED == config.layout)
    {
        MeanSetOffsetSize = MeanSetOffsetSize + VarSetOffsetSize + GaussConstSetOffsetSize;
        VarSetOffsetSize = MeanSetOffsetSize;
        GaussConstSetOffsetSize = MeanSetOffsetSize;
    }
    else {
        Expect::InRange(MeanSetOffsetSize, GMM_FV_ELEMENT_COUNT_MULTIPLE_OF,
            GMM_MIXTURE_COMP_COUNT_MAX * GMM_FV_ELEMENT_COUNT_MAX * GMM_MEAN_VALUE_SIZE, Gna2StatusGmmBadMeanSetoff);
        Expect::MultiplicityOf(MeanSetOffsetSize, GMM_MEM_ALIGNMENT);
        Expect::InRange(VarSetOffsetSize, GMM_FV_ELEMENT_COUNT_MULTIPLE_OF,
            GMM_MIXTURE_COMP_COUNT_MAX * GMM_FV_ELEMENT_COUNT_MAX * GMM_COVARIANCE_SIZE_MAX, Gna2StatusGmmBadVarSetoff);
        Expect::MultiplicityOf(VarSetOffsetSize, GMM_MEM_ALIGNMENT);
        Expect::InRange(GaussConstSetOffsetSize, GMM_FV_ELEMENT_COUNT_MULTIPLE_OF,
            GMM_MIXTURE_COMP_COUNT_MAX * GMM_CONSTANTS_SIZE, Gna2StatusGmmBadGconstOffset);
        Expect::MultiplicityOf(GaussConstSetOffsetSize, GMM_MEM_ALIGNMENT);
    }
}

GmmLayer::GmmLayer(const nn_layer& layer, const BaseValidator& validatorIn) :
    Layer(layer, validatorIn, {}, BaseAddress()),
    Config((static_cast<gna_gmm_layer*>(layer.pLayerStruct))->config),
    Data((static_cast<gna_gmm_layer*>(layer.pLayerStruct))->data),
    Params{ Config, Input.at(GNA_DIM_W) },
    gmmKernels{ AccelerationDetector::GetKernelMap<GmmMaxMix>(KERNEL_GMM, Config.mode) },
    gmmActiveListKernels{ AccelerationDetector::GetKernelMap<GmmMaxMixActiveList>(
        KERNEL_GMM_AL, Config.mode)},
    gmmHiddenConfig{ Input.at(GNA_DIM_N), Input.at(GNA_DIM_W), Config.mixtureComponentCount, Params.MeanSetOffsetSize, Params.VarSetOffsetSize,
                    Params.GaussConstSetOffsetSize, Config.maximumScore, Config.stateCount, &Data, Input.Buffer, Output.Buffer }
{
    validate();

    Layer::ComputeHidden = [this](AccelerationMode accel, ExecutionConfig const & executionConfig)
                    {this->computeHidden(accel, executionConfig); };

    Layer::Compute = [this](LayerConfiguration &layerConfiguration, AccelerationMode accel, ExecutionConfig const & executionConfig)
                    {this->compute(layerConfiguration, accel, executionConfig); };
}

void GmmLayer::UpdateKernelConfigs(LayerConfiguration& layerConfiguration) const
{
    Layer::UpdateKernelConfigs(layerConfiguration);

    BaseAddress inputBuffer = Input;
    if (layerConfiguration.Buffers.count(InputComponent) > 0)
    {
        inputBuffer = layerConfiguration.Buffers[InputComponent];
        Input.ValidateBuffer(inputBuffer);
    }

    BaseAddress outputBuffer = Output;
    if (layerConfiguration.Buffers.count(OutputComponent) > 0)
    {
        outputBuffer = layerConfiguration.Buffers[OutputComponent];
        Output.ValidateBuffer(outputBuffer);
    }

    auto& configs = layerConfiguration.Configs;
    if(!configs.Gmm)
    {
        configs.Gmm = std::make_unique<GmmConfig>(gmmHiddenConfig);
    }
    if (layerConfiguration.ActList)
    {
        ValidateActiveList(layerConfiguration.ActList.get());
        configs.Gmm->stateCount = layerConfiguration.ActList->IndicesCount;
    }
    configs.Gmm->input = inputBuffer;
    configs.Gmm->output = outputBuffer;
}

void GmmLayer::computeHidden(AccelerationMode accel, ExecutionConfig const & executionConfig) const
{
    auto gmmConfig = GmmConfig{&gmmHiddenConfig, reinterpret_cast<uint8_t*>(executionConfig.Intermediate->d0)};

    gmmKernels.at(accel)(&gmmConfig);

    checkScoresSaturation(Config.stateCount, Input.at(GNA_DIM_N), Output.Buffer, Config.maximumScore, *executionConfig.SaturationCount);
}

void GmmLayer::compute(const LayerConfiguration& layerConfiguration, AccelerationMode accel, ExecutionConfig const & executionConfig) const
{
    auto gmmConfig = GmmConfig{layerConfiguration.Configs.Gmm.get(), reinterpret_cast<uint8_t*>(executionConfig.Intermediate->d0)};

    if (layerConfiguration.ActList)
    {
        gmmActiveListKernels.at(accel)(&gmmConfig, layerConfiguration.ActList->Indices);
    }
    else
    {
        gmmKernels.at(accel)(&gmmConfig);
    }

    checkScoresSaturation(gmmConfig.stateCount, Input.at(GNA_DIM_N), gmmConfig.output, Config.maximumScore, *executionConfig.SaturationCount);
}

void GmmLayer::ValidateActiveList(ActiveList const * const activeList) const
{
    if (activeList != nullptr)
    {
        Expect::InRange(activeList->IndicesCount, ui32_1, Config.stateCount, Gna2StatusActiveListIndicesInvalid);
    }
}

DataConfig GmmLayer::GetDataMode() const
{
    auto constMode = GNA_INT32; // GMM_CONSTANTS_SIZE
    auto covarianceMode = (Config.mode == GNA_MAXMIX8) ? GNA_INT8 : GNA_INT16;
    // TODO:3: handle all parameters (means)
    return DataConfig(Input.Mode, covarianceMode, constMode, Output.Mode);
}

void GmmLayer::checkScoresSaturation(const uint32_t& nGMMs, const uint32_t& nVectors, const uint32_t * pS,
    const uint32_t& maximumScore, uint32_t& nSaturated) const
{
    for (auto i = uint32_t{0}; i < nGMMs * nVectors; i++)
    {
        if (maximumScore == *pS)
        {
            nSaturated++;
            return;
        }
        pS++;
    }
}

void GmmLayer::validate()
{
    Expect::Equal(Input.Mode.Size, GMM_FV_ELEMENT_SIZE, Gna2StatusXnnErrorInputBytes);
    Expect::InRange(Input.at(GNA_DIM_W), GMM_FV_ELEMENT_COUNT_MIN, GMM_FV_ELEMENT_COUNT_MAX, Gna2StatusBadFeatLength);
    Expect::Equal(Output.Mode.Value, GNA_INT32, Gna2StatusXnnErrorOutputBytes);
    Expect::InRange(Output.at(GNA_DIM_H), ui32_1, GMM_STATES_COUNT_MAX, Gna2StatusXnnErrorLyrCfg);
    Expect::InRange(Config.stateCount, ui32_1, GMM_STATES_COUNT_MAX, Gna2StatusGmmBadNumGmm);
    Expect::InRange(Config.mixtureComponentCount, ui32_1, GMM_MIXTURE_COMP_COUNT_MAX, Gna2StatusGmmBadMixCnum);
    Expect::InRange(Config.mode, GNA_MAXMIX8, GNA_MAXMIX16, Gna2StatusGmmBadMode);
    Expect::NotNull(Data.gaussianConstants);
    Expect::AlignedTo(Data.gaussianConstants, {GMM_MEM_ALIGNMENT, Gna2StatusGmmBadGconstAlign});
    Expect::NotNull(Data.meanValues);
    Expect::AlignedTo(Data.meanValues, {GMM_MEM_ALIGNMENT, Gna2StatusGmmBadMeanAlign});
    Expect::NotNull(Data.inverseCovariances.inverseCovariancesForMaxMix16);
    Expect::AlignedTo(Data.inverseCovariances.inverseCovariancesForMaxMix16, {GMM_MEM_ALIGNMENT, Gna2StatusGmmBadVarsAlign});
    Expect::InRange(Config.layout, GMM_LAYOUT_FLAT, GMM_LAYOUT_INTERLEAVED, Gna2StatusGmmCfgInvalidLayout);

    if (GMM_LAYOUT_INTERLEAVED != Config.layout)
    {
        validator->ValidateBufferIfSet(Data.gaussianConstants, Config.stateCount * Params.GaussConstSetOffsetSize);
        validator->ValidateBufferIfSet(Data.meanValues, Config.stateCount * Params.MeanSetOffsetSize);
        validator->ValidateBufferIfSet(Data.inverseCovariances.inverseCovariancesForMaxMix16, Config.stateCount * Params.VarSetOffsetSize);
    }
    // TODO:3: Verify if GMM_LAYOUT_INTERLEAVED vlaidation is ok, if not revert prev algorithm
}
