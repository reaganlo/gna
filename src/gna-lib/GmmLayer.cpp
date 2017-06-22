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

#include "GmmLayer.h"

#include "AccelerationDetector.h"
#include "LayerConfiguration.h"
#include "Validator.h"

using namespace GNA;

GmmParams::GmmParams(const gna_gmm_config &config, const uint32_t inputElementCount)
{
    VarianceSize = (GNA_MAXMIX16 == config.mode) ? sizeof(uint16_t) : sizeof(uint8_t);
    
    MeanSetOffsetSize = config.mixtureComponentCount * inputElementCount * GMM_MEAN_VALUE_SIZE;
    VarSetOffsetSize = config.mixtureComponentCount * inputElementCount * VarianceSize;
    GaussConstSetOffsetSize = ALIGN(config.mixtureComponentCount, 2) * GMM_CONSTANTS_SIZE;
    if (GMM_LAYOUT_INTERLEAVED == config.layout)
    {
        MeanSetOffsetSize = MeanSetOffsetSize + VarSetOffsetSize + GaussConstSetOffsetSize;
        VarSetOffsetSize = MeanSetOffsetSize;
        GaussConstSetOffsetSize = MeanSetOffsetSize;
    }
    Expect::InRange(MeanSetOffsetSize, GMM_FV_ELEMENT_COUNT_MULTIPLE_OF,
        GMM_MIXTURE_COMP_COUNT_MAX * GMM_FV_ELEMENT_COUNT_MAX * GMM_MEAN_VALUE_SIZE, GMM_BADMEANSETOFF);
    Expect::MultiplicityOf(MeanSetOffsetSize, GMM_MEM_ALIGNMENT);
    Expect::InRange(VarSetOffsetSize, GMM_FV_ELEMENT_COUNT_MULTIPLE_OF,
        GMM_MIXTURE_COMP_COUNT_MAX * GMM_FV_ELEMENT_COUNT_MAX * GMM_COVARIANCE_SIZE_MAX, GMM_BADVARSETOFF);
    Expect::MultiplicityOf(VarSetOffsetSize, GMM_MEM_ALIGNMENT);
    Expect::InRange(GaussConstSetOffsetSize, GMM_FV_ELEMENT_COUNT_MULTIPLE_OF,
        GMM_MIXTURE_COMP_COUNT_MAX * GMM_CONSTANTS_SIZE, GMM_BADGCONSTOFFSET);
    Expect::MultiplicityOf(GaussConstSetOffsetSize, GMM_MEM_ALIGNMENT);
}

GmmLayer::GmmLayer(const nn_layer *layer) :
    Layer(layer),
    Config((static_cast<gna_gmm_layer*>(layer->pLayerStruct))->config),
    Data((static_cast<gna_gmm_layer*>(layer->pLayerStruct))->data),
    Params{ Config, Input.ElementCount },
    gmmKernels{ AccelerationDetector::GetKernelMap<GmmMaxMix>(Config.mode) },
    gmmActiveListKernels{ AccelerationDetector::GetKernelMap<GmmMaxMixActiveList>(Config.mode) },
    gmmHiddenConfig{ Input.VectorCount, Input.ElementCount, Config.mixtureComponentCount, Params.MeanSetOffsetSize, Params.VarSetOffsetSize,
                    Params.GaussConstSetOffsetSize, Config.maximumScore, Config.stateCount, &Data, Input.Buffer, Output.Buffer }
{
    validate();

    Layer::ComputeHidden = [this](acceleration accel, KernelBuffers *fvBuffers, uint32_t *saturationCount) 
                    {this->computeHidden(accel, fvBuffers, saturationCount); };

    Layer::ComputeConfig = [this](LayerConfiguration &layerConfiguration, acceleration accel, KernelBuffers *fvBuffers, uint32_t *saturationCount) 
                    {this->computeConfig(layerConfiguration, accel, fvBuffers, saturationCount); };
}

void GmmLayer::UpdateKernelConfigs(LayerConfiguration& layerConfiguration) const
{
    auto inputBuffer = layerConfiguration.InputBuffer
        ? *layerConfiguration.InputBuffer : Input.Buffer;

    auto outputBuffer = layerConfiguration.OutputBuffer
        ? *layerConfiguration.OutputBuffer : Output.Buffer;

    auto& configs = layerConfiguration.Configs;

    if(!configs.Gmm)
        configs.Gmm = std::make_unique<GmmConfig>(gmmHiddenConfig);
    if (layerConfiguration.ActiveList)
    {
        ValidateActiveList(layerConfiguration.ActiveList.get());
        configs.Gmm->stateCount = layerConfiguration.ActiveList->IndicesCount;
    }
    configs.Gmm->input = inputBuffer;
    configs.Gmm->output = outputBuffer;
}

void GmmLayer::computeHidden(acceleration accel, KernelBuffers *fvBuffers, uint32_t *saturationCount) const
{
    auto gmmConfig = GmmConfig{&gmmHiddenConfig, reinterpret_cast<uint8_t*>(fvBuffers->d0)};

    gmmKernels.at(accel)(&gmmConfig);

    checkScoresSaturation(Config.stateCount, Input.VectorCount, Output.Buffer, Config.maximumScore, *saturationCount);
}

void GmmLayer::computeConfig(const LayerConfiguration& layerConfiguration, acceleration accel, KernelBuffers *fvBuffers, uint32_t *saturationCount) const
{
    auto gmmConfig = GmmConfig{layerConfiguration.Configs.Gmm.get(), reinterpret_cast<uint8_t*>(fvBuffers->d0)};

    if (layerConfiguration.ActiveList)
    {
        gmmActiveListKernels.at(accel)(&gmmConfig, layerConfiguration.ActiveList->Indices);
    }
    else
    {
        gmmKernels.at(accel)(&gmmConfig);
    }

    checkScoresSaturation(gmmConfig.stateCount, Input.VectorCount, gmmConfig.output, Config.maximumScore, *saturationCount);
}

void GmmLayer::ValidateActiveList(ActiveList const * const activeList) const
{
    if (activeList)
    {
        Expect::InRange(activeList->IndicesCount, 1, Config.stateCount, GNA_INVALIDINDICES);
    }
}

void GmmLayer::checkScoresSaturation(const uint32_t& nGMMs, const uint32_t& nVectors, const uint32_t * pS,
    const uint32_t& maximumScore, uint32_t& nSaturated) const
{
    for (auto i = 0ui32; i < nGMMs * nVectors; i++)
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
    Output.SetOutputMode(LayerOutput::NonActivatedOutput, GMM_SCORE_SIZE);
    Expect::InRange(Input.ElementCount, GMM_FV_ELEMENT_COUNT_MIN, GMM_FV_ELEMENT_COUNT_MAX, GNA_BADFEATLENGTH);
    Expect::InRange(Config.stateCount, 1, GMM_STATES_COUNT_MAX, GMM_BADNUMGMM);
    Expect::InRange(Config.mixtureComponentCount, 1, GMM_MIXTURE_COMP_COUNT_MAX, GMM_BADMIXCNUM);
    Expect::InRange(Config.mode, 0, GNA_MAXMIX16, GMM_BADMODE);
    Expect::NotNull(Data.gaussianConstants);
    Expect::AlignedTo(Data.gaussianConstants, GMM_MEM_ALIGNMENT, GMM_BADGCONSTALIGN);
    Expect::NotNull(Data.meanValues);
    Expect::AlignedTo(Data.meanValues, GMM_MEM_ALIGNMENT, GMM_BADMEANALIGN);
    Expect::NotNull(Data.inverseCovariancesForMaxMix16);
    Expect::AlignedTo(Data.inverseCovariancesForMaxMix16, GMM_MEM_ALIGNMENT, GMM_BADVARSALIGN);
    Expect::InRange(Config.layout, GMM_LAYOUT_FLAT, GMM_LAYOUT_INTERLEAVED, GMM_CFG_INVALID_LAYOUT);
}
