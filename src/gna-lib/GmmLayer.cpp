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
#include "Validator.h"

using namespace GNA;

GmmParams::GmmParams(const gna_gmm_config &config, const uint32_t inputElementCount)
{
    VarianceSize = config.mode + 1;
    if (GMM_LAYOUT_FLAT == config.layout)
    {
        MeanSetOffsetSize = config.mixtureComponentCount * inputElementCount * GMM_MEAN_VALUE_SIZE;
        VarSetOffsetSize = config.mixtureComponentCount * inputElementCount * VarianceSize;
        GaussConstSetOffsetSize = config.mixtureComponentCount * GMM_CONSTANTS_SIZE;
    }
    else if (GMM_LAYOUT_INTERLEAVED == config.layout)
    {
        MeanSetOffsetSize = config.mixtureComponentCount * inputElementCount * (VarianceSize + GMM_CONSTANTS_SIZE + GMM_MEAN_VALUE_SIZE);
        VarSetOffsetSize = MeanSetOffsetSize;
        GaussConstSetOffsetSize = MeanSetOffsetSize;
    }
}

GmmLayer::GmmLayer(const nn_layer *layer, const uint32_t inputVectorCount) :
    Layer(layer, inputVectorCount),
    Config((static_cast<gna_gmm_layer*>(layer->pLayerStruct))->config),
    Data((static_cast<gna_gmm_layer*>(layer->pLayerStruct))->data),
    Params(Config, Input.RowCount)
{
    validate();
}

void GmmLayer::ValidateActiveList(ActiveList const * const activeList)
{
    // TODO:KJ:implement active list for GMM
    if (activeList->Enabled)
    {
        Expect::InRange(activeList->IndicesCount, 1, Config.stateCount, GNA_INVALIDINDICES);
    }
}

void GmmLayer::validate()
{
    Expect::InRange(Input.RowCount, GMM_FV_ELEMENT_COUNT_MIN, GMM_FV_ELEMENT_COUNT_MAX, GNA_BADFEATLENGTH);
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