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

#pragma once

#include "ActiveList.h"
#include "Layer.h"
#include "XnnKernelApi.h"


namespace GNA
{

// GMM Advanced parameters that are model configuration dependent
struct GmmParams
{
    GmmParams(const gna_gmm_config &config, const uint32_t inputElementCount);

    uint32_t MeanSetOffsetSize;
    uint32_t VarSetOffsetSize;
    uint32_t GaussConstSetOffsetSize;
    uint32_t VarianceSize;
};

// GMM Calculation configuration
class GmmLayer : public Layer
{
public:
    GmmLayer(const nn_layer *layer, const BaseValidator& validator);
    virtual ~GmmLayer() = default;

    // TODO:3: Low priority: refactor components to Tensors
    const gna_gmm_config Config;
    const gna_gmm_data Data;
    const GmmParams Params;

    virtual void UpdateKernelConfigs(LayerConfiguration& layerConfiguration) const override;
    void ValidateActiveList(ActiveList const * const activeList) const;

    virtual DataConfig GetDataMode() const override;

private:
    virtual void computeHidden(AccelerationMode accel, ExecutionConfig const & executionConfig) const;
    virtual void compute(const LayerConfiguration& layerConfiguration, AccelerationMode accel, ExecutionConfig const & executionConfig) const;

    void checkScoresSaturation(const uint32_t& nGMMs, const uint32_t& nVectors, const uint32_t * pS,
        const uint32_t& maximumScore, uint32_t& nSaturated) const;
    inline void validate();

    const KernelMap<GmmMaxMix> gmmKernels;
    const KernelMap<GmmMaxMixActiveList> gmmActiveListKernels;

    GmmConfig gmmHiddenConfig;
};

}
