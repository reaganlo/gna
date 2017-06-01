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

#include "Layer.h"
#include "LayerFunctions.h"

namespace GNA
{

class AffineBaseLayer : public Layer
{
public:
    virtual ~AffineBaseLayer() = default;

    const std::unique_ptr<const AffineFunction> Affine;
    const std::unique_ptr<const ActivationFunction> Activation;

protected:
    AffineBaseLayer(const nn_layer *layer);

    virtual void UpdateKernelConfigs(LayerConfiguration& layerConfiguration) const override;

private:
    void computeHidden(acceleration accel, KernelBuffers *fvBuffers, uint32_t *saturationCount) const;
    void computeHiddenPwl(acceleration accel, KernelBuffers *fvBuffers, uint32_t *saturationCount) const;
    void computeConfig(const LayerConfiguration& layerConfiguration, acceleration accel, KernelBuffers *fvBuffers,
        uint32_t *saturationCount) const;
    void computeConfigPwl(const LayerConfiguration& layerConfiguration, acceleration accel, KernelBuffers *fvBuffers,
        uint32_t *saturationCount) const;
};

class AffineLayer : public AffineBaseLayer
{
public:
    AffineLayer(const nn_layer *layer);
    virtual ~AffineLayer() = default;

    virtual void UpdateKernelConfigs(LayerConfiguration& layerConfiguration) const override;
};

class AffineDiagonalLayer : public AffineBaseLayer
{
public:
    AffineDiagonalLayer(const nn_layer *layer);
    virtual ~AffineDiagonalLayer() = default;
};

}
