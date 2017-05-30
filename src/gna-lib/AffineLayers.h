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

#include "KernelArguments.h"
#include "Layer.h"
#include "LayerFunctions.h"

namespace GNA
{

class AffineLayer : public Layer
{
public:
    AffineLayer(const nn_layer *layer);
    virtual ~AffineLayer() = default;

    const std::unique_ptr<const AffineFunctionSingle> Affine;
    const std::unique_ptr<const ActivationFunction> Activation;

    virtual void UpdateKernelConfigs(LayerConfiguration& layerConfiguration) const override;

protected:
    virtual void computeHidden(acceleration accel, KernelBuffers *fvBuffers, uint32_t *saturationCount) const;
    virtual void computeHiddenPwl(acceleration accel, KernelBuffers *fvBuffers, uint32_t *saturationCount) const;
    virtual void computeConfig(const LayerConfiguration& layerConfiguration, acceleration accel, KernelBuffers *fvBuffers, uint32_t *saturationCount) const;
    virtual void computeConfigPwl(const LayerConfiguration& layerConfiguration, acceleration accel, KernelBuffers *fvBuffers, uint32_t *saturationCount) const;

    const std::map<const acceleration, const AffineKernel>& affineKernels;
    const std::map<const acceleration, const AffineActiveListKernel>& affineKernelsAl;

    AffineConfig affineHiddenConfig;
};

class AffineMultiBiasLayer : public Layer
{
public:
    AffineMultiBiasLayer(const nn_layer *layer);
    virtual ~AffineMultiBiasLayer() = default;
    virtual void UpdateKernelConfigs(LayerConfiguration& layerConfiguration) const override;

    const std::unique_ptr<const AffineFunctionMulti> Affine;
    const std::unique_ptr<const ActivationFunction> Activation;

private:
    virtual void computeHidden(acceleration accel, KernelBuffers *fvBuffers, uint32_t *saturationCount) const;
    virtual void computeHiddenPwl(acceleration accel, KernelBuffers *fvBuffers, uint32_t *saturationCount) const;
    virtual void computeConfig(const LayerConfiguration& layerConfiguration, acceleration accel, KernelBuffers *fvBuffers, uint32_t *saturationCount) const;
    virtual void computeConfigPwl(const LayerConfiguration& layerConfiguration, acceleration accel, KernelBuffers *fvBuffers, uint32_t *saturationCount) const;

    const std::map<const acceleration, const AffineKernel>& multibiasKernels;
    AffineConfig affineHiddenConfig;
};

class AffineDiagonalLayer : public AffineLayer
{
public:
    AffineDiagonalLayer(const nn_layer *layer);
    virtual ~AffineDiagonalLayer() = default;
    virtual void UpdateKernelConfigs(LayerConfiguration& layerConfiguration) const override;

private:
    virtual void computeHidden(acceleration accel, KernelBuffers *fvBuffers, uint32_t *saturationCount) const override;
    virtual void computeHiddenPwl(acceleration accel, KernelBuffers *fvBuffers, uint32_t *saturationCount) const override;
    virtual void computeConfig(const LayerConfiguration& layerConfiguration, acceleration accel, KernelBuffers *fvBuffers, uint32_t *saturationCount) const override;
    virtual void computeConfigPwl(const LayerConfiguration& layerConfiguration, acceleration accel, KernelBuffers *fvBuffers, uint32_t *saturationCount) const override;

    const std::map<const acceleration, const AffineKernel>& diagonalKernels;
};

}
