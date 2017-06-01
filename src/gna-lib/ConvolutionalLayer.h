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

struct FiltersConfig
{
    FiltersConfig(const nn_layer_conv *sourceLayer, const uint32_t inputElementCount);
    ~FiltersConfig() = default;

    const uint32_t Count;
    const uint32_t CoefficientCount;  // Actual filter size, including 0-padding if necessary.
    const int16_t * const Data;       // Filters stored one after the other.
    const Bias Biases;
};

// feature maps definition - used for filter stride calculation
struct FeatureMaps
{
    FeatureMaps(const nn_layer_conv *sourceLayer);
    ~FeatureMaps() = default;

    const uint32_t Count;
    const uint32_t RowCount;
    const uint32_t ColumnCount;
    const uint32_t Stride;   // Size of convolution filter stride (in elements).
};

struct ConvolutionFunction
{
    ConvolutionFunction(const nn_layer_conv *sourceLayer, const uint32_t inputElementCount);
    ~ConvolutionFunction() = default;

    const FiltersConfig Filters;
    const FeatureMaps FeatureMaps;
    const uint32_t OutputElementsCount;   // Number of outputs after convolution per filter
};

struct PoolingFunction
{
    PoolingFunction(const nn_layer_conv *sourceLayer);
    ~PoolingFunction() = default;

    const intel_pool_type_t Type;
    const uint32_t Size;
    const uint32_t Stride;
};

// TODO: refactor: consider extracting kernel stuff into [LayerType]Kernel class and attach to layer as member

class CnnLayer : public Layer
{
public:
    CnnLayer(nn_layer const * const layer);
    virtual ~CnnLayer() = default;
    virtual void UpdateKernelConfigs(LayerConfiguration& layerConfiguration) const override;

    const std::unique_ptr<const ActivationFunction> Activation;
    const ConvolutionFunction Convolution;
    const PoolingFunction Pooling;

private:
    void computeConfigPool(LayerConfiguration& layerConfiguration, acceleration accel, KernelBuffers *fvBuffers, uint32_t *saturationCount) const;
    void computeHiddenPool(acceleration accel, KernelBuffers *fvBuffers, uint32_t *saturationCount) const;
    void computeHidden(acceleration accel, KernelBuffers *fvBuffers, uint32_t *saturationCount) const;
    void computeHiddenPwl(acceleration accel, KernelBuffers *fvBuffers, uint32_t *saturationCount) const;
    void computeConfig(const LayerConfiguration& layerConfiguration, acceleration accel, KernelBuffers *fvBuffers, uint32_t *saturationCount) const;
    void computeConfigPwl(const LayerConfiguration& layerConfiguration, acceleration accel, KernelBuffers *fvBuffers, uint32_t *saturationCount) const;

    const std::map<const acceleration, const ConvolutionKernel>& filterKernels;
    const std::map<const acceleration, const ConvolutionPoolingKernel>& poolingKernels;

    const ConvolutionConfig convolutionHiddenConfig;
    const PoolingConfig poolingHiddenConfig;
};

}
