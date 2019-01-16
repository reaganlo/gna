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

#include "KernelArguments.h"
#include "pwl.h"

BaseConfig::BaseConfig(const BufferMap& buffers) :
    BufferMap{buffers}
{
    setPointers();
}

BaseConfig::BaseConfig(const BaseAddress& inputBuffer, const BaseAddress& outputBuffer) :
    BufferMap{inputBuffer, outputBuffer}
{
    setPointers();
}

void BaseConfig::Update(const BufferMap& buffers)
{
    for (const auto& buffer : buffers)
    {
       operator[](buffer.first) = buffer.second;
    }
    setPointers();
}

void BaseConfig::setPointers()
{
    if (count(InputComponent))
        Inputs = at(InputComponent);
    if (count(OutputComponent))
        Outputs = at(OutputComponent);
}

ActivationConfig::ActivationConfig(uint32_t elementCount, GNA::PwlCached const * kernel) :
    ElementCount{elementCount},
    Kernel{kernel}
{}

PoolingConfig2D::PoolingConfig2D(nn_layer_pool2d const & config) :
    Pooling{config}
{}

ConvolutionConfig2D::ConvolutionConfig2D(gna_3d_dimensions const & inputDimensions,
        gna_convolution_func const & config) :
    InputDimensions{ inputDimensions },
    Convolution{config}
{}

AffineConfig::AffineConfig(int16_t const * inputIn, int32_t * const outputIn,
    AffineConfig const * const source) :
    AffineConfig{*source}
{
    input = inputIn;
    output = outputIn;
}

AffineConfig::AffineConfig(AffineConfig const * const source, uint32_t * saturationCountIn,
    KernelBuffers * fvBuffersIn) :
    AffineConfig{*source}
{
    saturationCount = saturationCountIn;
    fvBuffers = fvBuffersIn;
}

AffineConfig::AffineConfig(uint32_t const outputElementCountIn, uint32_t const inputVectorCountIn,
    uint32_t const inputElementCountIn, int16_t const * inputIn, int32_t * const outputIn,
    void const * weightsIn, void const * biases, void const * multiBiasIn,
    uint32_t const multiBiasVectorCountIn) :
    outputElementCount{outputElementCountIn},
    inputVectorCount{inputVectorCountIn},
    inputElementCount{inputElementCountIn},
    input{inputIn},
    output{outputIn},
    saturationCount{nullptr},
    fvBuffers{nullptr},
    weights1B{static_cast<int8_t const *>(weightsIn)},
    biasesCompound{static_cast<nn_bias_c const *>(biases)},
    multiBias{static_cast<nn_bias_s const *>(multiBiasIn)},
    multiBiasVectorCount{multiBiasVectorCountIn}
{}

AffineConfig::AffineConfig(uint32_t const outputElementCountIn, uint32_t const inputVectorCountIn,
    uint32_t const inputElementCountIn, int16_t const * inputIn, int32_t * const outputIn,
    void const * weightsIn, void const * biases, void const * multiBiasIn,
    uint32_t const multiBiasVectorCountIn,
    const uint32_t bytesPerBiasIn) :
    outputElementCount{outputElementCountIn},
    inputVectorCount{inputVectorCountIn},
    inputElementCount{inputElementCountIn},
    input{inputIn},
    output{outputIn},
    saturationCount{nullptr},
    fvBuffers{nullptr},
    weights1B{static_cast<int8_t const *>(weightsIn)},
    biasesCompound{static_cast<nn_bias_c const *>(biases)},
    multiBias{static_cast<nn_bias_s const *>(multiBiasIn)},
    multiBiasVectorCount{multiBiasVectorCountIn},
    bytesPerBias{bytesPerBiasIn}
{}

AffineConfigAl::AffineConfigAl(uint32_t const * indicesIn, uint32_t const countIn) :
    indices{indicesIn},
    count{countIn}
{}

RecurrentConfig::RecurrentConfig(RecurrentConfig const * const source, uint32_t * saturationCountIn) :
    RecurrentConfig{*source}
{
    saturationCount = saturationCountIn;
}

RecurrentConfig::RecurrentConfig(uint32_t const outputElementCountIn,
    uint32_t const inputVectorCountIn, uint32_t const inputElementCountIn, int16_t const * inputIn,
    int16_t * const feedbackBufferIn, int32_t * const outputIn, int16_t * outputActivatedIn,
    void const * weightsIn, void const * biases, ActivationConfig const & pwl) :
    outputElementCount{outputElementCountIn},
    inputVectorCount{inputVectorCountIn},
    inputElementCount{inputElementCountIn},
    input{inputIn},
    feedbackBuffer{feedbackBufferIn},
    output{outputIn},
    saturationCount{nullptr},
    weights1B{static_cast<int8_t const *>(weightsIn)},
    biasesCompound{static_cast<nn_bias_c const *>(biases)},
    activation{pwl, BaseConfig(outputIn, outputActivatedIn)}
{
}

RecurrentConfig::RecurrentConfig(uint32_t const outputElementCountIn,
    uint32_t const inputVectorCountIn, uint32_t const inputElementCountIn, int16_t const * inputIn,
    int16_t * const feedbackBufferIn, int32_t * const outputIn, int16_t * outputActivatedIn,
    void const * weightsIn, void const * biases, uint32_t bytesPerBiasIn, uint32_t bytesPerOutputIn,
    ActivationConfig const & pwl) :
    outputElementCount{outputElementCountIn},
    inputVectorCount{inputVectorCountIn},
    inputElementCount{inputElementCountIn},
    input{inputIn},
    feedbackBuffer{feedbackBufferIn},
    output{outputIn},
    saturationCount{nullptr},
    bytesPerBias{bytesPerBiasIn},
    bytesPerOutput{bytesPerOutputIn},
    weights1B{static_cast<int8_t const *>(weightsIn)},
    biasesCompound{static_cast<nn_bias_c const *>(biases)},
    activation{pwl, BaseConfig(outputIn, outputActivatedIn)}
{}

TransposeConfig::TransposeConfig(uint32_t rowCountIn, uint32_t columntCountIn,
    int16_t const * const inputIn, int16_t * const outputIn) :
    rowCount{rowCountIn},
    columnCount{columntCountIn},
    input{inputIn},
    output{outputIn}
{}

CopyConfig::CopyConfig(uint32_t rowCountIn, uint32_t columntCountIn, uint32_t inputColumnCountIn,
    uint32_t outputColumnCountIn, int16_t const * const inputIn, int16_t * const outputIn) :
    rowCount{rowCountIn},
    columnCount{columntCountIn},
    inputColumnCount{inputColumnCountIn},
    outputColumnCount{outputColumnCountIn},
    input{inputIn},
    output{outputIn}
{}

ConvolutionConfig::ConvolutionConfig(ConvolutionConfig const * const source,
    int16_t const * const inputsIn, int32_t * const outputsIn) :
    ConvolutionConfig{*source}
{
    inputs = inputsIn;
    convolutedOutputs = outputsIn;
}

ConvolutionConfig::ConvolutionConfig(ConvolutionConfig const * const source,
    uint32_t * const saturationCountIn) :
    ConvolutionConfig{*source}
{
    saturationCount = saturationCountIn;
}

ConvolutionConfig::ConvolutionConfig(uint32_t const inputBandStrideIn,
    uint32_t const FilterOutputCountIn, uint32_t const FilterCountIn,
    uint32_t const FilterCoefficientCountIn, int16_t const * const inputsIn,
    int16_t const * const filtersIn, nn_bias_s const * const biasesIn, int32_t * const outputsIn) :
    inputBandStride{inputBandStrideIn},
    filterOutputCount{FilterOutputCountIn},
    filterCount{FilterCountIn},
    filterCoefficientCount{FilterCoefficientCountIn},
    inputs{inputsIn},
    filters{filtersIn},
    biases{biasesIn},
    convolutedOutputs{outputsIn},
    saturationCount{nullptr}
{}

ConvolutionConfig::ConvolutionConfig(uint32_t const inputBandStrideIn,
    uint32_t const FilterOutputCountIn, uint32_t const FilterCountIn,
    uint32_t const FilterCoefficientCountIn, int16_t const * const inputsIn,
    int16_t const * const filtersIn, nn_bias_s const * const biasesIn,
    int32_t * const outputsIn, uint32_t bytesPerBiasIn, uint32_t bytesPerFilterIn) :
    inputBandStride{inputBandStrideIn},
    filterOutputCount{FilterOutputCountIn},
    filterCount{FilterCountIn},
    filterCoefficientCount{FilterCoefficientCountIn},
    bytesPerBias{bytesPerBiasIn},
    bytesPerFilter{bytesPerFilterIn},
    inputs{inputsIn},
    filters{filtersIn},
    biases{biasesIn},
    convolutedOutputs{outputsIn},
    saturationCount{nullptr}

{}

PoolingConfig::PoolingConfig(PoolingConfig const * const source, int64_t * const bufferIn) :
    PoolingConfig{*source}
{
    buffer = bufferIn;
}

PoolingConfig::PoolingConfig(nn_pool_type const typeIn, uint32_t const sizeIn, uint32_t const stepIn) :
    type{typeIn},
    size{sizeIn},
    step{stepIn},
    buffer{nullptr}
{}

GmmConfig::GmmConfig(GmmConfig const * const source, uint8_t *inputScratchPadIn) :
    GmmConfig{*source}
{
    inputScratchPad = inputScratchPadIn;
}

GmmConfig::GmmConfig(uint32_t const inputVectorCountIn, uint32_t const inputElementCountIn,
    uint32_t const mixCountIn, uint32_t const meanSetOffsetSizeIn,
    uint32_t const varSetOffsetSizeIn, uint32_t const gaussConstSetOffsetSizeIn,
    uint32_t const maxScoreIn, uint32_t const stateCountIn, gna_gmm_data const * const dataIn,
    uint8_t const * const inputIn, uint32_t * const outputIn) :
    inputVectorCount{inputVectorCountIn},
    inputElementCount{inputElementCountIn},
    mixtureComponentCount{mixCountIn},
    meanSetOffsetSize{meanSetOffsetSizeIn},
    varSetOffsetSize{varSetOffsetSizeIn},
    gaussConstSetOffsetSize{gaussConstSetOffsetSizeIn},
    maximumScore{maxScoreIn},
    stateCount{stateCountIn},
    data{dataIn},
    input{inputIn},
    inputScratchPad{nullptr},
    output{outputIn}
{}
