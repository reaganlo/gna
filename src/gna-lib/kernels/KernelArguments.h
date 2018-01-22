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

#include "..\common.h"
#include "gna-api-types-gmm.h"

struct PwlOutputConfig
{
    PwlOutputConfig() :
        elementCount{}
    {}
    PwlOutputConfig(PwlOutputConfig const * const source, uint32_t * saturationCountIn) :
        PwlOutputConfig{*source}
    {
        saturationCount = saturationCountIn;
    }
    PwlOutputConfig(uint32_t elementCountIn, int32_t* inputIn, int16_t* outputIn) :
        elementCount{elementCountIn},
        input{inputIn},
        output{outputIn},
        saturationCount{nullptr}
    {}

    uint32_t elementCount;
    int32_t * input;
    int16_t * output;
    uint32_t * saturationCount;
};

// TODO: refactor: consider splitting into run config and basic constant config
struct AffineConfig
{
    AffineConfig(AffineConfig const * const source, int16_t const * inputIn, int32_t * const outputIn) :
        AffineConfig{*source}
    {
        input = inputIn;
        output = outputIn;
    }
    AffineConfig(AffineConfig const * const source, uint32_t * saturationCountIn, KernelBuffers * fvBuffersIn) :
        AffineConfig{*source}
    {
        saturationCount = saturationCountIn;
        fvBuffers = fvBuffersIn;
    }
    AffineConfig(uint32_t const outputElementCountIn, uint32_t const inputVectorCountIn, 
        uint32_t const inputElementCountIn, int16_t const * inputIn, int32_t * const outputIn, void const * weightsIn,
        void const * biases, nn_bias_s const * multiBiasIn, uint32_t const multiBiasVectorCountIn) :
        outputElementCount{outputElementCountIn},
        inputVectorCount{inputVectorCountIn},
        inputElementCount{inputElementCountIn},
        input{inputIn},
        output{outputIn},
        saturationCount{nullptr},
        fvBuffers{nullptr},
        weights1B{static_cast<int8_t const *>(weightsIn)},
        biasesCompound{static_cast<nn_bias_c const *>(biases)},
        multiBias{multiBiasIn},
        multiBiasVectorCount{multiBiasVectorCountIn}
    {}

    uint32_t const outputElementCount;  // M - out rows
    uint32_t const inputVectorCount;    // N - columns
    uint32_t const inputElementCount;   // K - rows
    int16_t const * input;              // I - (interleaved) [K;N]
    int32_t * output;                   // O - [M;N]
    uint32_t * saturationCount;
    KernelBuffers const * fvBuffers;

    union
    {
    int8_t const * const weights1B;     // W - [M;K]
    int16_t const * const weights2B;    // W - [M;K]
    } ;
    union
    {
    nn_bias_c const * const weightScaleFactors; // [M] Scaling factors for 1B weights or NULL for 2B weights. 
    nn_bias_c const * const biasesCompound;     // B - [M]
    nn_bias_s const * const biasesSimple;       // B - [M]
    };
    nn_bias_s const * const multiBias;
    uint32_t const multiBiasVectorCount;
};

struct AffineConfigAl
{
    AffineConfigAl(uint32_t const * indicesIn, uint32_t const countIn) :
        indices{indicesIn},
        count{countIn}
    {}

    uint32_t const * const indices; // AL [L]
    uint32_t const count;           // L
};

struct RecurrentConfig
{
    RecurrentConfig(RecurrentConfig const * const source, uint32_t * saturationCountIn) :
        RecurrentConfig{*source}
    {
        saturationCount = saturationCountIn;
        pwlOutputConfig.saturationCount = saturationCountIn;
    }
    RecurrentConfig(
        uint32_t const outputElementCountIn, uint32_t const inputVectorCountIn, uint32_t const inputElementCountIn,
        int16_t const * inputIn, int16_t * const feedbackBufferIn, int32_t * const outputIn, 
        int16_t * outputActivatedIn, void const * weightsIn, void const * biases) :
        outputElementCount{outputElementCountIn},
        inputVectorCount{inputVectorCountIn},
        inputElementCount{inputElementCountIn},
        input{inputIn},
        feedbackBuffer{feedbackBufferIn},
        output{outputIn},
        saturationCount{nullptr},
        weights1B{static_cast<int8_t const *>(weightsIn)},
        biasesCompound{static_cast<nn_bias_c const *>(biases)},
        pwlOutputConfig{outputElementCount, outputIn, outputActivatedIn}
    {}

    uint32_t const outputElementCount;      // M - cols
    uint32_t const inputVectorCount;        // N - rows
    uint32_t const inputElementCount;       // K - cols
    int16_t const * input;                  // I - (flat) [N;K]
    int16_t * feedbackBuffer;               // (flat) [N,M]
    int32_t * output;                       // O1 - [N,M]
    uint32_t * saturationCount;
    union
    {
    int8_t const * const weights1B;         // W - [M,K+M]
    int16_t const * const weights2B;        // W - [M,K+M]
    } ;
    union
    {
    nn_bias_c const * const biasesCompound; // B - [M]
    nn_bias_s const * const biasesSimple;   // B - [M]
    };
    PwlOutputConfig pwlOutputConfig;
};

struct TransposeConfig
{
    TransposeConfig(uint32_t rowCountIn, uint32_t columntCountIn, int16_t const * const inputIn,
        int16_t * const outputIn) :
        rowCount{rowCountIn},
        columnCount{columntCountIn},
        input{inputIn},
        output{outputIn}
    {}

    uint32_t const rowCount;
    uint32_t const columnCount;
    int16_t const * input;
    int16_t * output;
};

struct CopyConfig
{
    CopyConfig(uint32_t rowCountIn, uint32_t columntCountIn, uint32_t inputColumnCountIn, uint32_t outputColumnCountIn,
        int16_t const * const inputIn, int16_t * const outputIn) :
        rowCount{rowCountIn},
        columnCount{columntCountIn},
        inputColumnCount{inputColumnCountIn},
        outputColumnCount{outputColumnCountIn},
        input{inputIn},
        output{outputIn}
    {}

    uint32_t const rowCount;
    uint32_t const columnCount;
    uint32_t const inputColumnCount;
    uint32_t const outputColumnCount;
    int16_t const * input;
    int16_t * output;
};

struct ConvolutionConfig
{
    ConvolutionConfig(ConvolutionConfig const * const source, int16_t const * const inputsIn,
        int32_t * const outputsIn) :
        ConvolutionConfig{*source}
    {
        inputs = inputsIn;
        convolutedOutputs = outputsIn;
    }
    ConvolutionConfig(ConvolutionConfig const * const source, uint32_t * const saturationCountIn) :
        ConvolutionConfig{*source}
    {
        saturationCount = saturationCountIn;
    }
    ConvolutionConfig(uint32_t const inputBandStrideIn, uint32_t const FilterOutputCountIn, uint32_t const FilterCountIn,
        uint32_t const FilterCoefficientCountIn, int16_t const * const inputsIn, int16_t const * const filtersIn,
        nn_bias_s const * const biasesIn, int32_t * const outputsIn) :
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

    uint32_t const inputBandStride;
    uint32_t const filterOutputCount;
    uint32_t const filterCount;
    uint32_t const filterCoefficientCount;

    int16_t const * inputs;
    int16_t const * const filters;
    nn_bias_s const * const biases;

    union
    {
        int32_t * convolutedOutputs;
        int16_t * const pooledOutputs;
    };
    uint32_t * saturationCount;
};

struct PoolingConfig
{
    PoolingConfig(PoolingConfig const * const source, int64_t * const bufferIn) :
        PoolingConfig{*source}
    {
        buffer = bufferIn;
    }
    PoolingConfig(nn_pool_type const typeIn, uint32_t const sizeIn, uint32_t const stepIn) :
        type{typeIn},
        size{sizeIn},
        step{stepIn},
        buffer{nullptr}
    {}

    nn_pool_type const type;
    uint32_t const size;
    uint32_t const step;
    int64_t * buffer;
};

struct GmmConfig
{
    GmmConfig(GmmConfig const * const source, uint8_t *inputScratchPadIn) :
        GmmConfig{*source}
    {
        inputScratchPad = inputScratchPadIn;
    }
    GmmConfig(uint32_t const inputVectorCountIn, uint32_t const inputElementCountIn, uint32_t const mixCountIn,
        uint32_t const meanSetOffsetSizeIn, uint32_t const varSetOffsetSizeIn, uint32_t const gaussConstSetOffsetSizeIn,
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

    uint32_t const inputVectorCount;
    uint32_t const inputElementCount;
    uint32_t const mixtureComponentCount;
    uint32_t const meanSetOffsetSize;
    uint32_t const varSetOffsetSize;
    uint32_t const gaussConstSetOffsetSize;
    uint32_t const maximumScore;
    uint32_t stateCount;
    gna_gmm_data const * const data;
    uint8_t const * input;
    uint8_t const * inputScratchPad;
    uint32_t * output;
};
