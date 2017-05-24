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
    PwlOutputConfig(uint32_t rowFirstIn, uint32_t rowLastIn, uint32_t columnFirstIn, uint32_t columnLastIn,
        uint32_t columnCountIn, uint32_t * saturationCountIn, int16_t* outputIn) :
        rowFirst{rowFirstIn},
        rowLast{rowLastIn},
        columnFirst{columnFirstIn},
        columnLast{columnLastIn},
        columnCount{columnCountIn},
        output{outputIn},
        saturationCount{saturationCountIn}
    {}

    uint32_t rowFirst;
    uint32_t rowLast;
    uint32_t const columnFirst;
    uint32_t const columnLast;
    uint32_t const columnCount;
    int16_t * output;
    uint32_t * saturationCount;
};


// constant configuration for given layer 
struct PwlBaseConfig
{
    PwlBaseConfig(int32_t * const inputIn, nn_pwl_seg const * const segmentsIn, uint32_t segmentCountIn) :
        input{inputIn},
        segments{segmentsIn},
        segmentCount{segmentCountIn}
    {}

    int32_t const * const input;
    nn_pwl_seg const * const segments;
    uint32_t const segmentCount;
};

// Function pointer for apply PWL for single input-output
typedef void (*PwlApplySingle)(PwlCached const * const pwl, int32_t I, int16_t * const output,
    uint32_t * const saturationCount);

// Function pointer for apply PWL for all inputs-outputs
typedef void (*PwlApplyAll)(PwlCached const * const pwl, PwlOutputConfig const * const outputConfig);

struct AffineConfig
{
    AffineConfig(uint32_t const outputElementCountIn, uint32_t const inputVectorCountIn, 
        uint32_t const inputElementCountIn, int16_t const * inputIn, int32_t * const outputIn, 
        uint32_t * saturationCountIn, KernelBuffers * fvBuffersIn, void const * weightsIn,
        void const * biases, nn_bias_s const * multiBiasIn, uint32_t const multiBiasVectorCountIn) :
        outputElementCount{outputElementCountIn},
        inputVectorCount{inputVectorCountIn},
        inputElementCount{inputElementCountIn},
        input{inputIn},
        output{outputIn},
        saturationCount{saturationCountIn},
        fvBuffers{fvBuffersIn},
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
    RecurrentConfig(
        uint32_t const outputElementCountIn, uint32_t const inputVectorCountIn, uint32_t const inputElementCountIn,
        int16_t const * inputIn, int16_t * const feedbackBufferIn, int32_t * const outputIn, 
        int16_t * outputActivatedIn, uint32_t * saturationCountIn, void const * weightsIn, void const * biases) :
        outputElementCount{outputElementCountIn},
        inputVectorCount{inputVectorCountIn},
        inputElementCount{inputElementCountIn},
        input{inputIn},
        feedbackBuffer{feedbackBufferIn},
        output{outputIn},
        outputActivated{outputActivatedIn},
        saturationCount{saturationCountIn},
        weights1B{static_cast<int8_t const *>(weightsIn)},
        biasesCompound{static_cast<nn_bias_c const *>(biases)}
    {}

    uint32_t const outputElementCount;      // M - cols
    uint32_t const inputVectorCount;        // N - rows
    uint32_t const inputElementCount;       // K - cols
    int16_t const * input;                  // I - (flat) [N;K]
    int16_t * feedbackBuffer;               // (flat) [N,M]
    int32_t * output;                       // O1 - [N,M]
    int16_t * const outputActivated;        // O2- [N,M]
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
    ConvolutionConfig(uint32_t const inputBandStrideIn, uint32_t const FilterOutputCountIn, uint32_t const FilterCountIn,
        uint32_t const FilterCoefficientCountIn, int16_t const * const inputsIn, int16_t const * const filtersIn,
        nn_bias_s const * const biasesIn, int16_t * const outputsIn, uint32_t * const saturationCountIn) :
        inputBandStride{inputBandStrideIn},
        filterOutputCount{FilterOutputCountIn},
        filterCount{FilterCountIn},
        filterCoefficientCount{FilterCoefficientCountIn},
        inputs{inputsIn},
        filters{filtersIn},
        biases{biasesIn},
        pooledOutputs{outputsIn},
        saturationCount{saturationCountIn}
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

struct ConvolutionPoolingConfig
{
    ConvolutionPoolingConfig(nn_pool_type const typeIn, uint32_t const sizeIn, uint32_t const stepIn,
        int64_t * const bufferIn) :
        type{typeIn},
        size{sizeIn},
        step{stepIn},
        buffer{bufferIn}
    {}

    nn_pool_type const type;
    uint32_t const size;
    uint32_t const step;
    int64_t * buffer;
};

struct GmmConfig
{
    GmmConfig(uint32_t const inputVectorCountIn, uint32_t const inputElementCountIn, uint32_t const mixCountIn,
        uint32_t const meanSetOffsetSizeIn, uint32_t const varSetOffsetSizeIn, uint32_t const gaussConstSetOffsetSizeIn,
        uint32_t const maxScoreIn, uint32_t const stateCountIn, gna_gmm_data const * const dataIn,
        uint8_t const * const inputIn, uint32_t * const outputIn, uint8_t *inputScratchPadIn) :
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
        inputScratchPad{inputScratchPadIn},
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
    uint32_t * const output;
};
