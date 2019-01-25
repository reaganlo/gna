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

#include <stdint.h>
#include "../common.h"
#include "../Address.h"
#include "gna-api-types-gmm.h"

using GNA::BufferMap;
using GNA::BaseAddress;

/**
 * Structure will hold aligned deinterleaved feature vectors
 * and PWL activation function auxiliary buffers used for performance improvements
 * One structure per thread in thread pool will be created and managed by kernel dispatcher
 */
struct KernelBuffers
{
    int16_t *d0;
    int16_t *d1;
    int16_t *d2;
    int16_t *d3;
    int16_t *d4;
    int16_t *d5;
    int16_t *d6;
    int16_t *d7;
    int64_t *pool;
    int8_t *cnnFusedBuffer;
    KernelBuffers();
    ~KernelBuffers();
};

namespace GNA
{
struct PwlCached;
}

struct BaseConfig: public BufferMap
{
    using BufferMap::BufferMap;
    BaseConfig() = default;
    BaseConfig(const BufferMap& buffers);
    BaseConfig(const BaseAddress& inputBuffer, const BaseAddress& outputBuffer);

    void Update(const BufferMap& buffers);

    int8_t const * Inputs = nullptr;
    int8_t * Outputs = nullptr;
private:
    void setPointers(); // TODO:3:remove and use map in kernels?
};

template<typename TransformConfig>
struct KernelConfig : public BaseConfig
{
    using BaseConfig::BaseConfig;
    KernelConfig(KernelConfig const & source) = default;
    KernelConfig(TransformConfig const & source, BaseConfig const & io) :
        BaseConfig{io},
        Transform{source}
    {}

    TransformConfig Transform;
};

struct ExecutionConfig
{
    ExecutionConfig(KernelBuffers const * intermediate, uint32_t * saturationCount, uint32_t const * bufferElementCount) :
        Intermediate{intermediate},
        SaturationCount{saturationCount},
        BufferElementCount{bufferElementCount}
    {};

    KernelBuffers const * const Intermediate;
    uint32_t * const SaturationCount;
    uint32_t const * const BufferElementCount;
};

template<typename TransformConfig>
struct ExecutionKernelConfig : public ExecutionConfig
{
    ExecutionKernelConfig(KernelConfig<TransformConfig> * requestConfig,
        ExecutionConfig const & executioConfig) :
        ExecutionConfig{executioConfig},
        RequestConfig{requestConfig}
    {
        if (nullptr == RequestConfig->Inputs)
        {
            RequestConfig->Inputs = Intermediate->cnnFusedBuffer;
        }
        if (nullptr == RequestConfig->Outputs)
        {
            RequestConfig->Outputs = Intermediate->cnnFusedBuffer;
        }
    }

    KernelConfig<TransformConfig> * const RequestConfig;
};

struct ActivationConfig
{
    ActivationConfig() = default; // TODO:3:remove when all layers are using Transform as base class
    ActivationConfig(ActivationConfig const & source) = default;
    ActivationConfig(uint32_t elementCount, GNA::PwlCached const * kernel);

    uint32_t ElementCount;
    GNA::PwlCached const * const Kernel;
};

struct PoolingConfig2D
{
    PoolingConfig2D(PoolingConfig2D const & source) = default;
    PoolingConfig2D(nn_layer_pool2d const & config);

    nn_layer_pool2d const Pooling;
};

struct ConvolutionConfig2D
{
    ConvolutionConfig2D(ConvolutionConfig2D const & source) = default;
    ConvolutionConfig2D(gna_3d_dimensions const & inputDimensions,
        gna_convolution_func const & config);

    gna_3d_dimensions const InputDimensions;
    gna_convolution_func const Convolution;
};

// TODO: refactor: consider splitting into run config and basic constant config
struct AffineConfig
{
    AffineConfig(int16_t const * inputIn, int32_t * const outputIn, AffineConfig const * const source);
    AffineConfig(AffineConfig const * const source, ExecutionConfig const & executionConfig);
    AffineConfig(uint32_t const outputElementCountIn, uint32_t const inputVectorCountIn,
        uint32_t const inputElementCountIn, int16_t const * inputIn, int32_t * const outputIn, void const * weightsIn,
        void const * biases, void const * multiBiasIn, uint32_t const multiBiasVectorCountIn);
    AffineConfig(uint32_t const outputElementCountIn, uint32_t const inputVectorCountIn,
        uint32_t const inputElementCountIn, int16_t const * inputIn, int32_t * const outputIn, void const * weightsIn,
        void const * biases, void const * multiBiasIn, uint32_t const multiBiasVectorCountIn,
        const uint32_t bytesPerBiasIn);

    uint32_t const outputElementCount;  // M - out rows
    uint32_t const inputVectorCount;    // N - columns
    uint32_t const inputElementCount;   // K - rows
    int16_t const * input;              // I - (interleaved) [K;N]
    int32_t * output;                   // O - [M;N]
    ExecutionConfig const * execution;
    union
    {
    int8_t const * const weights1B;     // W - [M;K]
    int16_t const * const weights2B;    // W - [M;K]
    } ;
    union
    {
    nn_scaling const * const weightScaleFactors; // [M] Scaling factors for 1B weights or NULL for 2B weights. 
    nn_bias_c const * const biasesCompound;     // B - [M]
    nn_bias_s const * const biasesSimple;       // B - [M]
    };
    nn_bias_s const * const multiBias;
    uint32_t const multiBiasVectorCount;
    uint32_t const bytesPerBias = 0;
};

struct AffineConfigAl
{
    AffineConfigAl(uint32_t const * indicesIn, uint32_t const countIn);

    uint32_t const * const indices; // AL [L]
    uint32_t const count;           // L
};

struct RecurrentConfig
{
    RecurrentConfig(RecurrentConfig const * const source, ExecutionConfig const & executionConfig);
    RecurrentConfig(
        uint32_t const outputElementCountIn, uint32_t const inputVectorCountIn, uint32_t const inputElementCountIn,
        int16_t const * inputIn, int16_t * const feedbackBufferIn, int32_t * const outputIn,
        int16_t * outputActivatedIn, void const * weightsIn, void const * biases, ActivationConfig const & pwl);
    RecurrentConfig(
        uint32_t const outputElementCountIn, uint32_t const inputVectorCountIn, uint32_t const inputElementCountIn,
        int16_t const * inputIn, int16_t * const feedbackBufferIn, int32_t * const outputIn,
        int16_t * outputActivatedIn, void const * weightsIn, void const * biases,
        uint32_t bytesPerBiasIn, uint32_t bytesPerOutput, ActivationConfig const & pwl);

    uint32_t const outputElementCount;      // M - cols
    uint32_t const inputVectorCount;        // N - rows
    uint32_t const inputElementCount;       // K - cols
    int16_t const * input;                  // I - (flat) [N;K]
    int16_t * feedbackBuffer;               // (flat) [N,M]
    int32_t * output;                       // O1 - [N,M]
    ExecutionConfig const * execution;
    uint32_t bytesPerBias = 0;
    uint32_t bytesPerOutput = 0;
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
    KernelConfig<ActivationConfig> activation;
};

struct TransposeConfig
{
    TransposeConfig(uint32_t rowCountIn, uint32_t columntCountIn, int16_t const * const inputIn,
        int16_t * const outputIn);

    uint32_t const rowCount;
    uint32_t const columnCount;
    int16_t const * input;
    int16_t * output;
};

struct CopyConfig
{
    CopyConfig(uint32_t rowCountIn, uint32_t columntCountIn, uint32_t inputColumnCountIn, uint32_t outputColumnCountIn,
        int16_t const * const inputIn, int16_t * const outputIn);

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
        int32_t * const outputsIn);
    ConvolutionConfig(ConvolutionConfig const * const source, ExecutionConfig const & executionConfig);
    ConvolutionConfig(uint32_t const inputBandStrideIn, uint32_t const FilterOutputCountIn, uint32_t const FilterCountIn,
        uint32_t const FilterCoefficientCountIn, int16_t const * const inputsIn, int16_t const * const filtersIn,
        nn_bias_s const * const biasesIn, int32_t * const outputsIn);
    ConvolutionConfig(uint32_t const inputBandStrideIn, uint32_t const FilterOutputCountIn, uint32_t const FilterCountIn,
        uint32_t const FilterCoefficientCountIn, int16_t const * const inputsIn, int16_t const * const filtersIn,
        nn_bias_s const * const biasesIn, int32_t * const outputsIn, uint32_t bytesPerBiasIn, uint32_t bytesPerFilter);

    uint32_t const inputBandStride;
    uint32_t const filterOutputCount;
    uint32_t const filterCount;
    uint32_t const filterCoefficientCount;
    uint32_t const bytesPerBias = 0;
    uint32_t const bytesPerFilter = 0;

    int16_t const * inputs;
    int16_t const * const filters;
    nn_bias_s const * const biases;

    union
    {
        int32_t * convolutedOutputs;
        int16_t * pooledOutputs;
    };
    ExecutionConfig const * execution;
};

struct PoolingConfig
{
    PoolingConfig(PoolingConfig const * const source, int64_t * const bufferIn);
    PoolingConfig(nn_pool_type const typeIn, uint32_t const sizeIn, uint32_t const stepIn);

    nn_pool_type const type;
    uint32_t const size;
    uint32_t const step;
    int64_t * buffer;
};

struct GmmConfig
{
    GmmConfig(GmmConfig const * const source, uint8_t *inputScratchPadIn);
    GmmConfig(uint32_t const inputVectorCountIn, uint32_t const inputElementCountIn, uint32_t const mixCountIn,
        uint32_t const meanSetOffsetSizeIn, uint32_t const varSetOffsetSizeIn, uint32_t const gaussConstSetOffsetSizeIn,
        uint32_t const maxScoreIn, uint32_t const stateCountIn, gna_gmm_data const * const dataIn,
        uint8_t const * const inputIn, uint32_t * const outputIn);

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
