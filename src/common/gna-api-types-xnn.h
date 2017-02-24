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

/******************************************************************************
 *
 * GMM Scoring and Neural Network Accelerator Module
 * API Neural Network types definition
 *
 *****************************************************************************/

#ifndef __GNA_API_TYPES_XNN_H
#define __GNA_API_TYPES_XNN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Bias data type */
typedef int32_t intel_bias_t;

/** Compound bias - as read directly by accelerator */
typedef struct _compound_bias_t
{
    intel_bias_t bias;              // Bias value.  Not used for intel_affine_multibias_func_t.
    uint8_t multiplier;             // Scaling factor bias is multiplied by.
    uint8_t reserved[3];            // Not used.

} intel_compound_bias_t;


static_assert(8 == sizeof(intel_compound_bias_t), "Invalid size of intel_compound_bias_t");

/** Affine function details */
typedef struct _affine_func_t
{
    uint32_t nBytesPerWeight;       // Number of bytes per weight element, set to 1B or 2B.
    uint32_t nBytesPerBias;         // Number of bytes per bias element, 8B for nBytesPerWeight=1 or 4B biases for nBytesPerWeight=2.
    void* pWeights;                 // Weights data buffer. uint8_t for nBytesPerWeight=1 or uint16_t biases for nBytesPerWeight=2.
    void* pBiases;                  // Biases data buffer. intel_compound_bias_t for nBytesPerWeight=1 or intel_bias_t biases for nBytesPerWeight=2.

} intel_affine_func_t;

/** Affine function details with bias grouping */
typedef struct _affine_multibias_func_t
{
    uint32_t nBytesPerWeight;       // Number of bytes per weight element, set to 1B or 2B.
    void* pWeights;                 // Weights data buffer.
    intel_compound_bias_t* weightScaleFactors; // Scaling factors for 1B weights or NULL for 2B weights.
    uint32_t biasVectorIndex;       // Index of bias group for current layer.
    intel_bias_t* pBiases;          // 2D array with grouped biases.

} intel_affine_multibias_func_t;

/** PWL Segment - as read directly by accelerator */
typedef struct _pwl_segment_t
{
    int32_t xBase;                  // X Component of segment starting point, with scaling encoded if needed.
    int16_t yBase;                  // Y Component of segment starting point.
    int16_t slope;                  // Slope of linear function.

} intel_pwl_segment_t;

static_assert(8 == sizeof(intel_pwl_segment_t), "Invalid size of intel_pwl_segment_t");

/** Piecewise-linear activation function (PWL) details */
typedef struct _pwl_func_t
{
    uint32_t nSegments;             // Number of segments, set to 0 to disable activation function.
    intel_pwl_segment_t* pSegments; // Activation function segments data or NULL if disabled.

} intel_pwl_func_t;

/** Fully connected affine layer detailed descriptor */
typedef struct _affine_layer_t
{
    intel_affine_func_t affine;     // Affine function details.
    intel_pwl_func_t pwl;           // Activation function details.

} intel_affine_layer_t;

/** Fully connected affine layer with bias grouping, detailed descriptor */
typedef struct _affine_multibias_layer_t
{
    intel_affine_multibias_func_t affine;// Affine function with bias grouping.
    intel_pwl_func_t pwl;           // Activation function details.

} intel_affine_multibias_layer_t;

/** Pooling function types */
typedef enum _pool_type_t
{
    INTEL_NO_POOLING = 0,           // Pooling function disabled.
    INTEL_MAX_POOLING = 1,          // Max Pooling function.
    INTEL_SUM_POOLING = 2,          // Sum Pooling function.
    NUM_POOLING_TYPES               // Number of Pooling function types.

} intel_pool_type_t;

/** Convolutional Layer detailed descriptor */
typedef struct _convolutional_layer_t
{
    uint32_t nFilters;              // Number of filters.
    uint32_t nFilterCoefficients;   // Number of filter elements, including 0-padding if necessary.
    uint32_t nFilterRows;           // Number of rows in each filter.
    uint32_t nBytesFilterCoefficient;// Number of bytes per filter element, set to 1 or 2.
    uint32_t nBytesBias;            // Number of bytes per bias element, 8 B for nBytesFilterCoefficient=1 or 4 B biases for nBytesFilterCoefficient=2.
    uint32_t nFeatureMaps;          // Number of feature maps.
    uint32_t nFeatureMapRows;       // Number of rows in each feature map.
    uint32_t nFeatureMapColumns;    // Number of columns in each feature map.
    void* pFilters;                 // Filters data buffer, filters stored one after the other.
    void* pBiases;                  // Biases data buffer.
    intel_pool_type_t poolType;     // Pooling function type.
    uint32_t nPoolSize;             // Pool size, set 1 to disable pooling.
    uint32_t nPoolStride;           // Pool stride.
    intel_pwl_func_t pwl;           // Activation function details.

} intel_convolutional_layer_t;

/** Copying Layer detailed configuration */
typedef struct _copy_layer_t
{
    uint32_t nCopyRows;             // Number of rows affected (1-8).
    uint32_t nCopyCols;             // Number of columns in a row to copy.

} intel_copy_layer_t;

/** Copying Layer detailed descriptor */
typedef struct _recurrent_layer_t
{
    intel_affine_func_t affine;     // Affine function details.
    intel_pwl_func_t pwl;           // Activation function details.
    void* pFeedbackBuffer;          // Feedback input buffer. Size same as in the output buffer.

} intel_recurrent_layer_t;

/** Layer kind list */
typedef enum _layer_kind_t
{
    INTEL_AFFINE,                   // Cast pLayerStruct to intel_affine_layer_t.
    INTEL_AFFINE_DIAGONAL,          // Cast pLayerStruct to intel_affine_layer_t.
    INTEL_AFFINE_MULTIBIAS,         // Cast pLayerStruct to intel_affine_multibias_layer_t.
    INTEL_CONVOLUTIONAL,            // Cast pLayerStruct to intel_convolutional_layer_t.
    INTEL_COPY,                     // Cast pLayerStruct to intel_copy_layer_t.
    INTEL_DEINTERLEAVE,             // No casting, always set pLayerStruct to null.
    INTEL_GMM,                      // Cast pLayerStruct to intel_gmm_layer_t.
    INTEL_INTERLEAVE,               // No casting, always set pLayerStruct to null.
    INTEL_RECURRENT,                // Cast pLayerStruct to intel_recurrent_layer_t.
    NUM_LAYER_KINDS                 // Number of Layer kinds.

} intel_layer_kind_t;

/** Layer configuration descriptor */
typedef struct _nnet_layer_t
{
    intel_layer_kind_t nLayerKind;  // Layer kind.
    uint32_t nInputColumns;         // Number of input columns.
    uint32_t nInputRows;            // Number of input rows.
    uint32_t nOutputColumns;        // Number of output columns.
    uint32_t nOutputRows;           // Number of output rows.
    uint32_t nBytesPerInput;        // Number of bytes per input node, always set to 2.
    uint32_t nBytesPerOutput;       // Number of bytes per output node, set to 2 or 4.
    uint32_t nBytesPerIntermediateOutput;// Number of bytes per intermediate output node, always set to 4.
    void* pLayerStruct;             // Layer detailed configuration, cast to intel_[LAYER_KIND]_layer_t.
    void* pInputs;                  // NN or GMM input buffer.
    void* pOutputsIntermediate;     // Auxiliary output buffer. (Used for reading outputs before activation.)
    void* pOutputs;                 // Output buffer.

} intel_nnet_layer_t;

/** GNA Network descriptor */
typedef struct _nnet_type_t
{
    uint32_t nLayers;               // Number of layers in network.
    uint32_t nGroup;                // Input vector grouping level.
    intel_nnet_layer_t *pLayers;    // Layer configurations.

} intel_nnet_type_t;

/** Number of input groups constraint - max */
const uint32_t XNN_N_GROUP_MAX = 8;

/** Total number of input elements constraint - must be multiple of */
const uint32_t XNN_N_IN_ELEMS_MPLY = 8;

/** Total number of output elements constraint - must be multiple of */
const uint32_t RNN_N_OUT_ELEMS_MPLY = 32;

/** Total number of input elements constraint - max elements */
const uint32_t XNN_N_IN_ELEMS_MAX = UINT16_MAX;

/** Number of pwl segments constraint - max  */
const uint32_t XNN_N_PWL_SEGS_MAX = 128;

/** Number of pwl segments constraint - min  */
const uint32_t XNN_N_PWL_SEGS_MIN = 2;

/** Weight elements size constraint - max size B */
const uint32_t XNN_W_ELEM_SZ_MAX = 2;

/** xNN maximum number of Layers  */
const uint32_t GMM_LAYERS_MAX_COUNT = 8192;

/** xNN maximum number of Layers  */
const uint32_t XNN_LAYERS_MAX_COUNT = 8192;

/** CNN minimum number of filter coefficients */
const uint32_t CNN_N_FLT_COEFF_MIN = 48;

/** CNN maximum number of filter coefficients */
const uint32_t CNN_N_FLT_COEFF_MAX = 768;

/** CNN number of filter coefficients constraint - must be multiple of */
const uint32_t CNN_N_FLT_COEFF_MPLY = 4;

/** CNN maximum number of filters */
const uint32_t CNN_N_FLT_MAX = ((UINT16_MAX + 1) - 4);

/** CNN minimum size of pooling window */
const uint32_t CNN_POOL_SIZE_MIN = 1;

/** CNN maximum size of pooling window */
const uint32_t CNN_POOL_SIZE_MAX = 6;

#ifdef __cplusplus
}
#endif

#endif  // ifndef __GNA_API_TYPES_XNN_H
