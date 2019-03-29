/*
 @copyright

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing,
 software distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 see the License for the specific language governing permissions
 and limitations under the License.

 SPDX-License-Identifier: Apache-2.0
*/

/**************************************************************************//**
 @file gna2-model-api.h
 @brief Gaussian and Neural Accelerator (GNA) 2.0 API Definition.
 @nosubgrouping

 ******************************************************************************

 @addtogroup GNA2_API
 @{
 ******************************************************************************

 @addtogroup GNA2_MODEL_API Model API

 API for definition and management of GNA data-flow model.

 @{
 *****************************************************************************

 @addtogroup GNA2_MODEL_DATA_FLOW_API Data-Flow Model

 Fundamental API structures and types for definition and management
 of GNA data-flow model.

 @{
 *****************************************************************************/

#ifndef __GNA2_MODEL_API_H
#define __GNA2_MODEL_API_H

#include "gna2-common-api.h"

#if !defined(_WIN32)
#include <assert.h>
#endif
#include <stdint.h>

/* Model types forward declarations. */
enum GnaBiasMode;
enum GnaDataType;
enum GnaOperationType;
enum GnaPoolingMode;
enum GnaTensorMode;
struct GnaModel;
struct GnaOperation;
struct GnaShape;
struct GnaTensor;
struct GnaCompoundBias;
struct GnaWeightScaleFactor;
struct GnaPwlSegment;

/**
 Creates and compiles the model for use with a given device.

 @note
 - The model has to be placed in user's memory, not allocated by GnaMemoryAlloc().

 @param deviceIndex GNA device that will utilize the model.
 @param model Model descriptor which will govern the model creation.
 @param [out] modelId The model identifier assigned by GNA.
 @return Status of the operation.
 */
GNA_API enum GnaStatus GnaModelCreate(
    uint32_t deviceIndex,
    struct GnaModel const * model,
    uint32_t * modelId);

/**
 Releases model structures and request configurations

 @param modelId Model to release
 @return Status of the operation.
 */
GNA_API enum GnaStatus GnaModelRelease(
    uint32_t modelId);

/**
 GNA data-flow Model.

 @see https://en.wikipedia.org/wiki/Dataflow_programming

 Data-flow model is a directed graph of nodes that represent
 Operations (GnaOperation), either simple (e.g. addition) or composed
 (e.g. fused convolution).
 Operation nodes are connected with edges that represent Operands
 or data (GnaTensor).
 */
struct GnaModel
{
    /**
     Number of Operations.
     Maximal number of operations depends on available device generation.
    */
    uint32_t NumberOfOperations;

    /**
     Operations which define the graph.
    */
    struct GnaOperation * Operations;

    /**
     Maximal number of input tensors in a batch.
     Supported values: [1,8], depends on operation used.
    */
    uint32_t MaximumBatchSize;
};

/**
 Operation configuration.

 For composed operations Inputs and Outputs are always specified per whole
 operation, i.e. inputs for first operation and output from the last operation.
 Intermediate results and buffer are not directly accessible for composed
 operations.

 @see GnaModelOperationInit() That simplifies operation creation.
 */
struct GnaOperation
{
    /**
     Type of executed operation.
     */
    enum GnaOperationType Type;

    /**
     Operands that operation is executing on.

     Number of operands is defined by Type.
     @see GnaOperationType.

     @note
        Set unused Operands pointers to NULL.
    */
    struct GnaTensor const ** Operands;

    /**
     Number of Operands that are actually provided.
     */
    uint32_t NumberOfOperands;

    /**
     Constant parameters providing additional operation configuration.
     Number and types of parameters are defined by operation Type.
     Currently parameters can be enumerations, GnaShape or single integers.
     @see GnaOperationType.
    */
    void ** Parameters;

    /**
     Number of Parameters that are actually provided.
     */
    uint32_t NumberOfParameters;
};

/**
 Operation type.

 Defines type of single or composed operation.
 Composed operation is a "fused" transformation of a few chained operations,
 e.g. ::GnaOperationTypeFullyConnectedAffine is defined as dot product, addition and activation function,
 */
enum GnaOperationType
{
    /**
    Convolutional operation composed with activation function and pooling.

    Operation:
        - a) outputs = pooling(activation(convolution(inputs, filters) + biases), activationFunction))
        - b) outputs = pooling(activation(convolution(padding(inputs, zeroPadding), filters) + biases), activationFunction))
        .
        Where:
        - pooling is optional
        - activation is optional
        - padding is optional

    Operands:
        1. inputs
        2. outputs
        3. filters
        4. biases
            - For 1D convolution operation: {::GnaBiasModeDefault}
            - For 2D convolution operation: {::GnaBiasModeDefault, ::GnaBiasModePerStride}
        5. activationFunction

    Parameters:
        1. GnaShape zeroPadding [optional]:
            Supported only for 2D convolution.
            Specifies automatic input zero-padding dimensions.
             Used to maintain same input-output volume shape
             or when input dimensions have no common natural divider with filter and stride.
             Valid values:
                [0] When not used
                [W x H] 2D where: //TODO:3:API Redesign: provide shape info
                    - W is a number of 0s added each before and after each row of an input
                    - H is a number of 0s added each before and after each column of an input
        2. GnaShape concolutionStride [required]:
             Specifies filter stride shape.
             Valid values:
                For 1D convolution operation:
                    [W] 1D where: //TODO:3:API Redesign: provide shape info
                     - W is a number of elements to move in W dimension
                For 2D convolution operation:
                    [W x H] 2D where: //TODO:3:API Redesign: provide shape info
                     - W is a number of elements to move in W dimension
                     - H is a number of elements to move in H dimension
        3. GnaBiasMode biasMode - Mode of bias operation [optional] (GnaBiasMode):
            Supported values:
            - ::GnaBiasModeDefault: normal operation
            - ::GnaBiasModePerStride: TODO:3:API:elaborate
        4. GnaPoolingMode poolingType of bias operation [optional] (GnaPoolingMode):
        5. GnaShape poolingWindow:
            Specifies pooling window shape.
            Valid values:
            - For 1D convolution operation: [ W ] 1D where: //TODO:3:API Redesign: provide shape info
                - W is a width of window
            - For 2D convolution operation:
                - [ W x H ] 2D where:
                    - W is a width of window
                    - H is a height of window
        6. GnaShape poolingStride:
            Specifies pooling window stride dimensions.
            Valid values:
                - For 1D convolution operation: [W] 1D where: //TODO:3:API Redesign: provide shape info
                    - W is a number of elements to move in W dimension
                - For 2D convolution operation: [W x H] 2D where:
                    - W is a number of elements to move in W dimension
                    - H is a number of elements to move in H dimension
    */
    GnaOperationTypeConvolution = 1,

    /**
    Copy operation.

    Operation:
        output = copy(input, shape)

    Operands:
        - 0 inputs
        - 1 outputs

    Parameters:
        - 0 GnaShape copyParams [required]:
             Specifies dimensions of copied sub-tensor.
             Valid values:
                [W x H] 2D where: //TODO:3:API Redesign: provide shape info
                 - W is a number of elements to copy in W dimension
                 - H is a number of elements to copy in H dimension
    */
    GnaOperationTypeCopy = 2,

    /**
    Fully connected affine operation composed with activation function.

    Operation:
    - a: outputs = activation(((inputs x weights) + biases), activationFunction)
    - b: outputs = activation(((inputs x (weightScaleFactors * weights)) + biases[biasVectorIndex]), activationFunction)

    Where:
    - activation is optional
    - weightScaleFactors is optional, required only for ::GnaBiasModeGrouping Mode.

    Operands:
    - 0 inputs
    - 1 outputs
    - 2 filters
    - 3 biases
    - 4 activationFunction
    - 5 weightScaleFactors

    Parameters:
        - 0 GnaBiasMode Mode of bias operation [optional] (GnaBiasMode):
            Supported values:
            -::GnaBiasModeDefault: normal operation
            -::GnaBiasModeGrouping: Special optimized case // TODO:3:API: elaborate
                Requires weightScaleFactors, Bias vector index
        - 1 uint32_t biasVectorIndex [optional]:
            Index of the bias vector used for this operation.
            Supported values:
            -[0, N-1]: Where N is a number of all bias vectors in biases tensor.
             Default is 0.
            -GNA_DEFAULT: is equivalent of 0.
    */
    GnaOperationTypeFullyConnectedAffine = 3,

    /**
    Element wise affine operation composed with activation function and pooling.

    Weights are diagonal matrix, represented by 1D vector.
    output = activation((times(input, weights) + bias), pwl)
    Used e.g. for scaling input tensor.
    */
    GnaOperationTypeElementWiseAffine = 4,

    /**
    Gaussian Mixture Model scoring operation.

    Operation:
        a) output = GMM(input, means, inverseCovariances, constants)
        b) output = GMM(input, interleaved{means, inverseCovariances, constants})

    Operands a):
        - 0 inputs
        - 1 outputs
        - 2 means
        - 3 inverseCovariances
        - 4 constants
    Operands b):
        - 0 inputs
        - 1 outputs
        - 2 interleaved{means, inverseCovariances, constants}
    Parameters:
        - 0 uint32_t maximumScore [required]:
            Maximum Score value above which scores are saturated.
    */
    GnaOperationTypeGmm = 5,

    /**
    Fully connected affine operation with recurrence composed with activation function.

     Operation:
        output = activation((((input[t], output[t-delay]) x weights) + bias), activationFunction)
        Where:
            output[t-delay] - recurrent input (feedback) from t-delay output vector of current request.

     Operands:
        - 0 inputs
        - 1 outputs
        - 2 filters
        - 3 biases
        - 4 activationFunction

     Parameters:
        -uint32_t delay:
            Delay in term of number of vectors in request.
            Supported values:
            -[1, N-1]: Where N is a number of input vectors in current request.
    */
    GnaOperationTypeRecurrent = 6,

    /**
     Tensor transposition operation.

     output<layout> = transpose(input<layout>)
     //TODO:3:API: use Tensor.Layout in input and output tensor to determine type of transposition interleaved/flat/other
    */
    GnaOperationTypeTransposition = 7,

    /**
    Control-flow operation with threshold parameter.
    */
    GnaOperationTypeTreshold = 8,

    //GNA_OPERATION_TYPE_COUT,      // Number of Layer operation types.
    //// TODO:3: use more generic operation type and determine specialization via parameters
    //GnaOperationTypeTransposition (interleave),               // Auxiliary 2D tensor transpose operation (flat to interleave). No casting, always set Operations to null.
    //GnaOperationTypeTransposition (deinterleave),             // Auxiliary 2D tensor transpose operation (interleave to flat). No casting, always set Operations to null.
    //// NOT POR
    //GNA_OPERATION_CONVOLUTIONAL_2D_ADDITION,
    //GNA_OPERATION_CONVOLUTIONAL_2D_CONVERSION,
    //GNA_OPERATION_POOLING_2D,
    //// exemplary not existing operations
    //GNA_OPERATION_NEGATION, // unary, output = !(input)
    //GNA_OPERATION_ADDITION, // binary, output = (input + operand)
    //GNA_OPERATION_DOT_PRODUCT,// binary, output = (input x operand)
    //GNA_OPERATION_MULTIPLICATION,// binary, output = (input * operand)
};

/**
 Maximal number of supported shape dimensions.
 */
#define GNA_SHAPE_MAXIMUM_NUMBER_OF_DIMENSIONS 8

/**
 Shape specifying dimension values.
*/
struct GnaShape
{
    /**
     Number of dimensions or rank or order.

     Set:
     - 0 for scalars,
     - 1 for vectors,
     - 2 for matrices,
     and so on.
    */
    uint32_t NumberOfDimensions;

    /**
     Vector specifying value of each dimension.

     Set all zeros for scalars.
    */
    uint32_t Dimensions[GNA_SHAPE_MAXIMUM_NUMBER_OF_DIMENSIONS];
};

/**
 Tensor used as operation operand.

 Valid parameters:
 - Input Tensor:
    - Common:
        - #Mode: {::GnaTensorModeDefault, ::GnaTensorModeDisabled}
        - #Type: {::GnaDataTypeInt8, ::GnaDataTypeInt16},
        - #Shape: [ N x W ] 2D matrix (if not stated otherwise) where:
            - N is a batch size (number of vectors)
            - W is a number of vector elements
        - #Layout, where not stated otherwise: Column-major (interleaved), vectors are columns.
    - For 1D ::GnaOperationTypeConvolution operation:
        - #Shape: [ W ] 1D vector where:
            - W is a number of vector elements
        - #Layout: Row-major (flat)
    - For 2D ::GnaOperationTypeConvolution  operation:
        - #Shape: [N x H x W x C] 4D Tensor where:
            - N is a batch size (number of vectors), currently only N=1 is supported
            - H is a height of each filter
            - W is a width of each filter
            - C is a depth of each filter
    - For ::GnaOperationTypeTransposition (interleave), ::GnaOperationTypeRecurrent operation:
        - #Layout: Row-major (flat)
 - Output Tensor:
    - Common:
        - #Mode: {::GnaTensorModeDefault, ::GnaTensorModeDisabled}
        - #Type: {::GnaDataTypeInt8, ::GnaDataTypeInt16, ::GnaDataTypeInt32}
        - #Shape, where not stated otherwise, [ N x W ] 2D matrix where:
            - N is a batch size (number of vectors)
            - W is a number of vector elements
        - #Layout, where not stated otherwise: Column-major (interleaved), vectors are columns.
    - For 1D ::GnaOperationTypeConvolution  operation:
        - #Shape: [DxW] 2D matrix, where:
            - D is a number of feature maps
            - W is a number of elements of each feature map
        - #Layout: Column-major (interleaved)  //TODO:3:API Redesign: provide shape info
    - For 2D ::GnaOperationTypeConvolution operation:
        - #Shape: [N x H x W x C] 4D Tensor where:
            - N is a batch size (number of vectors), currently only N = 1 is supported
            - H is a height of each feature maps
            - W is a width of each feature maps
            - C is a number of feature maps
    - For ::GnaOperationTypeTransposition (deinterleave), ::GnaOperationTypeRecurrent operation:
        - #Layout: Row-major (flat), vectors are rows.
 - Bias tensor.
    - Common:
        - #Mode: {::GnaTensorModeDefault, ::GnaTensorModeConstantScalar}
        - #Type: {::GnaDataTypeInt8, ::GnaDataTypeInt16, ::GnaDataTypeInt32},
    - #Shape:
        - For ::GnaBiasModeDefault [ W ] 1D vector where:
            - W is a number of elements (same as number of outputs or filters)
        - For bias, only for ::GnaOperationTypeFullyConnectedAffine operation
          [ N x W ] 2D Matrix where:
            - N is a number of bias vectors
            - W is a number of elements in vector, (same as number of outputs or filters)
        - For ::GnaBiasModePerStride, valid only for 2D ::GnaOperationTypeConvolution  operation:
          [ H x W x C ] 3D tensor same as filter shape
 - Weight Scale Factor Tensor
    //TODO:3:API Redesign: seems rather like output scaling/?
     - Specifies scale factors for weights.
     - Required only for Weights->GnaTensor::Type = ::GnaDataTypeInt8 and Biases->GnaTensor::Type = ::GnaDataTypeCompoundBias.
     - Set GnaTensor::Mode = ::GnaTensorModeDisabled for other cases.
     //TODO:3:API Redesign: provide formula used
     - Valid values:
        - #Shape: 1D Vector //TODO:3:API Redesign: provide shape info
        - #Mode: {::GnaTensorModeDefault}
        - #Type: {::GnaDataTypeWeightScaleFactor},
*/
struct GnaTensor
{
    /**
     Specifies tensor dimensions.
    */
    struct GnaShape Shape;

    /**
     Mode of tensor interpretation.
     Use ::GnaTensorModeDefault as default.
    */
    enum GnaTensorMode Mode;

    /**
     Data layout or format in memory.

    - Specifies order of dimensions, i.e. how GnaShape::Dimensions are interpreted.
      Size of layout array must be the same as #Shape GnaShape::NumberOfDimensions.
    - E.g.:
        - "WN" W is a number of vector elements (rows) and N is a number of vectors
            in a batch (columns), where data is stored row-major, Elements of each vector
            are stored in columns. Aka interleaved layout.
            - For example let W=8, N=2:
                  v0 |   v1
                ---- | ----
                v0e0 | v1e0
                v0e1 | v1e1
                v0e2 | v1e2
                v0e3 | v1e3
                v0e4 | v1e4
                v0e5 | v1e5
                v0e6 | v1e6
                v0e7 | v1e7
        - "NW" N is a number of vectors in a batch (rows) and W is a number of vector
            elements (columns), where data is stored row-major. Whole vectors
            are stored one after another in memory. Aka flat layout.
            - For example let N=2, W=8:
                v\\e |  e0 |   e1 |   e2 |   e3 |   e4 |   e5 |   e6 |   e7
                ---- |---- | ---- | ---- | ---- | ---- | ---- | ---- | ----
                 v0: |v0e0 | v0e1 | v0e2 | v0e3 | v0e4 | v0e5 | v0e6 | v0e7
                 v1: |v1e0 | v1e1 | v1e2 | v1e3 | v1e4 | v1e5 | v1e6 | v1e7
        - "NHWC" is Number of tensors in a batch, Height, Width and number
            of Channels of tensor, where the rightmost dimension changes fastest.
    - Required for GnaOperation::Type = ::GnaOperationTypeTransposition or ::GnaOperationTypeGmm.
      Optional (set zeros) for other operations.
    */
    char Layout[GNA_SHAPE_MAXIMUM_NUMBER_OF_DIMENSIONS];

    /**
     Type of tensor data.
     */
    enum GnaDataType Type;

    /**
     Data buffer.
     Must be specified before enqueueing request, during model or request config creation.
     */
    // TODO:3:API redesign elaborate more.
    void * Data;
};

/**
 Tensor mode.

 Specifies interpretation or usage of tensor data.
 @note
 - Not all modes are supported by all data types and operations.
 - Only 16MSB are used.
 */
enum GnaTensorMode
{
    /**
     Data interpreted as read-write tensor of type specified by GnaDataType.
     */
    GnaTensorModeDefault = GNA_DEFAULT,

    /**
     Data interpreted as single constant scalar of type specified by GnaDataType.
     */
    GnaTensorModeConstantScalar = 0x010000,

    /**
     Data not used, set buffer to NULL.
     */
    GnaTensorModeDisabled = GNA_DISABLED,

    ///**
    // Indicates Data Mode property is not supported for given entity.
    // */
    //GnaTensorModeNotSupported = GNA_NOT_SUPPORTED,
};

/**
 Type and precision of data used.

 @note
 - Not all type are supported by all data modes and operations.
 - Only 16LSB are used.
 */
enum GnaDataType
{
    /**
     Data type not specified, can be used for disabled data.
     */
    GnaDataTypeVoid = GNA_DISABLED,

    /**
     1 Bit Boolean / binary type.
     Currently not supported.
     */
    GnaDataTypeBoolean = 1,

    /**
     4 bit Signed Integer.
     Currently not supported.
     */
    GnaDataTypeInt4 = 2,

    /**
     1 Byte Signed Integer, use int8_t data.
     */
    GnaDataTypeInt8 = 3,

    /**
     2 Byte Signed Integer, use int16_t data.
     */
    GnaDataTypeInt16 = 4,

    /**
     4 Byte Signed Integer, use int32_t data.
     */
    GnaDataTypeInt32 = 5,

    /**
     8 Byte Signed Integer, use int64_t data.
     */
    GnaDataTypeInt64 = 6,

    /**
     4 bit Unsigned Integer.
     Currently not supported.
     */
    GnaDataTypeUint4 =7,

    /**
     1 Byte Unsigned Integer, use uint8_t data.
     */
    GnaDataTypeUint8 = 8,

    /**
     2 Byte Unsigned Integer, use uint16_t data.
     */
    GnaDataTypeUint16 = 9,

    /**
     4 Byte Unsigned Integer, use uint32_t data.
     */
    GnaDataTypeUint32 = 10,

    /**
     8 Byte Unsigned Integer, use uint64_t data.
     */
    GnaDataTypeUint64 = 11,

    /**
     Rich bias data type, use GnaCompoundBias data.
     Used only for bias tensor and with ::GnaDataTypeInt8 weight mode.
     */
    GnaDataTypeCompoundBias = 12,

    /**
     PWL Activation function segment data type, use GnaPwlSegment data.
     Used only for PWL Activation function segment tensor.
     */
    GnaDataTypePwlSegment = 13,

    /**
     Weight scale factor type, use ::GnaOperationTypeFullyConnectedAffine data.
     Used only for GnaWeightScaleFactor tensor.
     */
    GnaDataTypeWeightScaleFactor = 14,

    // /**
    // Indicates Data Type property is not supported for given entity.
    // */
    //GnaDataTypeNotSupported = GNA_NOT_SUPPORTED,
};

/**
 * Mode of bias usage.
 */
enum GnaBiasMode
{
    /**
     Bias is added per output for affine transformations
     and per filter for convolutional.
    */
    GnaBiasModeDefault = GNA_DEFAULT,

    /**
     Bias is added per each filter stride of convolutional transformation.
    */
    GnaBiasModePerStride = 1,

    /**
     Optimized bias usage mode for operations that produce matrix of biases
     as output of an operation and consume single vectors from that matrix
     in a group of consecutive operations e.g. LSTM.
     Bias from selected vector (group) is added per output for affine
     transformations.
     Used with 2D Bias Tensor,
     only for ::GnaOperationTypeFullyConnectedAffine operation.
     @see GnaTensor.
    */
    GnaBiasModeGrouping = 2,

    // / **
    // Indicates Bias Mode is not supported for given entity.
    // */
    //GnaBiasModeNotSupported = GNA_NOT_SUPPORTED,
};

/**
 Mode of pooling operation.
 */
enum GnaPoolingMode
{
    /**
     Pooling operation is disabled.
     */
    GnaPoolingModeDisabled = GNA_DISABLED,

    /**
     Max pooling is used.
     */
    GnaPoolingModeMax = 1,

    /**
     Sum pooling is used.
     */
    GnaPoolingModeSum = 2,

    /*GnaPoolingNotSupported = GNA_NOT_SUPPORTED,*/
};

/**************************************************************************//**
 @}

 @addtogroup GNA2_MODEL_HW_TYPES_API Hardware Data Structures

 Hardware data types used by GNA library and hardware device for data bandwidth
 usage optimization.

 @{
 *****************************************************************************/

/**
 Compound bias.

 Used for Weights->GnaTensor::Type = ::GnaDataTypeInt8
 and Biases->GnaTensor::Type = ::GnaDataTypeInt16 only.
 Used with ::GnaDataTypeCompoundBias.

 @note
    Data format is read directly by the accelerator.
 */
struct GnaCompoundBias
{
    /**
     Bias (constant) value that is added to result of the dot product
     of the weight matrix row and the input vector, same as "regular" bias.
     */
    int32_t Bias;

    /**
     Weight scaling factor that elements of the corresponding weight matrix row are multiplied by.
     Utilized when Weights->GnaTensor::Type = ::GnaDataTypeInt8 is used.
     */
    uint8_t Multiplier;

    /**
     Padding to 8B only, field is not used.
     */
    uint8_t Reserved[3];
};

static_assert(8 == sizeof(GnaCompoundBias), "Invalid size of GnaCompoundBias");

/**
 Weight element scaling factor.

 Used with ::GnaDataTypeWeightScaleFactor
 and ::GnaOperationTypeFullyConnectedAffine (b variant).
 */
 struct GnaWeightScaleFactor
{
    /**
     Padding to 4B only, field is not used.
     */
    uint8_t Reserved0[4];

    /**
     Weight scaling factor.
     @see GnaCompoundBias::Multiplier.
     */
    uint8_t Multiplier;

    /**
     Padding to 8B only, field is not used.
     */
    uint8_t Reserved1[3];
};

static_assert(8 == sizeof(GnaWeightScaleFactor), "Invalid size of GnaWeightScaleFactor");

/**
 Piecewise-linear activation function segment.

 Defines a single segment of a piecewise linear activation function.
 Used with ::GnaDataTypePwlSegment.
 For a given input x, the appropriate segment is fetched and the output y is calculated as
 follows:
    slopeScale = xBase & 0x03
    xBaseValue = xBase & 0xFFFFFFFC
    y = ((x - xBaseValue) * slope) >> slopeScale + yBase
 */
struct GnaPwlSegment
{
    /**
     The x component of activation input segment starting point with packed slope scaling information.

     The x component value needs to be a multiple of 4, leaving 2 lower bits
     for the slopeScale factor packed into those lower bits.
     Possible slopeScale values:
        + 0 – shift right by 8 bits
        + 1 – shift right by 16 bits
        + 2 – shift right by 24 bits
        + 3 – shift right by 32 bits
     */
    int32_t xBase;

    /**
     The y component of activation output segment starting point.
     */
    int16_t yBase;

    /**
     Slope of linear function, describes the steepness of a line in this segment.
     */
    int16_t Slope;
};

static_assert(8 == sizeof(GnaPwlSegment), "Invalid size of GnaPwlSegment");

/**************************************************************************//**
 @}

 @addtogroup GNA2_MODEL_DEBUG_API Model Debugging

 Debugging functions that simplify debugging GNA Model issues.

 @{
 *****************************************************************************/

enum GnaItemType;
struct GnaModelItem;
enum GnaErrorType;
struct GnaModelError;

/**
 Retrieves information on error during model creation.

 Can be called after GnaModelCreate() have failed (e.g., returned ::GnaStatusInvalidModel status).

 @note
 The function should be called directly after GnaModelCreate() in the same thread.

 @param [out] error The detailed description of model issue.
 @return Status of fetching the model error.
    @retval ::GnaStatusSuccess The error has been fetched successfully.
    @retval ::GnaStatusUnknownError No issue to report.
    @retval ::GnaStatusNullargnotallowed The error pointer was NULL.
 */
GNA_API enum GnaStatus GnaModelGetLastError(struct GnaModelError * error);

/**
 Gets message with description of the last model error.

 @note
 TODO:3:API: provide maximum message size

 @param [out] messageBuffer User allocated buffer for the message.
 @param [in] messageBufferSize The size of the messageBuffer in bytes.
        The message is maximum X characters/bytes long.
        Message is truncated to messageBufferSize if it is longer than messageBufferSize characters.
 @return Status of fetching the model error.
    @retval ::GnaStatusSuccess The error was fully serialized into the messageBuffer.
    @retval ::GnaStatusUnknownError No issue to report.
    @retval ::GnaStatusErrResources The messageBuffer is too small. The message was truncated.
    @retval ::GnaStatusNullargnotallowed The messageBuffer was NULL or messageBufferSize was 0.
 */
GNA_API enum GnaStatus GnaModelGetLastErrorMessage(char * messageBuffer,
    uint32_t messageBufferSize);

//TODO:3:API:make documentation consistent: nouns vs verbs.

/**
 Describes the error that caused creating model to fail.
 */
struct GnaModelError
{
    /**
     Which item is the source of the problem.
     */
    struct GnaModelItem Source;

    /**
     Why item is faulty.
     */
    enum GnaErrorType Reason;

    /**
     Incorrect value given.
     */
    int64_t Value;
};

/**
 Number of additional properties for GnaModelItem.
 */
#define GNA_MODEL_ITEM_NUMBER_OF_PROPERTIES 4

/**
 Determines exact model item for capability query and model error.

 Item location is determined top-down.
 E.g. for querying if operation type is supported set:
 - #Type to ::GnaItemTypeOperationType,
 - #Operation to desired operation type,
 - ::GNA_DISABLED for the remaining fields.
 */
struct GnaModelItem
{
    /**
     Type of a model item.

     ::GnaItemTypeNone if not applicable.
     */
    enum GnaItemType Type;

    /**
     Index (0 based) of an operation.

     ::GNA_DISABLED if not applicable.
     */
    enum GnaOperationType Operation;

    /**
     Index (0 based) of an operation.

     ::GNA_DISABLED if not applicable.

     @note Not applicable for Capability querying.
     */
    int32_t OperationIndex;

    /**
     Index (0 based) of an operand.

     ::GNA_DISABLED if not applicable.
     Requires Operation or OperationIndex to be set.
     */
    int32_t OperandIndex;

    /**
     Index (0 based) of a parameter.

     ::GNA_DISABLED if not applicable.
     Requires Operation or OperationIndex to be set.
     */
    int32_t ParameterIndex;

    /**
     Index of a dimension of Operand or Parameter.

     ::GNA_DISABLED if not applicable.
     Requires OperandIndex or ParameterIndex to be set.
     */
    int32_t ShapeDimensionIndex;

    /**
     Additional properties for determining complex and future items.

     Number and type of Properties is determined by #Type.
     ::GNA_DISABLED if not applicable.
     */
    int32_t Properties[GNA_MODEL_ITEM_NUMBER_OF_PROPERTIES];
};

/**
 Determines the type of a model item i.e. model or operand property.

 Used for model debugging and Capability Query API.
 Most of the items are 1 to 1 mapping of data-flow model properties.
 */
enum GnaItemType
{
    /**
     Model context is not applicable or unnecessary.
     */
    GnaItemTypeNone = GNA_DISABLED,

    /**
     GnaModel::NumberOfOperations.
     */
    GnaItemTypeModelNumberOfOperations = 0,

    /**
     GnaModel::Operations array.
     */
    GnaItemTypeModelOperations = 1,

    /**
     GnaModel::MaximumBatchSize.
     */
    GnaItemTypeModelMaximumBatchSize = 2,

    /**
     GnaModel::Operations[x]->GnaOperation::Type.
     */
    GnaItemTypeOperationType = 3,

    /**
     GnaModel::Operations[x]->GnaOperation::Operands array.
     */
    GnaItemTypeOperationOperands = 4,

    /**
     GnaModel::Operations[x]->GnaOperation::NumberOfOperands.
     */
    GnaItemTypeOperationNumberOfOperands = 5,

    /**
     GnaModel::Operations[x]->GnaOperation::Parameters array.
     */
    GnaItemTypeOperationParameters = 6,

    /**
     GnaModel::Operations[x]->GnaOperation::NumberOfParameters.
     */
    GnaItemTypeOperationNumberOfParameters = 7,

    /**
     GnaModel::Operations[x]->GnaOperation::Operands[y]->GnaTensor::Mode.
     */
    GnaItemTypeOperandMode = 8,

    /**
     GnaModel::Operations[x]->GnaOperation::Operands[y]->GnaTensor::Layout.
     */
    GnaItemTypeOperandLayout = 9,

    /**
     GnaModel::Operations[x]->GnaOperation::Operands[y]->GnaTensor::Type.
     */
    GnaItemTypeOperandType = 10,

    /**
     GnaModel::Operations[x]->GnaOperation::Operands[y]->GnaTensor::Data.
     */
    GnaItemTypeOperandData = 11,

    /**
     GnaModel::Operations[x]->GnaOperation::Parameters[z]->Parameter, can be of type GnaShape, enumeration or integer.
     */
    GnaItemTypeParameter = 12,

    /**
     GnaModel::Operations[x]->{GnaTensor; Parameter}->GnaShape::NumberOfDimensions.
     */
    GnaItemTypeShapeNumberOfDimensions = 13,

    /**
     GnaModel::Operations[x]->{GnaTensor; Parameter}->GnaShape::Dimensions.
     */
    GnaItemTypeShapeDimensions = 14,

    /**
     Internal model item, that is a derivative of other model parameters.

     Used only for model debugging.
     When set detailed issue source and description will be reported
     via error message.
     */
    GnaItemTypeInternal = 15,
};

/**
 Type of model item error.

 Helps identifying root cause of model issue.
 */
enum GnaErrorType
{
    /**
     TODO:3:API: document
     */
    GnaErrorTypeNone = GNA_DEFAULT,

    /**
     TODO:3:API: document
     */
    GnaErrorTypeNotTrue = -1,

    /**
     TODO:3:API: document
     */
    GnaErrorTypeNotFalse = -2,

    /**
     TODO:3:API: document
     */
    GnaErrorTypeNullNotAllowed = -3,

    /**
     TODO:3:API: document
     */
    GnaErrorTypeNullRequired = -4,

    /**
     TODO:3:API: document
     */
    GnaErrorTypeBelowRange = -5,

    /**
     TODO:3:API: document
     */
    GnaErrorTypeAboveRange = -6,

    /**
     TODO:3:API: document
     */
    GnaErrorTypeNotEqual = -7,

    /**
     TODO:3:API: document
     */
    GnaErrorTypeNotGtzero = -8,

    /**
     TODO:3:API: document
     */
    GnaErrorTypeNotZero = -9,

    /**
     TODO:3:API: document
     */
    GnaErrorTypeNotOne = -10,

    /**
     TODO:3:API: document
     */
    GnaErrorTypeNotInSet = -11,

    /**
     TODO:3:API: document
     */
    GnaErrorTypeNotMultiplicity = -12,

    /**
     TODO:3:API: document
     */
    GnaErrorTypeNotSuccess = -13,

    /**
     TODO:3:API: document
     */
    GnaErrorTypeNotAligned = -14,

    /**
     Some operation argument was not provided.
     */
    GnaErrorTypeArgumentMissing = -15,

    /**
     Given operation argument was invalid or unexpected.
     */
    GnaErrorTypeArgumentInvalid= -16,

    /**
     Runtime error occurred during model creation.
     */
    GnaErrorTypeRuntime = -17,

    /**
     Unable to determine the root cause of the issue.
     */
    GnaErrorTypeOther = GNA_NOT_SUPPORTED,
};

/**************************************************************************//**
 @}

 @addtogroup GNA2_MODEL_UTILLITY_API Model Utilities

 Utility functions that simplify GNA Model creation.

 @{
 *****************************************************************************/

/**
 Initializes given operation.

 Helper function that initializes operation for user.
 This includes:
    1. GnaOperation::Type is set to type.
    2. GnaOperation::NumberOfOperands is set to value determined by GnaOperation::Type.
    3. GnaOperation::Operands array of pointers is allocated by userAllocator.
        Number of array elements is GnaOperation::NumberOfOperands.
        All pointers are set to NULL.
    2. GnaOperation::NumberOfParameters is set to value determined by GnaOperation::Type.
    3. GnaOperation::Parameters array of pointers is allocated by userAllocator.
        Number of array elements is GnaOperation::NumberOfParameters.
        All pointers are set to NULL.

 @warning
    User is responsible for releasing allocated GnaOperation::Operands
    and GnaOperation::Parameters buffers.

 @param operation The affected operation.
 @param type The type of executed operation.
 @param userAllocator User provided memory allocator.
 @return Status of the operation.
 */
GNA_API enum GnaStatus GnaModelOperationInit(
    struct GnaOperation * operation,
    enum GnaOperationType type,
    GnaUserAllocator userAllocator);

/**
 Gets the size in bytes of given data type.

 Useful for calculating the sizes of memory buffers.

 @param type The type of the data.
 @return Size in bytes of given data type.
 */
GNA_API uint32_t GnaDataTypeGetSize(enum GnaDataType type);

/**
 Gets the total number of elements of the given shape.

 Useful for calculating the sizes of memory buffers.

 @param shape The shape to calculate the number of elements.
 @return Total number of elements.
 */
GNA_API uint32_t GnaShapeGetNumberOfElements(struct GnaShape const * shape);

/**
 Gets the size in bytes of entire tensor data.

 Useful for calculating the sizes of memory buffers.

 @param tensor The tensor to calculate the size of.
 @return Size in bytes of given tensor.
 */
GNA_API uint32_t GnaTensorGetSize(struct GnaTensor const * tensor);

/**
 Helper function that simplifies common GnaShapes creation.

 @return Complete GnaShape representing scalar.
 */
GNA_API struct GnaShape GnaShapeInitScalar();

/**
 Helper function that simplifies common GnaShapes creation.

 @note
 No arguments validation is performed.

 @param x Size of a vector.
 @return Complete GnaShape representing 1D vector dimension.
 */
GNA_API struct GnaShape GnaShapeInit1D(uint32_t x);

/**
 Helper function that simplifies common GnaShapes creation.

 @note
 No arguments validation is performed.

 @param x First matrix dimension.
 @param y Second matrix dimension.
 @return Complete GnaShape representing 2D matrix dimensions.
 */
GNA_API struct GnaShape GnaShapeInit2D(uint32_t x, uint32_t y);

/**
 Helper function that simplifies common GnaShapes creation.

 @note
 No arguments validation is performed.

 @param x First tensor dimension.
 @param y Second tensor dimension.
 @param z Third tensor dimension.
 @return Complete GnaShape representing 3D tensor dimensions.
 */
GNA_API struct GnaShape GnaShapeInit3D(uint32_t x, uint32_t y, uint32_t z);

/**
 Helper function that simplifies common GnaShapes creation.

 @note
 No arguments validation is performed.

 @param n First tensor dimension, usually representing batch size or number of filters.
 @param x Second tensor dimension.
 @param y Third tensor dimension.
 @param z Fourth tensor dimension.
 @return Complete GnaShape representing 3D tensor dimensions.
 */
GNA_API struct GnaShape GnaShapeInit4D(uint32_t n, uint32_t x, uint32_t y,
    uint32_t z);


GNA_API struct GnaTensor GnaTensorInit1D(uint32_t x, enum GnaDataType type,
    void * data);

GNA_API struct GnaTensor GnaTensorInit2D(uint32_t x, uint32_t y,
    enum GnaDataType type, void * data);

GNA_API struct GnaTensor GnaTensorInit3D(uint32_t x, uint32_t y, uint32_t z,
    enum GnaDataType type, void * data);

GNA_API struct GnaTensor GnaTensorInit4D(uint32_t n, uint32_t x, uint32_t y,
    uint32_t z, enum GnaDataType type, void * data);

GNA_API struct GnaTensor GnaTensorInitDisabled();

GNA_API struct GnaTensor GnaTensorInitScalar(enum GnaDataType type, void * data);

GNA_API struct GnaTensor GnaTensorInitActivation(uint32_t numberOfSegments,
    struct GnaPwlSegment * segments);


GNA_API struct GnaOperation GnaOperationInitFullyConnectedAffine(
    struct GnaTensor * inputs, struct GnaTensor * outputs,
    struct GnaTensor * weights, struct GnaTensor * biases,
    struct GnaTensor * activation);

GNA_API struct GnaOperation GnaOperationInitElementWiseAffine(
    struct GnaTensor * inputs, struct GnaTensor * outputs,
    struct GnaTensor * weights, struct GnaTensor * biases,
    struct GnaTensor * activation);

GNA_API struct GnaOperation GnaOperationInitFullyConnectedBiasGrouping(
    struct GnaTensor * inputs, struct GnaTensor * outputs,
    struct GnaTensor * weights, struct GnaTensor * biases,
    struct GnaTensor * activation,
    struct GnaTensor * weightScaleFactors,
    enum GnaBiasMode* biasMode,
    uint32_t* biasVectorIndex);

GNA_API struct GnaOperation GnaOperationInitRecurrent(
    struct GnaTensor * inputs, struct GnaTensor * outputs,
    struct GnaTensor * weights, struct GnaTensor * biases,
    struct GnaTensor * activation,
    uint32_t* delay);


GNA_API struct GnaOperation GnaOperationInitConvolution(
    struct GnaTensor * inputs, struct GnaTensor * outputs,
    struct GnaTensor * filters, struct GnaTensor * biases,
    struct GnaTensor * activation,
    struct GnaShape * zeroPadding,
    struct GnaShape * concolutionStride,
    enum GnaBiasMode * biasMode);

GNA_API struct GnaOperation GnaOperationInitConvolutionFused(
    struct GnaTensor * inputs, struct GnaTensor * outputs,
    struct GnaTensor * filters, struct GnaTensor * biases,
    struct GnaTensor * activation,
    struct GnaShape * zeroPadding,
    struct GnaShape * concolutionStride,
    enum GnaBiasMode * biasMode,
    enum GnaPoolingMode * poolingMode,
    struct GnaShape * poolingWindow,
    struct GnaShape * poolingStride);

GNA_API struct GnaOperation GnaOperationInitPooling(
    struct GnaTensor * inputs, struct GnaTensor * outputs,
    struct GnaTensor * activation,
    struct GnaShape * zeroPadding,
    enum GnaPoolingMode * poolingMode,
    struct GnaShape * poolingWindow,
    struct GnaShape * poolingStride);

GNA_API struct GnaOperation GnaOperationInitCopy(
    struct GnaTensor * inputs, struct GnaTensor * outputs,
    struct GnaShape * copyParams);

GNA_API struct GnaOperation GnaOperationInitTranspose(
    struct GnaTensor * inputs, struct GnaTensor * outputs);

//TODO:3:API define
GNA_API struct GnaOperation GnaOperationInitGmm(
    struct GnaTensor * inputs, struct GnaTensor * outputs,
    struct GnaTensor * means,
    struct GnaTensor * inverseCovariances,
    struct GnaTensor * consts,
    uint32_t * maximumScore);

GNA_API struct GnaOperation GnaOperationInitGmInterleaved(
    struct GnaTensor * inputs, struct GnaTensor * outputs,
    struct GnaTensor * interleavedTensors,
    uint32_t * maximumScore);

// exemplary not existing operations
GNA_API struct GnaOperation GnaOperationInitNegation(
    struct GnaTensor * inputs, struct GnaTensor * outputs);

// exemplary not existing operations
GNA_API struct GnaOperation GnaOperationInitDotProduct(struct GnaTensor * inputs,
    struct GnaTensor * operand, struct GnaTensor * outputs);

#endif // __GNA2_MODEL_API_H

/**
 @}
 @}
 @}
 */



///** Piecewise-linear activation function (PWL) details */
//typedef struct _GnaActivationOperation
//{
//    /**
//     Specifies PWL activation function segment vector.
//     Set Mode = GnaTensorModeDisabled to disable activation.
//     Segments have to be contiguous.
//     Valid values:
//        Shape: [W] 1D vector where:
//            - W is a number of piecewise-linear segments
//        Mode: {GnaTensorModeDefault, GnaTensorModeDisabled}
//        Type: {GNA_PWL_SEGMENT},
//    */
//    struct GnaTensor * Pwl;
//
//} GnaActivationOperation;


//// TODO: 3.0 verify GnaTensorModeDisabled and GNA_TENSOR_CONSTANT_SCALAR usage
///** Affine function details */
//// GnaAffineOperation: GnaAffineTransform = I x Weights + Biases
//typedef struct _GnaAffineOperation
//{
//    /**
//     Specifies weight tensor.
//     Valid values:
//        Common:
//            Mode: {GnaTensorModeDefault, GNA_TENSOR_CONSTANT_SCALAR}
//            Type: {GnaDataTypeInt8, GnaDataTypeInt16},
//        For ::GnaOperationTypeFullyConnectedAffine, GnaOperationTypeRecurrent
//            Shape: [WxH] 2D Matrix //TODO:3:API Redesign: provide shape info
//        For ::GnaOperationTypeElementWiseMultiplication
//            Shape: [W] 1D Vector
//    */
//    struct GnaTensor * Weights;
//
//    /**
//     Specifies bias tensor.
//     Valid values:
//        For ::GnaOperationTypeFullyConnectedAffine,
//              ::GnaOperationTypeElementWiseMultiplication, GnaOperationTypeRecurrent
//            //TODO:3:API Redesign: provide shape info
//            Shape: [H] 1D Vector where
//            - H is a number of the output nodes (rows),
//            Mode: {GnaTensorModeDefault, GnaTensorModeDisabled}
//            Type: {GnaDataTypeInt8, GnaDataTypeInt16, GnaDataTypeInt32, GnaDataTypeCompoundBias},
//        For GNA_OPERATION_FULLY_CONNECTED_FUSED_MULTIBIAS operation:
//            Shape: [H x N] 2D Matrix where:
//            - H is a number of the output nodes (rows),
//            - N is a number of the bias vectors (columns),
//            Mode: {GnaTensorModeDefault}
//            Type: {GnaDataTypeInt8, GnaDataTypeInt16, GnaDataTypeInt32},
//    */
//    struct GnaTensor * Biases;
//
//} GnaAffineOperation;  //TODO:3:API Redesign: Use Transformation name (operation=arithmetic)




    /// **
    // Specifies base affine operation.
    // */
    //GnaAffineOperation Affine;

    ////TODO:3:API Redesign: seems rather like output scaling/?
    ///**
    // Specifies scale factors for weights.
    // Required only for Weights.Type = GnaDataTypeInt8 and Biases->Type = GnaDataTypeCompoundBias.
    // Set Mode = GnaTensorModeDisabled for other cases.
    // //TODO:3:API Redesign: provide formula used
    // Valid values:
    //    Shape: 1D Vector //TODO:3:API Redesign: provide shape info
    //    Mode: {GnaTensorModeDefault}
    //    Type: {GNA_WEIGHT_SCALE_FACTOR},
    //*/
    //struct GnaTensor * WeightScaleFactors;



 //gna_layer_mode mode;            // Layer connection mode. //TODO:3:remove

    /**
     Specifies input tensor.
     Valid values:
        //TODO:3:API Redesign: provide shape info
        Common:
            Mode: {GnaTensorModeDefault, GnaTensorModeDisabled}
            Type: {GnaDataTypeInt8, GnaDataTypeInt16},
            Shape, where not stated otherwise:
                [NxW] 2D matrix where:
                 - N is a batch size (number of vectors)
                 - W is a number of vector elements
            Layout, where not stated otherwise:
                Column-major (interleaved), vectors are columns.
        For 1D GnaOperationTypeConvolution  operation:
            Shape: [W] 1D vector where:
             - W is a number of vector elements
             Layout: Row-major (flat)
        For 2D GnaOperationTypeConvolution  operation:
            Shape: [N x H x W x C] 4D Tensor where:
             - N is a batch size (number of vectors), currently only N=1 is supported
             - H is a height of each filter
             - W is a width of each filter
             - C is a depth of each filter
        For GnaOperationTypeTransposition (interleave), GnaOperationTypeRecurrent  operation:
             Layout: Row-major (flat)
    */
    /*struct GnaTensor * Inputs;*/

    /**
     Specifies output tensor.
     Valid values:
        //TODO:3:API Redesign: provide shape info
        Common:
            Mode: {GnaTensorModeDefault, GnaTensorModeDisabled}
            Type: {GnaDataTypeInt8, GnaDataTypeInt16, GnaDataTypeInt32}
            Shape, where not stated otherwise:
                [NxW] 2D matrix where:
                 - N is a batch size (number of vectors)
                 - W is a number of vector elements
            Layout, where not stated otherwise:
                Column-major (interleaved), vectors are columns.
        For 1D GnaOperationTypeConvolution  operation:
            Shape: [DxW] 2D matrix, where:
             - D is a number of feature maps
             - W is a number of elements of each feature map
             Layout: Column-major (interleaved)  //TODO:3:API Redesign: provide shape info
        For 2D GnaOperationTypeConvolution  operation:
            Shape: [N x H x W x C] 4D Tensor where:
             - N is a batch size (number of vectors), currently only N=1 is supported
             - H is a height of each feature maps
             - W is a width of each feature maps
             - C is a number of feature maps
        For GnaOperationTypeTransposition (deinterleave), GnaOperationTypeRecurrent  operation:
             Layout: Row-major (flat), vectors are rows.
    */
    //struct GnaTensor * Outputs;
    //TODO:3:API Redesign: add debug interface
    //void* pOutputsIntermediate;     // 4B Signed integer Auxiliary output buffer.



 ///**
    // Specifies filters (kernels) tensor.
    // Filters stored one after the other.
    // Note:
    //    For 2D GnaOperationTypeConvolution  operation each filter must start
    //    at address which is 16B aligned.
    // Valid values:
    //    Common:
    //        Mode: {GnaTensorModeDefault, GNA_TENSOR_CONSTANT_SCALAR}
    //        Type: {GnaDataTypeInt8, GnaDataTypeInt16},
    //    For 1D GnaOperationTypeConvolution  operation:
    //        Shape: [N x W] 2D matrix where://TODO:3:API Redesign: provide shape info
    //         - N is a number of filters
    //         - W is a width of each filter
    //    For 2D GnaOperationTypeConvolution  operation:
    //        Shape: [N x H x W x C] 4D Tensor where://TODO:3:API Redesign: provide shape info
    //         - N is a number of filters
    //         - H is a height of each filter
    //         - W is a width of each filter
    //         - C is a depth of each filter
    //*/
    //struct GnaTensor * Filters;
