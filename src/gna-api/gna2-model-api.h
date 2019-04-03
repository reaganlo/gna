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
struct Gna2Model;
struct Gna2Operation;
struct Gna2Shape;
struct Gna2Tensor;
struct Gna2CompoundBias;
struct Gna2WeightScaleFactor;
struct Gna2PwlSegment;

/**
 Creates and compiles the model for use with a given device.

 @note
 - The model has to be placed in user's memory, not allocated by Gna2MemoryAlloc().

 @param deviceIndex GNA device that will utilize the model.
 @param model Model descriptor which will govern the model creation.
 @param [out] modelId The model identifier assigned by GNA.
 @return Status of the operation.
 */
GNA2_API enum Gna2Status Gna2ModelCreate(
    uint32_t deviceIndex,
    struct Gna2Model const * model,
    uint32_t * modelId);

/**
 Releases model structures and request configurations

 @param modelId Model to release
 @return Status of the operation.
 */
GNA2_API enum Gna2Status Gna2ModelRelease(
    uint32_t modelId);

/**
 GNA data-flow Model.

 @see https://en.wikipedia.org/wiki/Dataflow_programming

 Data-flow model is a directed graph of nodes that represent
 Operations (Gna2Operation), either simple (e.g. addition) or composed
 (e.g. fused convolution).
 Operation nodes are connected with edges that represent Operands
 or data (Gna2Tensor).
 */
struct Gna2Model
{
    /**
     Number of Operations.
     Maximal number of operations depends on available device generation.
    */
    uint32_t NumberOfOperations;

    /**
     Operations which define the graph.
    */
    struct Gna2Operation * Operations;

    /**
     Maximal number of input tensors in a batch.
     Supported values: [1,8], depends on operation used.
    */
    uint32_t MaximumBatchSize;
};

/**
 Operation type.

 Defines type of single or composed operation.
 Composed operation is a "fused" transformation of a few chained operations,
 e.g. ::Gna2OperationTypeFullyConnectedAffine is defined as dot product, addition and activation function,
 */
enum Gna2OperationType
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
            - For 1D convolution operation: {::Gna2BiasModeDefault}
            - For 2D convolution operation: {::Gna2BiasModeDefault, ::Gna2BiasModePerStride}
        5. activationFunction

    Parameters:
        1. Gna2Shape zeroPadding [optional]:
            Supported only for 2D convolution.
            Specifies automatic input zero-padding dimensions.
             Used to maintain same input-output volume shape
             or when input dimensions have no common natural divider with filter and stride.
             Valid values:
                [0] When not used
                [W x H] 2D where: //TODO:3:API Redesign: provide shape info
                    - W is a number of 0s added each before and after each row of an input
                    - H is a number of 0s added each before and after each column of an input
        2. Gna2Shape concolutionStride [required]:
             Specifies filter stride shape.
             Valid values:
                For 1D convolution operation:
                    [W] 1D where: //TODO:3:API Redesign: provide shape info
                     - W is a number of elements to move in W dimension
                For 2D convolution operation:
                    [W x H] 2D where: //TODO:3:API Redesign: provide shape info
                     - W is a number of elements to move in W dimension
                     - H is a number of elements to move in H dimension
        3. Gna2BiasMode biasMode - Mode of bias operation [optional] (Gna2BiasMode):
            Supported values:
            - ::Gna2BiasModeDefault: normal operation
            - ::Gna2BiasModePerStride: TODO:3:API:elaborate
        4. Gna2PoolingMode poolingType of bias operation [optional] (Gna2PoolingMode):
        5. Gna2Shape poolingWindow:
            Specifies pooling window shape.
            Valid values:
            - For 1D convolution operation: [ W ] 1D where: //TODO:3:API Redesign: provide shape info
                - W is a width of window
            - For 2D convolution operation:
                - [ W x H ] 2D where:
                    - W is a width of window
                    - H is a height of window
        6. Gna2Shape poolingStride:
            Specifies pooling window stride dimensions.
            Valid values:
                - For 1D convolution operation: [W] 1D where: //TODO:3:API Redesign: provide shape info
                    - W is a number of elements to move in W dimension
                - For 2D convolution operation: [W x H] 2D where:
                    - W is a number of elements to move in W dimension
                    - H is a number of elements to move in H dimension
    */
    Gna2OperationTypeConvolution = 1,

    /**
    Copy operation.

    Operation:
        output = copy(input, shape)

    Operands:
        - 0 inputs
        - 1 outputs

    Parameters:
        - 0 Gna2Shape copyParams [required]:
             Specifies dimensions of copied sub-tensor.
             Valid values:
                [W x H] 2D where: //TODO:3:API Redesign: provide shape info
                 - W is a number of elements to copy in W dimension
                 - H is a number of elements to copy in H dimension
    */
    Gna2OperationTypeCopy = 2,

    /**
    Fully connected affine operation composed with activation function.

    Operation:
    - a: outputs = activation(((inputs x weights) + biases), activationFunction)
    - b: outputs = activation(((inputs x (weightScaleFactors * weights)) + biases[biasVectorIndex]), activationFunction)

    Where:
    - activation is optional
    - weightScaleFactors is optional, required only for ::Gna2BiasModeGrouping Mode.

    Operands:
    - 0 inputs
    - 1 outputs
    - 2 filters
    - 3 biases
    - 4 activationFunction
    - 5 weightScaleFactors

    Parameters:
        - 0 Gna2BiasMode Mode of bias operation [optional] (Gna2BiasMode):
            Supported values:
            -::Gna2BiasModeDefault: normal operation
            -::Gna2BiasModeGrouping: Special optimized case // TODO:3:API: elaborate
                Requires weightScaleFactors, Bias vector index
        - 1 uint32_t biasVectorIndex [optional]:
            Index of the bias vector used for this operation.
            Supported values:
            -[0, N-1]: Where N is a number of all bias vectors in biases tensor.
             Default is 0.
            -GNA2_DEFAULT: is equivalent of 0.
    */
    Gna2OperationTypeFullyConnectedAffine = 3,

    /**
    Element wise affine operation composed with activation function and pooling.

    Weights are diagonal matrix, represented by 1D vector.
    output = activation((times(input, weights) + bias), pwl)
    Used e.g. for scaling input tensor.
    */
    Gna2OperationTypeElementWiseAffine = 4,

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
    Gna2OperationTypeGmm = 5,

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
    Gna2OperationTypeRecurrent = 6,

    /**
     Tensor transposition operation.

     output<layout> = transpose(input<layout>)
     //TODO:3:API: use Tensor.Layout in input and output tensor to determine type of transposition interleaved/flat/other
    */
    Gna2OperationTypeTransposition = 7,

    /**
    Control-flow operation with threshold parameter.
    */
    Gna2OperationTypeTreshold = 8,

    //GNA2_OPERATION_TYPE_COUT,      // Number of Layer operation types.
    //// TODO:3: use more generic operation type and determine specialization via parameters
    //Gna2OperationTypeTransposition (interleave),               // Auxiliary 2D tensor transpose operation (flat to interleave). No casting, always set Operations to null.
    //Gna2OperationTypeTransposition (deinterleave),             // Auxiliary 2D tensor transpose operation (interleave to flat). No casting, always set Operations to null.
    //// NOT POR
    //GNA2_OPERATION_CONVOLUTIONAL_2D_ADDITION,
    //GNA2_OPERATION_CONVOLUTIONAL_2D_CONVERSION,
    //GNA2_OPERATION_POOLING_2D,
    //// exemplary not existing operations
    //GNA2_OPERATION_NEGATION, // unary, output = !(input)
    //GNA2_OPERATION_ADDITION, // binary, output = (input + operand)
    //GNA2_OPERATION_DOT_PRODUCT,// binary, output = (input x operand)
    //GNA2_OPERATION_MULTIPLICATION,// binary, output = (input * operand)
};

/**
 Operation configuration.

 For composed operations Inputs and Outputs are always specified per whole
 operation, i.e. inputs for first operation and output from the last operation.
 Intermediate results and buffer are not directly accessible for composed
 operations.

 @see Gna2ModelOperationInit() That simplifies operation creation.
 */
struct Gna2Operation
{
    /**
     Type of executed operation.
     */
    enum Gna2OperationType Type;

    /**
     Operands that operation is executing on.

     Number of operands is defined by Type.
     @see Gna2OperationType.

     @note
        Set unused Operands pointers to NULL.
    */
    struct Gna2Tensor const ** Operands;

    /**
     Number of Operands that are actually provided.
     */
    uint32_t NumberOfOperands;

    /**
     Constant parameters providing additional operation configuration.
     Number and types of parameters are defined by operation Type.
     Currently parameters can be enumerations, Gna2Shape or single integers.
     @see Gna2OperationType.
    */
    void ** Parameters;

    /**
     Number of Parameters that are actually provided.
     */
    uint32_t NumberOfParameters;
};

/**
 Maximal number of supported shape dimensions.
 */
#define GNA2_SHAPE_MAXIMUM_NUMBER_OF_DIMENSIONS 8

/**
 Shape specifying dimension values.
*/
struct Gna2Shape
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
    uint32_t Dimensions[GNA2_SHAPE_MAXIMUM_NUMBER_OF_DIMENSIONS];
};

/**
 Tensor mode.

 Specifies interpretation or usage of tensor data.
 @note
 - Not all modes are supported by all data types and operations.
 - Only 16MSB are used.
 */
enum Gna2TensorMode
{
    /**
     Data interpreted as read-write tensor of type specified by Gna2DataType.
     */
    Gna2TensorModeDefault = GNA2_DEFAULT,

    /**
     Data interpreted as single constant scalar of type specified by Gna2DataType.
     */
    Gna2TensorModeConstantScalar = 0x010000,

    /**
     Data not used, set buffer to NULL.
     */
    Gna2TensorModeDisabled = GNA2_DISABLED,

    ///**
    // Indicates Data Mode property is not supported for given entity.
    // */
    //Gna2TensorModeNotSupported = GNA2_NOT_SUPPORTED,
};

/**
 Type and precision of data used.

 @note
 - Not all type are supported by all data modes and operations.
 - Only 16LSB are used.
 */
enum Gna2DataType
{
    /**
     Data type not specified, can be used for disabled data.
     */
    Gna2DataTypeVoid = GNA2_DISABLED,

    /**
     1 Bit Boolean / binary type.
     Currently not supported.
     */
    Gna2DataTypeBoolean = 1,

    /**
     4 bit Signed Integer.
     Currently not supported.
     */
    Gna2DataTypeInt4 = 2,

    /**
     1 Byte Signed Integer, use int8_t data.
     */
    Gna2DataTypeInt8 = 3,

    /**
     2 Byte Signed Integer, use int16_t data.
     */
    Gna2DataTypeInt16 = 4,

    /**
     4 Byte Signed Integer, use int32_t data.
     */
    Gna2DataTypeInt32 = 5,

    /**
     8 Byte Signed Integer, use int64_t data.
     */
    Gna2DataTypeInt64 = 6,

    /**
     4 bit Unsigned Integer.
     Currently not supported.
     */
    Gna2DataTypeUint4 =7,

    /**
     1 Byte Unsigned Integer, use uint8_t data.
     */
    Gna2DataTypeUint8 = 8,

    /**
     2 Byte Unsigned Integer, use uint16_t data.
     */
    Gna2DataTypeUint16 = 9,

    /**
     4 Byte Unsigned Integer, use uint32_t data.
     */
    Gna2DataTypeUint32 = 10,

    /**
     8 Byte Unsigned Integer, use uint64_t data.
     */
    Gna2DataTypeUint64 = 11,

    /**
     Rich bias data type, use Gna2CompoundBias data.
     Used only for bias tensor and with ::Gna2DataTypeInt8 weight mode.
     */
    Gna2DataTypeCompoundBias = 12,

    /**
     PWL Activation function segment data type, use Gna2PwlSegment data.
     Used only for PWL Activation function segment tensor.
     */
    Gna2DataTypePwlSegment = 13,

    /**
     Weight scale factor type, use ::Gna2OperationTypeFullyConnectedAffine data.
     Used only for Gna2WeightScaleFactor tensor.
     */
    Gna2DataTypeWeightScaleFactor = 14,

    // /**
    // Indicates Data Type property is not supported for given entity.
    // */
    //Gna2DataTypeNotSupported = GNA2_NOT_SUPPORTED,
};


/**
 Tensor used as operation operand.

 Valid parameters:
 - Input Tensor:
    - Common:
        - #Mode: {::Gna2TensorModeDefault, ::Gna2TensorModeDisabled}
        - #Type: {::Gna2DataTypeInt8, ::Gna2DataTypeInt16},
        - #Shape: [ N x W ] 2D matrix (if not stated otherwise) where:
            - N is a batch size (number of vectors)
            - W is a number of vector elements
        - #Layout, where not stated otherwise: Column-major (interleaved), vectors are columns.
    - For 1D ::Gna2OperationTypeConvolution operation:
        - #Shape: [ W ] 1D vector where:
            - W is a number of vector elements
        - #Layout: Row-major (flat)
    - For 2D ::Gna2OperationTypeConvolution  operation:
        - #Shape: [N x H x W x C] 4D Tensor where:
            - N is a batch size (number of vectors), currently only N=1 is supported
            - H is a height of each filter
            - W is a width of each filter
            - C is a depth of each filter
    - For ::Gna2OperationTypeTransposition (interleave), ::Gna2OperationTypeRecurrent operation:
        - #Layout: Row-major (flat)
 - Output Tensor:
    - Common:
        - #Mode: {::Gna2TensorModeDefault, ::Gna2TensorModeDisabled}
        - #Type: {::Gna2DataTypeInt8, ::Gna2DataTypeInt16, ::Gna2DataTypeInt32}
        - #Shape, where not stated otherwise, [ N x W ] 2D matrix where:
            - N is a batch size (number of vectors)
            - W is a number of vector elements
        - #Layout, where not stated otherwise: Column-major (interleaved), vectors are columns.
    - For 1D ::Gna2OperationTypeConvolution  operation:
        - #Shape: [DxW] 2D matrix, where:
            - D is a number of feature maps
            - W is a number of elements of each feature map
        - #Layout: Column-major (interleaved)  //TODO:3:API Redesign: provide shape info
    - For 2D ::Gna2OperationTypeConvolution operation:
        - #Shape: [N x H x W x C] 4D Tensor where:
            - N is a batch size (number of vectors), currently only N = 1 is supported
            - H is a height of each feature maps
            - W is a width of each feature maps
            - C is a number of feature maps
    - For ::Gna2OperationTypeTransposition (deinterleave), ::Gna2OperationTypeRecurrent operation:
        - #Layout: Row-major (flat), vectors are rows.
 - Bias tensor.
    - Common:
        - #Mode: {::Gna2TensorModeDefault, ::Gna2TensorModeConstantScalar}
        - #Type: {::Gna2DataTypeInt8, ::Gna2DataTypeInt16, ::Gna2DataTypeInt32},
    - #Shape:
        - For ::Gna2BiasModeDefault [ W ] 1D vector where:
            - W is a number of elements (same as number of outputs or filters)
        - For bias, only for ::Gna2OperationTypeFullyConnectedAffine operation
          [ N x W ] 2D Matrix where:
            - N is a number of bias vectors
            - W is a number of elements in vector, (same as number of outputs or filters)
        - For ::Gna2BiasModePerStride, valid only for 2D ::Gna2OperationTypeConvolution  operation:
          [ H x W x C ] 3D tensor same as filter shape
 - Weight Scale Factor Tensor
    //TODO:3:API Redesign: seems rather like output scaling/?
     - Specifies scale factors for weights.
     - Required only for Weights->Gna2Tensor::Type = ::Gna2DataTypeInt8 and Biases->Gna2Tensor::Type = ::Gna2DataTypeCompoundBias.
     - Set Gna2Tensor::Mode = ::Gna2TensorModeDisabled for other cases.
     //TODO:3:API Redesign: provide formula used
     - Valid values:
        - #Shape: 1D Vector //TODO:3:API Redesign: provide shape info
        - #Mode: {::Gna2TensorModeDefault}
        - #Type: {::Gna2DataTypeWeightScaleFactor},
*/
struct Gna2Tensor
{
    /**
     Specifies tensor dimensions.
    */
    struct Gna2Shape Shape;

    /**
     Mode of tensor interpretation.
     Use ::Gna2TensorModeDefault as default.
    */
    enum Gna2TensorMode Mode;

    /**
     Data layout or format in memory.

    - Specifies order of dimensions, i.e. how Gna2Shape::Dimensions are interpreted.
      Size of layout array must be the same as #Shape Gna2Shape::NumberOfDimensions.
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
    - Required for Gna2Operation::Type = ::Gna2OperationTypeTransposition or ::Gna2OperationTypeGmm.
      Optional (set zeros) for other operations.
    */
    char Layout[GNA2_SHAPE_MAXIMUM_NUMBER_OF_DIMENSIONS];

    /**
     Type of tensor data.
     */
    enum Gna2DataType Type;

    /**
     Data buffer.
     Must be specified before enqueueing request, during model or request config creation.
     */
    // TODO:3:API redesign elaborate more.
    void * Data;
};

/**
 * Mode of bias usage.
 */
enum Gna2BiasMode
{
    /**
     Bias is added per output for affine transformations
     and per filter for convolutional.
    */
    Gna2BiasModeDefault = GNA2_DEFAULT,

    /**
     Bias is added per each filter stride of convolutional transformation.
    */
    Gna2BiasModePerStride = 1,

    /**
     Optimized bias usage mode for operations that produce matrix of biases
     as output of an operation and consume single vectors from that matrix
     in a group of consecutive operations e.g. LSTM.
     Bias from selected vector (group) is added per output for affine
     transformations.
     Used with 2D Bias Tensor,
     only for ::Gna2OperationTypeFullyConnectedAffine operation.
     @see Gna2Tensor.
    */
    Gna2BiasModeGrouping = 2,

    // / **
    // Indicates Bias Mode is not supported for given entity.
    // */
    //Gna2BiasModeNotSupported = GNA2_NOT_SUPPORTED,
};

/**
 Mode of pooling operation.
 */
enum Gna2PoolingMode
{
    /**
     Pooling operation is disabled.
     */
    Gna2PoolingModeDisabled = GNA2_DISABLED,

    /**
     Max pooling is used.
     */
    Gna2PoolingModeMax = 1,

    /**
     Sum pooling is used.
     */
    Gna2PoolingModeSum = 2,

    /*Gna2PoolingNotSupported = GNA2_NOT_SUPPORTED,*/
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

 Used for Weights->Gna2Tensor::Type = ::Gna2DataTypeInt8
 and Biases->Gna2Tensor::Type = ::Gna2DataTypeInt16 only.
 Used with ::Gna2DataTypeCompoundBias.

 @note
    Data format is read directly by the accelerator.
 */
struct Gna2CompoundBias
{
    /**
     Bias (constant) value that is added to result of the dot product
     of the weight matrix row and the input vector, same as "regular" bias.
     */
    int32_t Bias;

    /**
     Weight scaling factor that elements of the corresponding weight matrix row are multiplied by.
     Utilized when Weights->Gna2Tensor::Type = ::Gna2DataTypeInt8 is used.
     */
    uint8_t Multiplier;

    /**
     Padding to 8B only, field is not used.
     */
    uint8_t Reserved[3];
};

static_assert(8 == sizeof(Gna2CompoundBias), "Invalid size of Gna2CompoundBias");

/**
 Weight element scaling factor.

 Used with ::Gna2DataTypeWeightScaleFactor
 and ::Gna2OperationTypeFullyConnectedAffine (b variant).
 */
 struct Gna2WeightScaleFactor
{
    /**
     Padding to 4B only, field is not used.
     */
    uint8_t Reserved0[4];

    /**
     Weight scaling factor.
     @see Gna2CompoundBias::Multiplier.
     */
    uint8_t Multiplier;

    /**
     Padding to 8B only, field is not used.
     */
    uint8_t Reserved1[3];
};

static_assert(8 == sizeof(Gna2WeightScaleFactor), "Invalid size of Gna2WeightScaleFactor");

/**
 Piecewise-linear activation function segment.

 Defines a single segment of a piecewise linear activation function.
 Used with ::Gna2DataTypePwlSegment.
 For a given input x, the appropriate segment is fetched and the output y is calculated as
 follows:
    slopeScale = xBase & 0x03
    xBaseValue = xBase & 0xFFFFFFFC
    y = ((x - xBaseValue) * slope) >> slopeScale + yBase
 */
struct Gna2PwlSegment
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

static_assert(8 == sizeof(Gna2PwlSegment), "Invalid size of Gna2PwlSegment");

/**************************************************************************//**
 @}

 @addtogroup GNA2_MODEL_DEBUG_API Model Debugging

 Debugging functions that simplify debugging GNA Model issues.

 @{
 *****************************************************************************/

struct Gna2ModelItem;
struct Gna2ModelError;

/**
 Retrieves information on error during model creation.

 Can be called after Gna2ModelCreate() have failed (e.g., returned ::Gna2StatusInvalidModel status).

 @note
 The function should be called directly after Gna2ModelCreate() in the same thread.

 @param [out] error The detailed description of model issue.
 @return Status of fetching the model error.
    @retval ::Gna2StatusSuccess The error has been fetched successfully.
    @retval ::Gna2StatusUnknownError No issue to report.
    @retval ::Gna2StatusNullargnotallowed The error pointer was NULL.
 */
GNA2_API enum Gna2Status Gna2ModelGetLastError(struct Gna2ModelError * error);

/**
 Gets message with description of the last model error.

 @note
 TODO:3:API: provide maximum message size

 @param [out] messageBuffer User allocated buffer for the message.
 @param [in] messageBufferSize The size of the messageBuffer in bytes.
        The message is maximum X characters/bytes long.
        Message is truncated to messageBufferSize if it is longer than messageBufferSize characters.
 @return Status of fetching the model error.
    @retval ::Gna2StatusSuccess The error was fully serialized into the messageBuffer.
    @retval ::Gna2StatusUnknownError No issue to report.
    @retval ::Gna2StatusErrResources The messageBuffer is too small. The message was truncated.
    @retval ::Gna2StatusNullargnotallowed The messageBuffer was NULL or messageBufferSize was 0.
 */
GNA2_API enum Gna2Status Gna2ModelGetLastErrorMessage(char * messageBuffer,
    uint32_t messageBufferSize);

//TODO:3:API:make documentation consistent: nouns vs verbs.


/**
 Determines the type of a model item i.e. model or operand property.

 Used for model debugging and Capability Query API.
 Most of the items are 1 to 1 mapping of data-flow model properties.
 */
enum Gna2ItemType
{
    /**
     Model context is not applicable or unnecessary.
     */
    Gna2ItemTypeNone = GNA2_DISABLED,

    /**
     Gna2Model::NumberOfOperations.
     */
    Gna2ItemTypeModelNumberOfOperations = 0,

    /**
     Gna2Model::Operations array.
     */
    Gna2ItemTypeModelOperations = 1,

    /**
     Gna2Model::MaximumBatchSize.
     */
    Gna2ItemTypeModelMaximumBatchSize = 2,

    /**
     Gna2Model::Operations[x]->Gna2Operation::Type.
     */
    Gna2ItemTypeOperationType = 3,

    /**
     Gna2Model::Operations[x]->Gna2Operation::Operands array.
     */
    Gna2ItemTypeOperationOperands = 4,

    /**
     Gna2Model::Operations[x]->Gna2Operation::NumberOfOperands.
     */
    Gna2ItemTypeOperationNumberOfOperands = 5,

    /**
     Gna2Model::Operations[x]->Gna2Operation::Parameters array.
     */
    Gna2ItemTypeOperationParameters = 6,

    /**
     Gna2Model::Operations[x]->Gna2Operation::NumberOfParameters.
     */
    Gna2ItemTypeOperationNumberOfParameters = 7,

    /**
     Gna2Model::Operations[x]->Gna2Operation::Operands[y]->Gna2Tensor::Mode.
     */
    Gna2ItemTypeOperandMode = 8,

    /**
     Gna2Model::Operations[x]->Gna2Operation::Operands[y]->Gna2Tensor::Layout.
     */
    Gna2ItemTypeOperandLayout = 9,

    /**
     Gna2Model::Operations[x]->Gna2Operation::Operands[y]->Gna2Tensor::Type.
     */
    Gna2ItemTypeOperandType = 10,

    /**
     Gna2Model::Operations[x]->Gna2Operation::Operands[y]->Gna2Tensor::Data.
     */
    Gna2ItemTypeOperandData = 11,

    /**
     Gna2Model::Operations[x]->Gna2Operation::Parameters[z]->Parameter, can be of type Gna2Shape, enumeration or integer.
     */
    Gna2ItemTypeParameter = 12,

    /**
     Gna2Model::Operations[x]->{Gna2Tensor; Parameter}->Gna2Shape::NumberOfDimensions.
     */
    Gna2ItemTypeShapeNumberOfDimensions = 13,

    /**
     Gna2Model::Operations[x]->{Gna2Tensor; Parameter}->Gna2Shape::Dimensions.
     */
    Gna2ItemTypeShapeDimensions = 14,

    /**
     Internal model item, that is a derivative of other model parameters.

     Used only for model debugging.
     When set detailed issue source and description will be reported
     via error message.
     */
    Gna2ItemTypeInternal = 15,
};

/**
 Number of additional properties for Gna2ModelItem.
 */
#define GNA2_MODEL_ITEM_NUMBER_OF_PROPERTIES 4

/**
 Determines exact model item for capability query and model error.

 Item location is determined top-down.
 E.g. for querying if operation type is supported set:
 - #Type to ::Gna2ItemTypeOperationType,
 - #Operation to desired operation type,
 - ::GNA2_DISABLED for the remaining fields.
 */
struct Gna2ModelItem
{
    /**
     Type of a model item.

     ::Gna2ItemTypeNone if not applicable.
     */
    enum Gna2ItemType Type;

    /**
     Index (0 based) of an operation.

     ::GNA2_DISABLED if not applicable.
     */
    enum Gna2OperationType Operation;

    /**
     Index (0 based) of an operation.

     ::GNA2_DISABLED if not applicable.

     @note Not applicable for Capability querying.
     */
    int32_t OperationIndex;

    /**
     Index (0 based) of an operand.

     ::GNA2_DISABLED if not applicable.
     Requires Operation or OperationIndex to be set.
     */
    int32_t OperandIndex;

    /**
     Index (0 based) of a parameter.

     ::GNA2_DISABLED if not applicable.
     Requires Operation or OperationIndex to be set.
     */
    int32_t ParameterIndex;

    /**
     Index of a dimension of Operand or Parameter.

     ::GNA2_DISABLED if not applicable.
     Requires OperandIndex or ParameterIndex to be set.
     */
    int32_t ShapeDimensionIndex;

    /**
     Additional properties for determining complex and future items.

     Number and type of Properties is determined by #Type.
     ::GNA2_DISABLED if not applicable.
     */
    int32_t Properties[GNA2_MODEL_ITEM_NUMBER_OF_PROPERTIES];
};

/**
 Type of model item error.

 Helps identifying root cause of model issue.
 */
enum Gna2ErrorType
{
    /**
     TODO:3:API: document
     */
    Gna2ErrorTypeNone = GNA2_DEFAULT,

    /**
     TODO:3:API: document
     */
    Gna2ErrorTypeNotTrue = -1,

    /**
     TODO:3:API: document
     */
    Gna2ErrorTypeNotFalse = -2,

    /**
     TODO:3:API: document
     */
    Gna2ErrorTypeNullNotAllowed = -3,

    /**
     TODO:3:API: document
     */
    Gna2ErrorTypeNullRequired = -4,

    /**
     TODO:3:API: document
     */
    Gna2ErrorTypeBelowRange = -5,

    /**
     TODO:3:API: document
     */
    Gna2ErrorTypeAboveRange = -6,

    /**
     TODO:3:API: document
     */
    Gna2ErrorTypeNotEqual = -7,

    /**
     TODO:3:API: document
     */
    Gna2ErrorTypeNotGtzero = -8,

    /**
     TODO:3:API: document
     */
    Gna2ErrorTypeNotZero = -9,

    /**
     TODO:3:API: document
     */
    Gna2ErrorTypeNotOne = -10,

    /**
     TODO:3:API: document
     */
    Gna2ErrorTypeNotInSet = -11,

    /**
     TODO:3:API: document
     */
    Gna2ErrorTypeNotMultiplicity = -12,

    /**
     TODO:3:API: document
     */
    Gna2ErrorTypeNotSuccess = -13,

    /**
     TODO:3:API: document
     */
    Gna2ErrorTypeNotAligned = -14,

    /**
     Some operation argument was not provided.
     */
    Gna2ErrorTypeArgumentMissing = -15,

    /**
     Given operation argument was invalid or unexpected.
     */
    Gna2ErrorTypeArgumentInvalid= -16,

    /**
     Runtime error occurred during model creation.
     */
    Gna2ErrorTypeRuntime = -17,

    /**
     Unable to determine the root cause of the issue.
     */
    Gna2ErrorTypeOther = GNA2_NOT_SUPPORTED,
};

/**
 Describes the error that caused creating model to fail.
 */
struct Gna2ModelError
{
    /**
     Which item is the source of the problem.
     */
    struct Gna2ModelItem Source;

    /**
     Why item is faulty.
     */
    enum Gna2ErrorType Reason;

    /**
     Incorrect value given.
     */
    int64_t Value;
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
    1. Gna2Operation::Type is set to type.
    2. Gna2Operation::NumberOfOperands is set to value determined by Gna2Operation::Type.
    3. Gna2Operation::Operands array of pointers is allocated by userAllocator.
        Number of array elements is Gna2Operation::NumberOfOperands.
        All pointers are set to NULL.
    2. Gna2Operation::NumberOfParameters is set to value determined by Gna2Operation::Type.
    3. Gna2Operation::Parameters array of pointers is allocated by userAllocator.
        Number of array elements is Gna2Operation::NumberOfParameters.
        All pointers are set to NULL.

 @warning
    User is responsible for releasing allocated Gna2Operation::Operands
    and Gna2Operation::Parameters buffers.

 @param operation The affected operation.
 @param type The type of executed operation.
 @param userAllocator User provided memory allocator.
 @return Status of the operation.
 */
GNA2_API enum Gna2Status Gna2ModelOperationInit(
    struct Gna2Operation * operation,
    enum Gna2OperationType type,
    Gna2UserAllocator userAllocator);

/**
 Gets the size in bytes of given data type.

 Useful for calculating the sizes of memory buffers.

 @param type The type of the data.
 @return Size in bytes of given data type.
 */
GNA2_API uint32_t Gna2DataTypeGetSize(enum Gna2DataType type);

/**
 Gets the total number of elements of the given shape.

 Useful for calculating the sizes of memory buffers.

 @param shape The shape to calculate the number of elements.
 @return Total number of elements.
 */
GNA2_API uint32_t Gna2ShapeGetNumberOfElements(struct Gna2Shape const * shape);

/**
 Gets the size in bytes of entire tensor data.

 Useful for calculating the sizes of memory buffers.

 @param tensor The tensor to calculate the size of.
 @return Size in bytes of given tensor.
 */
GNA2_API uint32_t Gna2TensorGetSize(struct Gna2Tensor const * tensor);

/**
 Helper function that simplifies common Gna2Shapes creation.

 @return Complete Gna2Shape representing scalar.
 */
GNA2_API struct Gna2Shape Gna2ShapeInitScalar();

/**
 Helper function that simplifies common Gna2Shapes creation.

 @note
 No arguments validation is performed.

 @param x Size of a vector.
 @return Complete Gna2Shape representing 1D vector dimension.
 */
GNA2_API struct Gna2Shape Gna2ShapeInit1D(uint32_t x);

/**
 Helper function that simplifies common Gna2Shapes creation.

 @note
 No arguments validation is performed.

 @param x First matrix dimension.
 @param y Second matrix dimension.
 @return Complete Gna2Shape representing 2D matrix dimensions.
 */
GNA2_API struct Gna2Shape Gna2ShapeInit2D(uint32_t x, uint32_t y);

/**
 Helper function that simplifies common Gna2Shapes creation.

 @note
 No arguments validation is performed.

 @param x First tensor dimension.
 @param y Second tensor dimension.
 @param z Third tensor dimension.
 @return Complete Gna2Shape representing 3D tensor dimensions.
 */
GNA2_API struct Gna2Shape Gna2ShapeInit3D(uint32_t x, uint32_t y, uint32_t z);

/**
 Helper function that simplifies common Gna2Shapes creation.

 @note
 No arguments validation is performed.

 @param n First tensor dimension, usually representing batch size or number of filters.
 @param x Second tensor dimension.
 @param y Third tensor dimension.
 @param z Fourth tensor dimension.
 @return Complete Gna2Shape representing 3D tensor dimensions.
 */
GNA2_API struct Gna2Shape Gna2ShapeInit4D(uint32_t n, uint32_t x, uint32_t y,
    uint32_t z);


GNA2_API struct Gna2Tensor Gna2TensorInit1D(uint32_t x, enum Gna2DataType type,
    void * data);

GNA2_API struct Gna2Tensor Gna2TensorInit2D(uint32_t x, uint32_t y,
    enum Gna2DataType type, void * data);

GNA2_API struct Gna2Tensor Gna2TensorInit3D(uint32_t x, uint32_t y, uint32_t z,
    enum Gna2DataType type, void * data);

GNA2_API struct Gna2Tensor Gna2TensorInit4D(uint32_t n, uint32_t x, uint32_t y,
    uint32_t z, enum Gna2DataType type, void * data);

GNA2_API struct Gna2Tensor Gna2TensorInitDisabled();

GNA2_API struct Gna2Tensor Gna2TensorInitScalar(enum Gna2DataType type, void * data);

GNA2_API struct Gna2Tensor Gna2TensorInitActivation(uint32_t numberOfSegments,
    struct Gna2PwlSegment * segments);


GNA2_API struct Gna2Operation Gna2OperationInitFullyConnectedAffine(
    struct Gna2Tensor * inputs, struct Gna2Tensor * outputs,
    struct Gna2Tensor * weights, struct Gna2Tensor * biases,
    struct Gna2Tensor * activation);

GNA2_API struct Gna2Operation Gna2OperationInitElementWiseAffine(
    struct Gna2Tensor * inputs, struct Gna2Tensor * outputs,
    struct Gna2Tensor * weights, struct Gna2Tensor * biases,
    struct Gna2Tensor * activation);

GNA2_API struct Gna2Operation Gna2OperationInitFullyConnectedBiasGrouping(
    struct Gna2Tensor * inputs, struct Gna2Tensor * outputs,
    struct Gna2Tensor * weights, struct Gna2Tensor * biases,
    struct Gna2Tensor * activation,
    struct Gna2Tensor * weightScaleFactors,
    enum Gna2BiasMode* biasMode,
    uint32_t* biasVectorIndex);

GNA2_API struct Gna2Operation Gna2OperationInitRecurrent(
    struct Gna2Tensor * inputs, struct Gna2Tensor * outputs,
    struct Gna2Tensor * weights, struct Gna2Tensor * biases,
    struct Gna2Tensor * activation,
    uint32_t* delay);


GNA2_API struct Gna2Operation Gna2OperationInitConvolution(
    struct Gna2Tensor * inputs, struct Gna2Tensor * outputs,
    struct Gna2Tensor * filters, struct Gna2Tensor * biases,
    struct Gna2Tensor * activation,
    struct Gna2Shape * zeroPadding,
    struct Gna2Shape * concolutionStride,
    enum Gna2BiasMode * biasMode);

GNA2_API struct Gna2Operation Gna2OperationInitConvolutionFused(
    struct Gna2Tensor * inputs, struct Gna2Tensor * outputs,
    struct Gna2Tensor * filters, struct Gna2Tensor * biases,
    struct Gna2Tensor * activation,
    struct Gna2Shape * zeroPadding,
    struct Gna2Shape * concolutionStride,
    enum Gna2BiasMode * biasMode,
    enum Gna2PoolingMode * poolingMode,
    struct Gna2Shape * poolingWindow,
    struct Gna2Shape * poolingStride);

GNA2_API struct Gna2Operation Gna2OperationInitPooling(
    struct Gna2Tensor * inputs, struct Gna2Tensor * outputs,
    struct Gna2Tensor * activation,
    struct Gna2Shape * zeroPadding,
    enum Gna2PoolingMode * poolingMode,
    struct Gna2Shape * poolingWindow,
    struct Gna2Shape * poolingStride);

GNA2_API struct Gna2Operation Gna2OperationInitCopy(
    struct Gna2Tensor * inputs, struct Gna2Tensor * outputs,
    struct Gna2Shape * copyParams);

GNA2_API struct Gna2Operation Gna2OperationInitTranspose(
    struct Gna2Tensor * inputs, struct Gna2Tensor * outputs);

//TODO:3:API define
GNA2_API struct Gna2Operation Gna2OperationInitGmm(
    struct Gna2Tensor * inputs, struct Gna2Tensor * outputs,
    struct Gna2Tensor * means,
    struct Gna2Tensor * inverseCovariances,
    struct Gna2Tensor * consts,
    uint32_t * maximumScore);

GNA2_API struct Gna2Operation Gna2OperationInitGmInterleaved(
    struct Gna2Tensor * inputs, struct Gna2Tensor * outputs,
    struct Gna2Tensor * interleavedTensors,
    uint32_t * maximumScore);

// exemplary not existing operations
GNA2_API struct Gna2Operation Gna2OperationInitNegation(
    struct Gna2Tensor * inputs, struct Gna2Tensor * outputs);

// exemplary not existing operations
GNA2_API struct Gna2Operation Gna2OperationInitDotProduct(struct Gna2Tensor * inputs,
    struct Gna2Tensor * operand, struct Gna2Tensor * outputs);

#endif // __GNA2_MODEL_API_H

/**
 @}
 @}
 @}
 */



///** Piecewise-linear activation function (PWL) details */
//typedef struct _Gna2ActivationOperation
//{
//    /**
//     Specifies PWL activation function segment vector.
//     Set Mode = Gna2TensorModeDisabled to disable activation.
//     Segments have to be contiguous.
//     Valid values:
//        Shape: [W] 1D vector where:
//            - W is a number of piecewise-linear segments
//        Mode: {Gna2TensorModeDefault, Gna2TensorModeDisabled}
//        Type: {GNA2_PWL_SEGMENT},
//    */
//    struct Gna2Tensor * Pwl;
//
//} Gna2ActivationOperation;


//// TODO: 3.0 verify Gna2TensorModeDisabled and GNA2_TENSOR_CONSTANT_SCALAR usage
///** Affine function details */
//// Gna2AffineOperation: Gna2AffineTransform = I x Weights + Biases
//typedef struct _Gna2AffineOperation
//{
//    /**
//     Specifies weight tensor.
//     Valid values:
//        Common:
//            Mode: {Gna2TensorModeDefault, GNA2_TENSOR_CONSTANT_SCALAR}
//            Type: {Gna2DataTypeInt8, Gna2DataTypeInt16},
//        For ::Gna2OperationTypeFullyConnectedAffine, Gna2OperationTypeRecurrent
//            Shape: [WxH] 2D Matrix //TODO:3:API Redesign: provide shape info
//        For ::Gna2OperationTypeElementWiseMultiplication
//            Shape: [W] 1D Vector
//    */
//    struct Gna2Tensor * Weights;
//
//    /**
//     Specifies bias tensor.
//     Valid values:
//        For ::Gna2OperationTypeFullyConnectedAffine,
//              ::Gna2OperationTypeElementWiseMultiplication, Gna2OperationTypeRecurrent
//            //TODO:3:API Redesign: provide shape info
//            Shape: [H] 1D Vector where
//            - H is a number of the output nodes (rows),
//            Mode: {Gna2TensorModeDefault, Gna2TensorModeDisabled}
//            Type: {Gna2DataTypeInt8, Gna2DataTypeInt16, Gna2DataTypeInt32, Gna2DataTypeCompoundBias},
//        For GNA2_OPERATION_FULLY_CONNECTED_FUSED_MULTIBIAS operation:
//            Shape: [H x N] 2D Matrix where:
//            - H is a number of the output nodes (rows),
//            - N is a number of the bias vectors (columns),
//            Mode: {Gna2TensorModeDefault}
//            Type: {Gna2DataTypeInt8, Gna2DataTypeInt16, Gna2DataTypeInt32},
//    */
//    struct Gna2Tensor * Biases;
//
//} Gna2AffineOperation;  //TODO:3:API Redesign: Use Transformation name (operation=arithmetic)




    /// **
    // Specifies base affine operation.
    // */
    //Gna2AffineOperation Affine;

    ////TODO:3:API Redesign: seems rather like output scaling/?
    ///**
    // Specifies scale factors for weights.
    // Required only for Weights.Type = Gna2DataTypeInt8 and Biases->Type = Gna2DataTypeCompoundBias.
    // Set Mode = Gna2TensorModeDisabled for other cases.
    // //TODO:3:API Redesign: provide formula used
    // Valid values:
    //    Shape: 1D Vector //TODO:3:API Redesign: provide shape info
    //    Mode: {Gna2TensorModeDefault}
    //    Type: {GNA2_WEIGHT_SCALE_FACTOR},
    //*/
    //struct Gna2Tensor * WeightScaleFactors;



 //gna_layer_mode mode;            // Layer connection mode. //TODO:3:remove

    /**
     Specifies input tensor.
     Valid values:
        //TODO:3:API Redesign: provide shape info
        Common:
            Mode: {Gna2TensorModeDefault, Gna2TensorModeDisabled}
            Type: {Gna2DataTypeInt8, Gna2DataTypeInt16},
            Shape, where not stated otherwise:
                [NxW] 2D matrix where:
                 - N is a batch size (number of vectors)
                 - W is a number of vector elements
            Layout, where not stated otherwise:
                Column-major (interleaved), vectors are columns.
        For 1D Gna2OperationTypeConvolution  operation:
            Shape: [W] 1D vector where:
             - W is a number of vector elements
             Layout: Row-major (flat)
        For 2D Gna2OperationTypeConvolution  operation:
            Shape: [N x H x W x C] 4D Tensor where:
             - N is a batch size (number of vectors), currently only N=1 is supported
             - H is a height of each filter
             - W is a width of each filter
             - C is a depth of each filter
        For Gna2OperationTypeTransposition (interleave), Gna2OperationTypeRecurrent  operation:
             Layout: Row-major (flat)
    */
    /*struct Gna2Tensor * Inputs;*/

    /**
     Specifies output tensor.
     Valid values:
        //TODO:3:API Redesign: provide shape info
        Common:
            Mode: {Gna2TensorModeDefault, Gna2TensorModeDisabled}
            Type: {Gna2DataTypeInt8, Gna2DataTypeInt16, Gna2DataTypeInt32}
            Shape, where not stated otherwise:
                [NxW] 2D matrix where:
                 - N is a batch size (number of vectors)
                 - W is a number of vector elements
            Layout, where not stated otherwise:
                Column-major (interleaved), vectors are columns.
        For 1D Gna2OperationTypeConvolution  operation:
            Shape: [DxW] 2D matrix, where:
             - D is a number of feature maps
             - W is a number of elements of each feature map
             Layout: Column-major (interleaved)  //TODO:3:API Redesign: provide shape info
        For 2D Gna2OperationTypeConvolution  operation:
            Shape: [N x H x W x C] 4D Tensor where:
             - N is a batch size (number of vectors), currently only N=1 is supported
             - H is a height of each feature maps
             - W is a width of each feature maps
             - C is a number of feature maps
        For Gna2OperationTypeTransposition (deinterleave), Gna2OperationTypeRecurrent  operation:
             Layout: Row-major (flat), vectors are rows.
    */
    //struct Gna2Tensor * Outputs;
    //TODO:3:API Redesign: add debug interface
    //void* pOutputsIntermediate;     // 4B Signed integer Auxiliary output buffer.



 ///**
    // Specifies filters (kernels) tensor.
    // Filters stored one after the other.
    // Note:
    //    For 2D Gna2OperationTypeConvolution  operation each filter must start
    //    at address which is 16B aligned.
    // Valid values:
    //    Common:
    //        Mode: {Gna2TensorModeDefault, GNA2_TENSOR_CONSTANT_SCALAR}
    //        Type: {Gna2DataTypeInt8, Gna2DataTypeInt16},
    //    For 1D Gna2OperationTypeConvolution  operation:
    //        Shape: [N x W] 2D matrix where://TODO:3:API Redesign: provide shape info
    //         - N is a number of filters
    //         - W is a width of each filter
    //    For 2D Gna2OperationTypeConvolution  operation:
    //        Shape: [N x H x W x C] 4D Tensor where://TODO:3:API Redesign: provide shape info
    //         - N is a number of filters
    //         - H is a height of each filter
    //         - W is a width of each filter
    //         - C is a depth of each filter
    //*/
    //struct Gna2Tensor * Filters;
