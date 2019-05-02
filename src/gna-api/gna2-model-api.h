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
            - padding is optional, not supported for 1D convolution

    Operands:
        1. inputs:
            Specifies input tensor.
            Supported values:
                - Mode: {::Gna2TensorModeDefault}
                - Type: {::Gna2DataTypeInt8, ::Gna2DataTypeInt16},
                - Shape: [N x W] for 1D Convolution and [N x H x W x C] for 2D Convolution, where:
                    - N is a batch size (number of vectors), currently only N=1 is supported
                    - H is a height of input tensor
                    - W is a width of input tensor
                    - C is a depth of input tensor
        2. outputs:
            Specifies output tensor, as the final output of all composed functions.
            Supported values:
                - Mode: {::Gna2TensorModeDefault}
                - Type: {::Gna2DataTypeInt8, ::Gna2DataTypeInt16, ::Gna2DataTypeInt32},
                    @note When activationFunction is disabled Type is always ::Gna2DataTypeInt32.
                - Shape: [N x W x C] for 1D Convolution and [N x H x W x C] for 2D Convolution, where:
                    - N is a batch size (number of vectors), currently only N=1 is supported
                    - H is a height of result tensor
                    - W is a width of result tensor
                    - C is a depth of result tensor, same as the number of filters (filter N dimension)
        3. filters:
            Specifies filters (kernels) tensor. Filters are stored one after the other.
            @note: For 2D ::Gna2OperationTypeConvolution operation each filter must start
                   at address which is 16B aligned.
            Supported values:
                - Mode: {::Gna2TensorModeDefault, ::Gna2TensorModeConstantScalar}
                - Type: {::Gna2DataTypeInt8, ::Gna2DataTypeInt16},
                - Shape: [N x W] for 1D Convolution and [N x H x W x C] for 2D Convolution, where:
                    - N is a number of filters
                    - H is a height of each filter
                    - W is a width of each filter
                    - C is a depth of each filter
        4. biases:
            Supported values:
                - Mode: {::Gna2TensorModeDefault, ::Gna2TensorModeDisabled}
                - Type: {::Gna2DataTypeInt8, ::Gna2DataTypeInt16, ::Gna2DataTypeInt32},
                - Shape: (@see biasMode parameter), can be set 0, as is calculated by GNA,
                    - For biasMode ::Gna2BiasModeDefault: [N] 1D Vector, where
                        - N is a number of filters,
                    - For biasMode ::Gna2BiasModePerStride: [N x H x W] 1D Vector, where
                        - N is a number of filters,
                        - H is a number of the convolution output rows, 
                        - W is a number of the convolution output columns,
        5. activationFunction [optional]:
            Specifies PWL activation function segment tensor.
            - Segments have to be contiguous.
            - Set NULL to disable activation.
            Supported values:
                - Shape: [W] 1D vector, where:
                    W is a number of piecewise-linear segments
                - Mode: {::Gna2TensorModeDefault}
                - Type: {::Gna2DataTypePwlSegment},

    Parameters:
        1. Gna2Shape convolutionStride:
             Specifies filter stride shape.
             Supported values:
                For 1D convolution operation:
                    [W] 1D where: //TODO:3:API Redesign: provide shape info
                     - W is a number of elements to move in W dimension
                For 2D convolution operation:
                    [W x H] 2D where: //TODO:3:API Redesign: provide shape info
                     - W is a number of elements to move in W dimension
                     - H is a number of elements to move in H dimension
        2. Gna2BiasMode biasMode - Mode of bias operation:
            Supported values: {::Gna2BiasModeDefault, ::Gna2BiasModePerStride}
        3. Gna2PoolingMode poolingType of bias operation [optional]
        4. Gna2Shape poolingWindow [optional]:
            Specifies pooling window shape.
            Supported values:
            - For 1D convolution operation: [ W ] 1D where: //TODO:3:API Redesign: provide shape info
                - W is a width of window
            - For 2D convolution operation:
                - [ W x H ] 2D where:
                    - W is a width of window
                    - H is a height of window
        5. Gna2Shape poolingStride [optional]:
            Specifies pooling window stride dimensions.
            Supported values:
                - For 1D convolution operation: [W] 1D where: //TODO:3:API Redesign: provide shape info
                    - W is a number of elements to move in W dimension
                - For 2D convolution operation: [W x H] 2D where:
                    - W is a number of elements to move in W dimension
                    - H is a number of elements to move in H dimension
        6. Gna2Shape zeroPadding [optional]:
            Supported only for 2D convolution.
            Specifies automatic input zero-padding dimensions.
            Used to maintain same input-output volume shape
            or when input dimensions have no common natural divider with filter and stride.
            Supported values:
                [0] When not used
                [W x H] 2D where: //TODO:3:API Redesign: provide shape info
                    - W is a number of 0s added each before and after each row of an input
                    - H is a number of 0s added each before and after each column of an input
    */
    Gna2OperationTypeConvolution = 1,

    /**
    Copy operation.

    Operation:
        output = copy(input, shape)
        // TODO:3:provide detailed formula

    Operands:
        1. inputs:
            Specifies input tensor.
            Supported values:
                - Mode: {::Gna2TensorModeDefault}
                - Type: {::Gna2DataTypeInt8, ::Gna2DataTypeInt16},
                - Layout: Default: [N x W] Row-major, (aka flat), 
                - Shape: [N x W] 2D matrix, where:
                    - N is a batch size (number of vectors),
                    - W is a width of input tensor
        2. outputs:
            Specifies output tensor.
            @see inputs, with exclusion to:
                - Shape: [N x W] 2D matrix, where:
                    - N is a destination batch size (number of vectors),
                    - W is a width of a destination tensor,

    Parameters:
        1. Gna2Shape shape [required]:
             Specifies dimensions of copied sub-tensor.
             Supported values:
                [W x H] 2D where: //TODO:3:API Redesign: provide shape info
                 - W is a number of elements to copy in W dimension
                 - H is a number of elements to copy in H dimension
    */
    Gna2OperationTypeCopy = 2,

    /**
    Fully connected affine operation composed with activation function.

    Operation:
        - a) outputs = activation(((inputs x weights) + biases), activationFunction)
        - b) outputs = activation(((inputs x (weightScaleFactors * weights)) + biases[biasVectorIndex]), activationFunction)
        .
        Where:
            - activation is optional
            - weightScaleFactors is optional, required only for ::Gna2BiasModeGrouping Mode.

    Operands:
        1. inputs:
            Specifies input tensor.
            Supported values:
                - Mode: {::Gna2TensorModeDefault}
                - Type: {::Gna2DataTypeInt8, ::Gna2DataTypeInt16},
                - Layout:
                    @note [W x N] Row-major (aka interleaved), vectors are columns.
                - Shape: [W x N] 2D matrix, where:
                    - W is a width of input tensor
                    - N is a batch size (number of vectors),
        2. outputs:
            Specifies output tensor.
            Supported values:
                - Mode: {::Gna2TensorModeDefault}
                - Type: {::Gna2DataTypeInt8, ::Gna2DataTypeInt16, ::Gna2DataTypeInt32},
                    @note When activationFunction is disabled Type is always ::Gna2DataTypeInt32.
                - Layout: same as inputs
                - Shape:  same as inputs
        3. weights:
            Specifies weight tensor.
            Supported values:
                - Mode: {::Gna2TensorModeDefault, ::Gna2TensorModeConstantScalar}
                - Type: {::Gna2DataTypeInt8, ::Gna2DataTypeInt16},
                - Shape: [W x H] 2D Matrix, where:
                    - W is a number of input tensor elements (input Shape W dimension)
                    - H is a number of output tensor elements (output Shape H dimension)
        4. biases:
            Supported values:
                - For ::Gna2BiasModeDefault:
                    - Mode: {::Gna2TensorModeDefault, ::Gna2TensorModeDisabled}
                    - Type: {::Gna2DataTypeInt8, ::Gna2DataTypeInt16, ::Gna2DataTypeInt32, ::Gna2DataTypeCompoundBias},
                    - Shape: can be set 0, as is calculated by GNA,
                        - [H] 1D Vector, where
                            - H is a number of the output nodes (rows),
                - For ::Gna2BiasModeGrouping:
                    - Mode: {::Gna2TensorModeDefault, ::Gna2TensorModeDisabled // TODO:3:verify is applicable}
                    - Type: {::Gna2DataTypeInt32 // TODO:3:verify is applicable},
                    - Shape: [N x H] 2D Matrix where:
                        - N is a number of the bias vectors (columns), @see biasVectorIndex,
                        - H is a number of the output nodes (rows),
        5. activationFunction:
            - @see ::Gna2OperationTypeConvolution activationFunction operand
        6. weightScaleFactors:
            Specifies separate scale factors for weights.
            Required only for weights type of ::Gna2DataTypeInt8 and ::Gna2BiasModeGrouping.
            Set NULL for other cases.
            //TODO:3:API Redesign: provide formula used
            Supported values:
                - Shape: [H] 1D Vector, where:
                    - H is a number of the output nodes (rows),
                - Mode: {::Gna2TensorModeDefault}
                - Type: {::Gna2DataTypeWeightScaleFactor},

    Parameters:
        1. Gna2BiasMode Mode of bias operation [optional]:
            Supported values:
                -::Gna2BiasModeDefault: normal operation
                -::Gna2BiasModeGrouping: Special optimized case // TODO:3:API: elaborate
                    Requires weightScaleFactors, Bias vector index
        2. uint32_t biasVectorIndex, [optional, used only with ::Gna2BiasModeGrouping]:
            Index of the bias vector used for this operation.
            Supported values:
                - [0, N-1]: Where N is a number of all bias vectors in biases tensor.
                    Default is 0.
                - GNA2_DEFAULT: is equivalent of 0.
    */
    Gna2OperationTypeFullyConnectedAffine = 3,

    /**
    Element wise affine operation composed with activation function and pooling.
    Used e.g. for scaling input tensor.

    Operation:
        - output = activation((times(input, weights) + biases), activationFunction)
        .
        Where:
            - Activation is optional
            - Weights is diagonal matrix, represented by 1D vector.

    Operands:
        1. inputs
           @see ::Gna2OperationTypeFullyConnectedAffine input operand.
        2. outputs:
           @see ::Gna2OperationTypeFullyConnectedAffine output operand.
        3. weights:
            @see ::Gna2OperationTypeFullyConnectedAffine weights operand, with exclusion to:
            - Shape: [H] 1D vector, where:
                - H is a number of output tensor elements (output Shape H dimension)
        4. biases:
            @see ::Gna2OperationTypeFullyConnectedAffine biases for ::Gna2BiasModeDefault
        5. activationFunction:
            - @see ::Gna2OperationTypeConvolution activationFunction operand

    Parameters:
        None
    */
    Gna2OperationTypeElementWiseAffine = 4,

    /**
    Gaussian Mixture Model scoring operation.

    Operation:
        - a) output = GMM(input, means, inverseCovariances, constants)
        - b) output = GMM(input, interleaved{means, inverseCovariances, constants})
    
    Shape dimensions meaning:
        - N is a batch size (number of feature vectors),
        - H is a number of GMM states,
        - C is a number of mixtures,
        - W is a number of feature elements in single vector,

    Operands:
    a) "flat" layout:
        1. inputs:
           @see ::Gna2OperationTypeCopy input operand.
           - W shape dimension represents number of features here.
        2. outputs:
            @see ::Gna2OperationTypeCopy output operand.
           - W shape dimension represents number of GMM states here.
        3. means:
            Specifies mean data tensor.
            Supported values:
                - Mode: {::Gna2TensorModeDefault}
                - Type: {::Gna2DataTypeUint8},
                - Shape and Layout: [H x C x W] 3D tensor, with dimensions as described above
        4. inverseCovariances:
            Specifies inverse covariances data tensor.
            Supported values:
                - Mode: {::Gna2TensorModeDefault}
                - Type: {::Gna2DataTypeUint8, ::Gna2DataTypeUint16},
                - Shape and Layout: [H x C x W] 3D tensor, with dimensions as described above
        5. constants:
            Specifies gaussian constants data tensor.
            Supported values:
                - Mode: {::Gna2TensorModeDefault}
                - Type: {::Gna2DataTypeUint32},
                - Shape and Layout: [H x C] 2D matrix, with dimensions as described above
    b) "interleaved" layout:
        1. inputs:
            Same as in Operands a.
        2. outputs:
            Same as in Operands a.
        3. interleaved{means, inverseCovariances, constants}:
            Specifies data tensor with interleaved mean, inverse covariances and gaussian constants data
            for optimized memory bandwidth usage and performance.
            Supported values:
                - Mode: Same as in "flat" layout.
                - Type: {::Gna2DataTypeVoid},
                - Shape and Layout: [H x C x W x C x W x C] 6D tensor, with dimensions as described above,

    Parameters:
        1. uint32_t maximumScore [required]:
            Maximum Score value above which scores are saturated.
    */
    Gna2OperationTypeGmm = 5,

    /**
    Fully connected affine operation with recurrence composed with activation function.

     Operation:
        - output = activation((((input[t], output[t-delay]) x weights) + bias), activationFunction)
        .
        Where:
            - output[t-delay] - recurrent input (feedback) from t-delay output vector of current request.

     Operands:
        1. inputs:
            @see ::Gna2OperationTypeCopy input operand.
        2. outputs:
            @see ::Gna2OperationTypeCopy output operand, with notice:
            @note 1. Output type has to match input type, implicitly activationFunction has to be enabled.
            @note 2. Output data layout has to match overlap with implicit feedback defined by the delay parameter.
            // TODO:3:API: provide I/O data layout requirements
        3. weights:
            @see ::Gna2OperationTypeFullyConnectedAffine weights operand.
        4. biases:
            @see ::Gna2OperationTypeFullyConnectedAffine biases for ::Gna2BiasModeDefault
        5. activationFunction:
            - @see ::Gna2OperationTypeConvolution activationFunction operand,

     Parameters:
        1. uint32_t delay:
            Delay in term of number of vectors in request.
            Supported values:
                - [1, N-1]: Where N is a number of input vectors in current request.
    */
    Gna2OperationTypeRecurrent = 6,

    /**
    Tensor transposition operation.

    Operation:
        - output<layoutOut> = transpose(input<layoutIn>)
        .
        Where:
            - layout[In/Out] specifies transposition direction. 

    Operands:
        1. inputs:
            @see ::Gna2OperationTypeCopy input operand, with notice:
            @note: Layout:
                - [N x W] aka "interleave operation"
                - [W x N] aka "deinterleave operation"
        2. outputs:
            @see ::Gna2OperationTypeCopy input operand, with notice:
            @note: Layout:
                - [W x N] aka "interleave operation"
                - [N x W] aka "deinterleave operation"

    Parameters: none.
    */
    Gna2OperationTypeTransposition = 7,

    /**
    Control-flow operation with threshold parameter.
    */
    Gna2OperationTypeThreshold = 8,
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
*/
struct Gna2Tensor
{
    /**
    Specifies tensor dimensions.

    Default parameters, when not stated otherwise:
        [N x W] 2D matrix (if not stated otherwise) where:
            - N is a batch size (number of vectors)
            - W is a number of vector elements
    */
    struct Gna2Shape Shape;

    /**
    Mode of tensor interpretation.
     
    Default value, when not stated otherwise:
        {::Gna2TensorModeDefault}
    */
    enum Gna2TensorMode Mode;

    /**
     Data layout or format in memory [optional].

    - Specifies order of dimensions, i.e. how Gna2Shape::Dimensions are interpreted.
      Size of layout array must be the same as #Shape Gna2Shape::NumberOfDimensions.
    - Required for Gna2Operation::Type = ::Gna2OperationTypeTransposition and ::Gna2OperationTypeGmm.
    - Optional for other cases, can be left zeroed, layout as in Shape description is assumed,
    - E.g.:
        - [N x W]  N is a number of vectors in a batch (rows) and W is a number of vector
            elements (columns), where data is stored row-major. Whole vectors
            are stored one after another in memory. Aka flat layout.
            - For example let N=2, W=8:
                v\\e |  e0 |   e1 |   e2 |   e3 |   e4 |   e5 |   e6 |   e7
                ---- |---- | ---- | ---- | ---- | ---- | ---- | ---- | ----
                 v0: |v0e0 | v0e1 | v0e2 | v0e3 | v0e4 | v0e5 | v0e6 | v0e7
                 v1: |v1e0 | v1e1 | v1e2 | v1e3 | v1e4 | v1e5 | v1e6 | v1e7    
        - [W x N] W is a number of vector elements (rows) and N is a number of vectors
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
        
        - [N x H x W x C] is Number of tensors in a batch, Height, Width and number
            of Channels of tensor, where the rightmost dimension changes fastest.
    */
    char Layout[GNA2_SHAPE_MAXIMUM_NUMBER_OF_DIMENSIONS];

    /**
     Type of tensor data.
     */
    enum Gna2DataType Type;

    /**
     Data buffer.
     Must be specified before queueing request, during model or request config creation.
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
        + 0 � shift right by 8 bits
        + 1 � shift right by 16 bits
        + 2 � shift right by 24 bits
        + 3 � shift right by 32 bits
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

 Can be called after Gna2ModelCreate() have failed (e.g., returned ::Gna2StatusModelConfigurationInvalid status).

 @note
 The function should be called directly after Gna2ModelCreate() in the same thread.

 @param [out] error The detailed description of model issue.
 @return Status of fetching the model error.
    @retval ::Gna2StatusSuccess The error has been fetched successfully.
    @retval ::Gna2StatusUnknownError No issue to report.
    @retval ::Gna2StatusNullArgumentNotAllowed The error pointer was NULL.
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
    @retval ::Gna2StatusResourceAllocationError The messageBuffer is too small. The message was truncated.
    @retval ::Gna2StatusNullArgumentNotAllowed The messageBuffer was NULL or messageBufferSize was 0.
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
    Gna2ErrorTypeNotGtZero = -8,

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

 @addtogroup GNA2_MODEL_UTILITY_API Model Utilities

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
 @retval GNA2_NOT_SUPPORTED If type is invalid.
 */
GNA2_API uint32_t Gna2DataTypeGetSize(enum Gna2DataType type);

/**
 Gets the total number of elements of the given shape.

 Useful for calculating the sizes of memory buffers.

 @param shape The shape to calculate the number of elements.
 @return Total number of elements.
 @retval GNA2_NOT_SUPPORTED If shape is NULL or malformed.
 @retval 0 If shape has no dimensions.
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
    struct Gna2Shape * convolutionStride,
    enum Gna2BiasMode * biasMode);

GNA2_API struct Gna2Operation Gna2OperationInitConvolutionFused(
    struct Gna2Tensor * inputs, struct Gna2Tensor * outputs,
    struct Gna2Tensor * filters, struct Gna2Tensor * biases,
    struct Gna2Tensor * activation,
    struct Gna2Shape * zeroPadding,
    struct Gna2Shape * convolutionStride,
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

GNA2_API struct Gna2Operation Gna2OperationInitInterleave(
    struct Gna2Tensor * inputs, struct Gna2Tensor * outputs);

GNA2_API struct Gna2Operation Gna2OperationInitDeInterleave(
    struct Gna2Tensor * inputs, struct Gna2Tensor * outputs);

//TODO:3:API define
GNA2_API struct Gna2Operation Gna2OperationInitGmm(
    struct Gna2Tensor * inputs, struct Gna2Tensor * outputs,
    struct Gna2Tensor * means,
    struct Gna2Tensor * inverseCovariances,
    struct Gna2Tensor * constants,
    uint32_t * maximumScore);

GNA2_API struct Gna2Operation Gna2OperationInitGmmInterleaved(
    struct Gna2Tensor * inputs, struct Gna2Tensor * outputs,
    struct Gna2Tensor * interleavedTensors,
    uint32_t * maximumScore);

#endif // __GNA2_MODEL_API_H

/**
 @}
 @}
 @}
 */

    //TODO:3:API Redesign: add debug interface
    //void* pOutputsIntermediate;     // 4B Signed integer Auxiliary output buffer.
