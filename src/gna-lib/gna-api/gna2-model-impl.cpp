/*
 INTEL CONFIDENTIAL
 Copyright 2019 Intel Corporation.

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

#include "gna2-model-impl.h"
#include "gna2-common-impl.h"

#include "ApiWrapper.h"
#include "Device.h"
#include "DeviceManager.h"
#include "ModelWrapper.h"

#include "gna2-model-api.h"
#include "gna2-common-api.h"

#include <stdint.h>

using namespace GNA;

GNA2_API enum Gna2Status Gna2ModelCreate(uint32_t deviceIndex,
    struct Gna2Model const * model, uint32_t * modelId)
{
    UNREFERENCED_PARAMETER(model);
    UNREFERENCED_PARAMETER(modelId);
    UNREFERENCED_PARAMETER(deviceIndex);

    const std::function<ApiStatus()> command = [&]()
    {
        auto& device = DeviceManager::Get().GetDevice(deviceIndex);
        //device.LoadModel(modelId, model);
        UNREFERENCED_PARAMETER(device);
        return Gna2StatusSuccess;
    };
    return ApiWrapper::ExecuteSafely(command);
}

GNA2_API enum Gna2Status Gna2ModelRelease(uint32_t modelId)
{
    const std::function<ApiStatus()> command = [&]()
    {
        auto& device = DeviceManager::Get().GetDevice(0);
        device.ReleaseModel(modelId);
        return Gna2StatusSuccess;
    };
    return ApiWrapper::ExecuteSafely(command);
}

GNA2_API enum Gna2Status Gna2ModelGetLastError(struct Gna2ModelError * error)
{
    UNREFERENCED_PARAMETER(error);
    // TODO:3:API: implement P2
    const std::function<ApiStatus()> command = [&]()
    {
        return Gna2StatusNotImplemented;
    };
    return ApiWrapper::ExecuteSafely(command);
}

GNA2_API enum Gna2Status Gna2ModelGetLastErrorMessage(char * messageBuffer,
    uint32_t messageBufferSize)
{
    UNREFERENCED_PARAMETER(messageBuffer);
    UNREFERENCED_PARAMETER(messageBufferSize);

    // TODO:3:API: implement P2
    const std::function<ApiStatus()> command = [&]()
    {
        return Gna2StatusNotImplemented;
    };
    return ApiWrapper::ExecuteSafely(command);
}

GNA2_API enum Gna2Status Gna2ModelOperationInit(struct Gna2Operation * operation,
    enum Gna2OperationType type, Gna2UserAllocator userAllocator)
{
    const std::function<ApiStatus()> command = [&]()
    {
        ModelWrapper::OperationInit(operation, type, userAllocator);
        return Gna2StatusSuccess;
    };
    return ApiWrapper::ExecuteSafely(command);
}

GNA2_API uint32_t Gna2DataTypeGetSize(enum Gna2DataType type)
{
    const std::function<uint32_t()> command = [&]()
    {
        return ModelWrapper::DataTypeGetSize(type);
    };
    return ApiWrapper::ExecuteSafely(command, Gna2NotSupportedU32);
}

GNA2_API uint32_t Gna2ShapeGetNumberOfElements(struct Gna2Shape const * shape)
{
    const std::function<uint32_t()> command = [&]()
    {
        return ModelWrapper::ShapeGetNumberOfElements(shape);
    };
    return ApiWrapper::ExecuteSafely(command, Gna2NotSupportedU32);
}

GNA2_API uint32_t Gna2TensorGetSize(struct Gna2Tensor const * tensor)
{
    const std::function<uint32_t()> command = [&]()
    {
         // TODO:3:API: implement P1
        UNREFERENCED_PARAMETER(tensor);
        return Gna2DefaultU32;
    };
    return ApiWrapper::ExecuteSafely(command, Gna2NotSupportedU32);
}

GNA2_API struct Gna2Shape Gna2ShapeInitScalar()
{
    const std::function<ApiShape()> command = []()
    {
        return ModelWrapper::ShapeInit();
    };
    return ApiWrapper::ExecuteSafely(command, Gna2Shape{});
}

GNA2_API struct Gna2Shape Gna2ShapeInit1D(uint32_t x)
{
    const std::function<ApiShape()> command = [&]()
    {
        return ModelWrapper::ShapeInit(GNA_TENSOR_ORDER_ANY, x);
    };
    return ApiWrapper::ExecuteSafely(command, ApiShape{});
}

GNA2_API struct Gna2Shape Gna2ShapeInit2D(uint32_t x, uint32_t y)
{
    // TODO:3:API: implement P2
    const std::function<ApiShape()> command = [&]()
    {
        return ModelWrapper::ShapeInit(GNA_TENSOR_ORDER_ANY, x, y);
    };
    return ApiWrapper::ExecuteSafely(command, ApiShape{});
}

GNA2_API struct Gna2Shape Gna2ShapeInit3D(uint32_t x, uint32_t y, uint32_t z)
{
    const std::function<ApiShape()> command = [&]()
    {
        return ModelWrapper::ShapeInit(GNA_TENSOR_ORDER_ANY, x, y, z);
    };
    return ApiWrapper::ExecuteSafely(command, ApiShape{});
}

GNA2_API struct Gna2Shape Gna2ShapeInit4D(uint32_t n, uint32_t x, uint32_t y,
    uint32_t z)
{
    const std::function<ApiShape()> command = [&]()
    {
        return ModelWrapper::ShapeInit(GNA_TENSOR_ORDER_ANY, n, x, y, z);
    };
    return ApiWrapper::ExecuteSafely(command, ApiShape{});
}

GNA2_API struct Gna2Tensor Gna2TensorInit1D(uint32_t x, enum Gna2DataType type,
    void * data)
{
    UNREFERENCED_PARAMETER(x);
    UNREFERENCED_PARAMETER(type);
    UNREFERENCED_PARAMETER(data);
    // TODO:3:API: implement P2
    const std::function<ApiTensor()> command = [&]()
    {
        return ApiTensor{};
    };
    return ApiWrapper::ExecuteSafely(command, ApiTensor{});
}

GNA2_API struct Gna2Tensor Gna2TensorInit2D(uint32_t x, uint32_t y,
    enum Gna2DataType type, void * data)
{
    UNREFERENCED_PARAMETER(x);
    UNREFERENCED_PARAMETER(y);
    UNREFERENCED_PARAMETER(type);
    UNREFERENCED_PARAMETER(data);
    // TODO:3:API: implement P2
    const std::function<ApiTensor()> command = [&]()
    {
        return ApiTensor{};
    };
    return ApiWrapper::ExecuteSafely(command, ApiTensor{});
}

GNA2_API struct Gna2Tensor Gna2TensorInit3D(uint32_t x, uint32_t y, uint32_t z,
    enum Gna2DataType type, void * data)
{
    UNREFERENCED_PARAMETER(x);
    UNREFERENCED_PARAMETER(y);
    UNREFERENCED_PARAMETER(z);
    UNREFERENCED_PARAMETER(type);
    UNREFERENCED_PARAMETER(data);
    // TODO:3:API: implement P2
    const std::function<ApiTensor()> command = [&]()
    {
        return ApiTensor{};
    };
    return ApiWrapper::ExecuteSafely(command, ApiTensor{});
}

GNA2_API struct Gna2Tensor Gna2TensorInit4D(uint32_t n, uint32_t x, uint32_t y,
    uint32_t z, enum Gna2DataType type, void * data)
{
    UNREFERENCED_PARAMETER(n);
    UNREFERENCED_PARAMETER(x);
    UNREFERENCED_PARAMETER(y);
    UNREFERENCED_PARAMETER(z);
    UNREFERENCED_PARAMETER(type);
    UNREFERENCED_PARAMETER(data);
    // TODO:3:API: implement P2
    const std::function<ApiTensor()> command = [&]()
    {
        return ApiTensor{};
    };
    return ApiWrapper::ExecuteSafely(command, ApiTensor{});
}

GNA2_API struct Gna2Tensor Gna2TensorInitDisabled()
{
    // TODO:3:API: implement P2
    const std::function<ApiTensor()> command = [&]()
    {
        return ApiTensor{};
    };
    return ApiWrapper::ExecuteSafely(command, ApiTensor{});
}

GNA2_API struct Gna2Tensor Gna2TensorInitScalar(enum Gna2DataType type, void * data)
{
    UNREFERENCED_PARAMETER(type);
    UNREFERENCED_PARAMETER(data);
    // TODO:3:API: implement P2
    const std::function<ApiTensor()> command = [&]()
    {
        return ApiTensor{};
    };
    return ApiWrapper::ExecuteSafely(command, ApiTensor{});
}

GNA2_API struct Gna2Tensor Gna2TensorInitActivation(uint32_t numberOfSegments,
    struct Gna2PwlSegment * segments)
{
    UNREFERENCED_PARAMETER(numberOfSegments);
    UNREFERENCED_PARAMETER(segments);
    // TODO:3:API: implement P2
    const std::function<ApiTensor()> command = [&]()
    {
        return ApiTensor{};
    };
    return ApiWrapper::ExecuteSafely(command, ApiTensor{});
}

GNA2_API struct Gna2Operation Gna2OperationInitFullyConnectedAffine(
    struct Gna2Tensor * inputs, struct Gna2Tensor * outputs,
    struct Gna2Tensor * weights, struct Gna2Tensor * biases,
    struct Gna2Tensor * activation)
{
    UNREFERENCED_PARAMETER(inputs);
    UNREFERENCED_PARAMETER(outputs);
    UNREFERENCED_PARAMETER(weights);
    UNREFERENCED_PARAMETER(biases);
    UNREFERENCED_PARAMETER(activation);
    // TODO:3:API: implement P2
    const std::function<ApiOperation()> command = [&]()
    {
        return ApiOperation{};
    };
    return ApiWrapper::ExecuteSafely(command, ApiOperation{});
}

GNA2_API struct Gna2Operation Gna2OperationInitElementWiseAffine(
    struct Gna2Tensor * inputs, struct Gna2Tensor * outputs,
    struct Gna2Tensor * weights, struct Gna2Tensor * biases,
    struct Gna2Tensor * activation)
{
    UNREFERENCED_PARAMETER(inputs);
    UNREFERENCED_PARAMETER(outputs);
    UNREFERENCED_PARAMETER(weights);
    UNREFERENCED_PARAMETER(biases);
    UNREFERENCED_PARAMETER(activation);
    // TODO:3:API: implement P2
    const std::function<ApiOperation()> command = [&]()
    {
        return ApiOperation{};
    };
    return ApiWrapper::ExecuteSafely(command, ApiOperation{});
}

GNA2_API struct Gna2Operation Gna2OperationInitFullyConnectedBiasGrouping(
    struct Gna2Tensor * inputs, struct Gna2Tensor * outputs,
    struct Gna2Tensor * weights, struct Gna2Tensor * biases,
    struct Gna2Tensor * activation,
    struct Gna2Tensor * weightScaleFactors,
    enum Gna2BiasMode* biasMode,
    uint32_t* biasVectorIndex)
{
    UNREFERENCED_PARAMETER(inputs);
    UNREFERENCED_PARAMETER(outputs);
    UNREFERENCED_PARAMETER(weights);
    UNREFERENCED_PARAMETER(biases);
    UNREFERENCED_PARAMETER(activation);
    UNREFERENCED_PARAMETER(weightScaleFactors);
    UNREFERENCED_PARAMETER(biasMode);
    UNREFERENCED_PARAMETER(biasVectorIndex);
    // TODO:3:API: implement P2
    const std::function<ApiOperation()> command = [&]()
    {
        return ApiOperation{};
    };
    return ApiWrapper::ExecuteSafely(command, ApiOperation{});
}

GNA2_API struct Gna2Operation Gna2OperationInitRecurrent(
    struct Gna2Tensor * inputs, struct Gna2Tensor * outputs,
    struct Gna2Tensor * weights, struct Gna2Tensor * biases,
    struct Gna2Tensor * activation,
    uint32_t* delay)
{
    UNREFERENCED_PARAMETER(inputs);
    UNREFERENCED_PARAMETER(outputs);
    UNREFERENCED_PARAMETER(weights);
    UNREFERENCED_PARAMETER(biases);
    UNREFERENCED_PARAMETER(activation);
    UNREFERENCED_PARAMETER(delay);
    // TODO:3:API: implement P2
    const std::function<ApiOperation()> command = [&]()
    {
        return ApiOperation{};
    };
    return ApiWrapper::ExecuteSafely(command, ApiOperation{});
}

GNA2_API struct Gna2Operation Gna2OperationInitConvolution(
    struct Gna2Tensor * inputs, struct Gna2Tensor * outputs,
    struct Gna2Tensor * filters, struct Gna2Tensor * biases,
    struct Gna2Tensor * activation,
    struct Gna2Shape * zeroPadding,
    struct Gna2Shape * convolutionStride,
    enum Gna2BiasMode * biasMode)
{
    UNREFERENCED_PARAMETER(inputs);
    UNREFERENCED_PARAMETER(outputs);
    UNREFERENCED_PARAMETER(filters);
    UNREFERENCED_PARAMETER(biases);
    UNREFERENCED_PARAMETER(activation);
    UNREFERENCED_PARAMETER(zeroPadding);
    UNREFERENCED_PARAMETER(convolutionStride);
    UNREFERENCED_PARAMETER(biasMode);
    // TODO:3:API: implement P2
    const std::function<ApiOperation()> command = [&]()
    {
        return ApiOperation{};
    };
    return ApiWrapper::ExecuteSafely(command, ApiOperation{});
}

GNA2_API struct Gna2Operation Gna2OperationInitConvolutionFused(
    struct Gna2Tensor * inputs, struct Gna2Tensor * outputs,
    struct Gna2Tensor * filters, struct Gna2Tensor * biases,
    struct Gna2Tensor * activation,
    struct Gna2Shape * zeroPadding,
    struct Gna2Shape * convolutionStride,
    enum Gna2BiasMode * biasMode,
    enum Gna2PoolingMode * poolingMode,
    struct Gna2Shape * poolingWindow,
    struct Gna2Shape * poolingStride)
{
    UNREFERENCED_PARAMETER(inputs);
    UNREFERENCED_PARAMETER(outputs);
    UNREFERENCED_PARAMETER(filters);
    UNREFERENCED_PARAMETER(biases);
    UNREFERENCED_PARAMETER(activation);
    UNREFERENCED_PARAMETER(zeroPadding);
    UNREFERENCED_PARAMETER(convolutionStride);
    UNREFERENCED_PARAMETER(biasMode);
    UNREFERENCED_PARAMETER(poolingMode);
    UNREFERENCED_PARAMETER(poolingWindow);
    UNREFERENCED_PARAMETER(poolingStride);
    // TODO:3:API: implement P2
    const std::function<ApiOperation()> command = [&]()
    {
        return ApiOperation{};
    };
    return ApiWrapper::ExecuteSafely(command, ApiOperation{});
}

GNA2_API struct Gna2Operation Gna2OperationInitPooling(
    struct Gna2Tensor * inputs, struct Gna2Tensor * outputs,
    struct Gna2Tensor * activation,
    struct Gna2Shape * zeroPadding,
    enum Gna2PoolingMode * poolingMode,
    struct Gna2Shape * poolingWindow,
    struct Gna2Shape * poolingStride)
{
     UNREFERENCED_PARAMETER(inputs);
    UNREFERENCED_PARAMETER(outputs);
    UNREFERENCED_PARAMETER(activation);
    UNREFERENCED_PARAMETER(zeroPadding);
    UNREFERENCED_PARAMETER(poolingMode);
    UNREFERENCED_PARAMETER(poolingWindow);
    UNREFERENCED_PARAMETER(poolingStride);
    // TODO:3:API: implement P2
    const std::function<ApiOperation()> command = [&]()
    {
        return ApiOperation{};
    };
    return ApiWrapper::ExecuteSafely(command, ApiOperation{});
}

GNA2_API struct Gna2Operation Gna2OperationInitCopy(
    struct Gna2Tensor * inputs, struct Gna2Tensor * outputs,
    struct Gna2Shape * copyParams)
{
    UNREFERENCED_PARAMETER(inputs);
    UNREFERENCED_PARAMETER(outputs);
    UNREFERENCED_PARAMETER(copyParams);
    // TODO:3:API: implement P2
    const std::function<ApiOperation()> command = [&]()
    {
        return ApiOperation{};
    };
    return ApiWrapper::ExecuteSafely(command, ApiOperation{});
}

GNA2_API struct Gna2Operation Gna2OperationInitTranspose(
    struct Gna2Tensor * inputs, struct Gna2Tensor * outputs)
{
    UNREFERENCED_PARAMETER(inputs);
    UNREFERENCED_PARAMETER(outputs);
    // TODO:3:API: implement P2
    const std::function<ApiOperation()> command = [&]()
    {
        return ApiOperation{};
    };
    return ApiWrapper::ExecuteSafely(command, ApiOperation{});
}

//TODO:3:API define
GNA2_API struct Gna2Operation Gna2OperationInitGmm(
    struct Gna2Tensor * inputs, struct Gna2Tensor * outputs,
    struct Gna2Tensor * means,
    struct Gna2Tensor * inverseCovariances,
    struct Gna2Tensor * consts,
    uint32_t * maximumScore)
{
    UNREFERENCED_PARAMETER(inputs);
    UNREFERENCED_PARAMETER(outputs);
    UNREFERENCED_PARAMETER(means);
    UNREFERENCED_PARAMETER(inverseCovariances);
    UNREFERENCED_PARAMETER(consts);
    UNREFERENCED_PARAMETER(maximumScore);
    // TODO:3:API: implement P2
    const std::function<ApiOperation()> command = [&]()
    {
        return ApiOperation{};
    };
    return ApiWrapper::ExecuteSafely(command, ApiOperation{});
}

GNA2_API struct Gna2Operation Gna2OperationInitGmmInterleaved(
    struct Gna2Tensor * inputs, struct Gna2Tensor * outputs,
    struct Gna2Tensor * interleavedTensors,
    uint32_t * maximumScore)
{
     UNREFERENCED_PARAMETER(inputs);
    UNREFERENCED_PARAMETER(outputs);
    UNREFERENCED_PARAMETER(interleavedTensors);
    UNREFERENCED_PARAMETER(maximumScore);
    // TODO:3:API: implement P2
    const std::function<ApiOperation()> command = [&]()
    {
        return ApiOperation{};
    };
    return ApiWrapper::ExecuteSafely(command, ApiOperation{});
}
