/*
 INTEL CONFIDENTIAL
 Copyright 2017-2020 Intel Corporation.

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

#include "gna2-api.h"

constexpr uint32_t InputOperandIndex = 0;
constexpr uint32_t OutputOperandIndex = 1;
constexpr uint32_t WeightOperandIndex = 2;
constexpr uint32_t FilterOperandIndex = 2;
constexpr uint32_t BiasOperandIndex = 3;
constexpr uint32_t PwlOperandIndex = 4;
constexpr uint32_t WeightScaleFactorOperandIndex = 5;
constexpr uint32_t GmmInterleavedOperandIndex = 2;
constexpr uint32_t GmmMeanOperandIndex = 2;
constexpr uint32_t GmmInverseCovarianceOperandIndex = 3;
constexpr uint32_t GmmGaussianConstantOperandIndex = 4;

constexpr uint32_t ConvolutionStrideParamIndex = 0;
constexpr uint32_t BiasModeConvolutionParamIndex = 1;
constexpr uint32_t PoolingModeParamIndex = 2;
constexpr uint32_t PoolingWindowParamIndex = 3;
constexpr uint32_t PoolingStrideParamIndex = 4;
constexpr uint32_t ZeroPaddingParamIndex = 5;

constexpr uint32_t BiasModeAffineParamIndex = 0;
constexpr uint32_t BiasVectorParamIndex = 1;

class DeviceController
{
public:
    DeviceController();
    ~DeviceController();

    uint8_t * Alloc(uint32_t sizeRequested, uint32_t * sizeGranted);

    void Free(void *memory);

    void ModelCreate(const Gna2Model *model, uint32_t *modelId) const;
    void ModelRelease(uint32_t modelId) const;

    static uint32_t ConfigAdd(uint32_t modelId);
    static void BufferAdd(uint32_t configId, uint32_t operationIndex, uint32_t operandIndex, void * address);
    static void RequestSetAcceleration(uint32_t, Gna2AccelerationMode);
    static void RequestSetConsistency(uint32_t, Gna2DeviceVersion);
    static void BufferAddIO(uint32_t configId,  uint32_t outputOperationIndex, void * input, void * output);


    static void RequestEnqueue(uint32_t, uint32_t *);
    static void RequestWait(uint32_t);

    static void ActiveListAdd(uint32_t configId, uint32_t layerIndex, uint32_t indicesCount, uint32_t* indices);

#if HW_VERBOSE == 1
    void AfterscoreDebug(uint32_t modelId, uint32_t nActions, dbg_action *actions);

    void PrescoreDebug(uint32_t modelId, uint32_t nActions, dbg_action *actions);
#endif

private:
    uint32_t gnaHandle;

    void *gnaMemory = nullptr;

    static void ThrowOnStatusUnsuccessful(Gna2Status status, char const* message);
};
