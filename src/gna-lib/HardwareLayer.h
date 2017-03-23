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

#include <map>

#include "ActiveList.h"
#include "AffineLayers.h"
#include "ConvolutionalLayer.h"
#include "GmmLayer.h"
#include "Layer.h"
#include "Memory.h"
#include "RecurrentLayer.h"
#include "SimpleLayers.h"
#include "SwHw.h"

using std::map;
using std::unique_ptr;

namespace GNA
{

// Hardware Layer descriptor converter
class HardwareLayer
{
public:
    static XNN_LYR Convert(const Layer& softwareLayer, void * const memoryBase, 
        const uint32_t hardwareInternalBufferSize);

    virtual ~HardwareLayer() = default;

    /**
    * Converts API layer active list to hardware layer and stores to hwLyr
    *
    * @NOTE:   Convert must be called earlier
    * @al          (in)    active list parameters
    */
    void convertAL(ActiveList &activeList);

protected:
    HardwareLayer(const Layer& swLayer, void * const memoryBase);

    void save();

    inline const uint32_t getOffset(const void* address) const
    {
        return Hw::getAddrOffset(address, memoryBaseAddress);
    }

    static XNN_LYR layerDescriptor; // single layer descriptor
    const Layer& softwareLayer;

private:
    static const map<const nn_layer_kind, const NN_OP_TYPE> OperationsMap;

    void * const memoryBaseAddress;
};

// Extended Hardware Layer descriptor converter
class HardwareLayerExt : public HardwareLayer
{
public:
    ~HardwareLayerExt() = default;
    HardwareLayerExt(const HardwareLayerExt &) = delete;
    HardwareLayerExt& operator=(const HardwareLayerExt&) = delete;

protected:
    HardwareLayerExt(const Layer& swLayer, void * const memoryBase, const uint32_t bufferSize,
        const uint32_t effectiveGrouping);

    void save();

    const uint32_t bufferElementCount;
    uint32_t lastIterationElementCount;
    const AffineFunction* affine = nullptr;
    const ActivationFunction* activation = nullptr;

private:
    // Number of data elements that may be stored in hw buffer
    const static map<const uint32_t, std::array<const uint32_t, XNN_N_GROUP_MAX>> bufferElementsMap;

    const uint32_t iterationGrouping; // grouping for iteration calculation
    uint32_t iterationCount; // number of iterations = data chunks/parts
};

// Affine, Diagonal and transpose layers Layer descriptor converter
class HardwareLayerAffDiagTrans : public HardwareLayerExt
{
public:
    HardwareLayerAffDiagTrans(const Layer& swLayer, void * const memoryBase, const uint32_t hwInBuffSize);

    virtual ~HardwareLayerAffDiagTrans() = default;
};

// Hardware Copy Layer descriptor converter
class HardwareLayerCopy : public HardwareLayer
{
public:
    HardwareLayerCopy(const Layer& swLayer, void * const memoryBase);
    HardwareLayerCopy(const HardwareLayerCopy &) = delete;
    HardwareLayerCopy& operator=(const HardwareLayerCopy&) = delete;
    virtual ~HardwareLayerCopy() = default;

protected:
    void save();
};

// Recurrent Layer descriptor converter
class HardwareLayerRnn : public HardwareLayerExt
{
public:
    HardwareLayerRnn(const Layer& swLayer, void * const memoryBase, const uint32_t hwInBuffSize);
    HardwareLayerRnn(const HardwareLayerRnn &) = delete;
    HardwareLayerRnn& operator=(const HardwareLayerRnn&) = delete;
    virtual ~HardwareLayerRnn() = default;

    // calculates feedback buffer offset for per RequestConfiguration output buffer
    const uint32_t CalculateFeedbackBuffer(const void * const outputBuffer) const;

protected:
    void convert();
    void save();

private:
    uint32_t feedbackIterationsCount;
    uint32_t feedbackFirstIterElementCount; // number of el. in first feedback data iter.
    uint32_t feedbackLastIterElementCount; // number of el. in last feedback data iter.
};

// Convolutional Layer descriptor converter
class HardwareLayerCnn : public HardwareLayerExt
{
public:
    HardwareLayerCnn(const Layer& swLayer, void * const memoryBase, const uint32_t hwInBuffSize);
    HardwareLayerCnn(const HardwareLayerRnn &) = delete;
    HardwareLayerCnn& operator=(const HardwareLayerRnn&) = delete;
    virtual ~HardwareLayerCnn() = default;

protected:
    void save();

private:
    static const uint32_t CNN_N_FLT_ITER_MAX = 16; // CNN maximum number of filters per iteration

    uint32_t filtersIterationCount;                // Number of iterations  to process all filters.
    uint32_t filtersCountInLastIteration;          // Number of filters in last iteration.
    uint32_t filtersCountInFullIteration;          // Number of filters in buffer in full iterations.
    uint32_t filtersElementCountInFullIteration;   // Size of filter in non-last iterations (elements).
    uint32_t filtersElementCountInLastIteration;   // Size of filter in last iterations (elements).
};

}
