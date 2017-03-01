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
#include <memory>

#include "common.h"
#include "Validator.h"
#include "GnaConfig.h"

using std::unique_ptr;

namespace GNA
{

typedef enum _Orientations
{
    INTERLEAVED,
    FLAT
} Orientations;

struct LayerConfig
{
public:
    static const std::map<const nn_layer_type, const NN_OP_TYPE> OperationsMap;
    static const std::map<const nn_layer_type, const Orientations> OrientationsMap;

    LayerConfig(const nn_layer_type type);

    const nn_layer_type Type;
    const NN_OP_TYPE Operation;
    const Orientations Orientation;

    LayerConfig()= delete;
    ~LayerConfig() = default;
};

struct LayerMatrix
{
public:
    LayerMatrix(const nn_layer &layer, const Orientations orientation);
    ~LayerMatrix() = default;

    const uint32_t ColumnCount;
    const uint32_t RowCount;
    const uint32_t ElementCount;
    void const * const Buffer;
};

struct LayerInput : public LayerMatrix
{
public:
    LayerInput(const nn_layer &layer, const Orientations orientation, const uint32_t vectorCount);
    ~LayerInput() = default;

    const uint32_t VectorCount;
};

struct LayerOutput : public LayerMatrix
{
public:
    static const uint32_t NonActivatedOutputSize = 4;
    static const uint32_t ActivatedOutputSize = 2;
    
    LayerOutput(const nn_layer &layer, const Orientations orientation);
    ~LayerOutput() = default;

    void Validate(const bool ActivationEnabled, const uint32_t outputSize) const;

    uint32_t const * const BufferIntermediate;
};


class Layer
{
public:
    friend class HwLayer;

    static unique_ptr<Layer> Create(const nn_layer *layer, const uint32_t inputVectorCount);

    virtual ~Layer() {};
    
    const LayerConfig Config;
    const nn_layer sourceLayer;
    const LayerInput Input;
    const LayerOutput Output;

protected:
    Layer(const nn_layer *layer, const uint32_t inputVectorCount);

private:
    static const nn_layer validate(const nn_layer *layer);
};

}
