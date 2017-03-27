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
#include "GnaConfig.h"

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
    static const std::map<const nn_layer_kind, const Orientations> OrientationsMap;

    LayerConfig(const nn_layer_kind kind, const nn_layer_type type);

    const nn_layer_kind Kind;
    const nn_layer_type Type;
    const Orientations Orientation;

    LayerConfig()= delete;
    ~LayerConfig() = default;
};

struct LayerMatrix
{
public:
    LayerMatrix(const uint32_t rowCount, const uint32_t columnCount, void const * buffer, const LayerConfig& config);
    ~LayerMatrix() = default;

    const uint32_t ColumnCount;
    const uint32_t RowCount;
    const uint32_t ElementCount;
    void const * const Buffer;
};

struct LayerInput : public LayerMatrix
{
public:
    LayerInput(const nn_layer &layer, const LayerConfig& config, const uint32_t vectorCount);
    ~LayerInput() = default;

    const uint32_t VectorCount;
};

struct LayerOutput : public LayerMatrix
{
public:
    typedef enum _OutputMode
    {
        NonActivatedOutput,
        ActivatedOutput,
    } OutputMode;

    static const uint32_t NonActivatedOutputSize = 4;
    static const uint32_t ActivatedOutputSize = 2;

    LayerOutput(const nn_layer &layer, const LayerConfig& config);
    ~LayerOutput() = default;

    void SetOutputMode(const bool activationEnabled, const uint32_t outputSize);
    const OutputMode GetOutputMode() const
    {
        return mode;
    };

    uint32_t const * const ScratchPad;

private:
    OutputMode mode;
};


class Layer
{
public:
    static std::unique_ptr<Layer> Create(const nn_layer *layer, const uint32_t inputVectorCount);

    virtual ~Layer() {};

    const nn_layer sourceLayer;// TODO: move to private when integration completed
    const LayerConfig Config;
    const LayerInput Input;
    LayerOutput Output;

protected:
    Layer(const nn_layer *layer, const uint32_t inputVectorCount);

private:
    static const nn_layer getSafeCopy(const nn_layer *layer);
};

}
