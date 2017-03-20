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

#include "AffineLayers.h"
#include "Validator.h"

using std::make_unique;

using namespace GNA;

AffineLayer::AffineLayer(const nn_layer *layer, const uint32_t inputVectorCount) :
    Layer(layer, inputVectorCount),
    Activation(&static_cast<const nn_layer_affine*>(layer->pLayerStruct)->pwl),
    sourceAffineLayer(static_cast<const nn_layer_affine*>(layer->pLayerStruct))
{
    Output.Validate(Activation.Enabled, layer->nBytesPerOutput, Config.Type);

    Affine = AffineFunction::Create(&sourceAffineLayer->affine);
};

AffineMultiBiasLayer::AffineMultiBiasLayer(const nn_layer *layer, const uint32_t inputVectorCount) :
    Layer(layer, inputVectorCount),
    Activation(&static_cast<const nn_layer_affine_multi*>(layer->pLayerStruct)->pwl),
    sourceAffineLayer(static_cast<const nn_layer_affine_multi*>(layer->pLayerStruct))
{
    Output.Validate(Activation.Enabled, layer->nBytesPerOutput, Config.Type);

    Affine = AffineFunction::Create(&sourceAffineLayer->affine);

    Expect::True(Affine->BiasVectorIndex < Input.VectorCount, XNN_ERR_BIAS_INDEX);
};

AffineDiagonalLayer::AffineDiagonalLayer(const nn_layer *layer, const uint32_t inputVectorCount)
    : AffineLayer(layer, inputVectorCount)
{
    Expect::True(Output.RowCount == Input.RowCount, XNN_ERR_LYR_CFG);
    Expect::True(Input.VectorCount == Input.ColumnCount, XNN_ERR_LYR_CFG);
    Expect::True(Input.VectorCount == Output.ColumnCount, XNN_ERR_LYR_CFG);
}
