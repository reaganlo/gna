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

#include "SimpleLayers.h"

#include "Validator.h"

using namespace GNA;

TransposeLayer::TransposeLayer(nn_layer const * const layer, const uint32_t inputVectorCount) :
    Layer(layer, inputVectorCount)
{
    Expect::True(Input.RowCount == Output.ColumnCount, XNN_ERR_LYR_CFG);
    Expect::True(Input.ColumnCount == Output.RowCount, XNN_ERR_LYR_CFG);
    Expect::Null(layer->pLayerStruct); // transpose layers do not have layer details
    Expect::Null(Output.ScratchPad); // in transpose layer no 4B output array is allowed
}

CopyLayer::CopyLayer(const nn_layer *layer) :
    Layer(layer, static_cast<const nn_layer_copy*>(layer->pLayerStruct)->nCopyRows),
    CopyElementsCount(static_cast<const nn_layer_copy*>(layer->pLayerStruct)->nCopyCols),
    sourceLayer(static_cast<const nn_layer_copy*>(layer->pLayerStruct))
{
    Expect::MultiplicityOf(CopyElementsCount, XNN_N_IN_ELEMS_MPLY);
    Expect::InRange(CopyElementsCount, XNN_N_IN_ELEMS_MPLY, XNN_N_IN_ELEMS_MAX, XNN_ERR_LYR_CFG);
    Expect::True(Input.VectorCount <= Input.RowCount, XNN_ERR_LYR_CFG);
}
