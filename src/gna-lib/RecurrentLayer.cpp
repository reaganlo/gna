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

#include "RecurrentLayer.h"

#include "Validator.h"

using namespace GNA;

RnnLayer::RnnLayer(nn_layer const * const layer, const uint32_t inputVectorCount) :
    Layer(layer, inputVectorCount),
    Affine(AffineFunction::Create(&static_cast<const nn_layer_reccurent*>(layer->pLayerStruct)->affine)),
    // RNN has only 2B output with Activation always enabled
    Activation(ActivationFunction::Create(&static_cast<const nn_layer_reccurent*>(layer->pLayerStruct)->pwl, true)),
    FeedbackDelay{static_cast<const nn_layer_reccurent * const>(layer->pLayerStruct)->feedbackFrameDelay},
    sourceLayer{static_cast<const nn_layer_reccurent * const>(layer->pLayerStruct)}
{
    Expect::InRange(FeedbackDelay, 1, Input.VectorCount-1, XNN_ERR_NO_FEEDBACK);

    // must be multiple 32 to keep 64B output buffer alignment
    Expect::MultiplicityOf(Output.ElementCount, RNN_N_OUT_ELEMS_MPLY);
    Expect::ValidBuffer(Output.ScratchPad); // intermediate output buffer must be set always
    Output.SetOutputMode(Activation.operator bool(), layer->nBytesPerOutput);

    Expect::True(Input.VectorCount == Input.RowCount, XNN_ERR_LYR_CFG);
    Expect::True(Input.VectorCount == Output.RowCount, XNN_ERR_LYR_CFG);

    if (INTEL_INPUT == Config.Type || INTEL_HIDDEN == Config.Type)
    {
        SetFeedbackBuffer();
    }
}

const OutputBuffer RnnLayer::CalculateFeedbackBuffer(const OutputBuffer& outputBuffer) const
{
    const auto buffer = outputBuffer - (FeedbackDelay * Output.ElementCount);
    Expect::ValidBuffer(buffer, XNN_ERR_NO_FEEDBACK);
    return buffer;
}

void RnnLayer::SetFeedbackBuffer(const OutputBuffer& outputBuffer)
{
    feedbackBuffer = CalculateFeedbackBuffer(outputBuffer);
    // TODO: remove when kernels use new layers
    const_cast<nn_layer_reccurent*>(sourceLayer)->pFeedbackBuffer = feedbackBuffer;
}
