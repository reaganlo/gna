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

#include "HardwareLayer.h"
#include "Validator.h"

using namespace GNA;

const map<const nn_layer_kind, const NN_OP_TYPE> HardwareLayer::OperationsMap =
{
    { INTEL_AFFINE, NN_AFFINE },
    { INTEL_AFFINE_DIAGONAL, NN_DIAG },
    { INTEL_AFFINE_MULTIBIAS, NN_AFF_MB },
    { INTEL_CONVOLUTIONAL, NN_CNN },
    { INTEL_COPY, NN_COPY },
    { INTEL_DEINTERLEAVE, NN_DEINT },
    { INTEL_GMM, NN_GMM },
    { INTEL_INTERLEAVE, NN_INTER },
    { INTEL_RECURRENT, NN_RNN }
};

const uint32_t HardwareLayer::nBuffElems24K[8] =
{
    12288,
    12288,
    12096,
    12288,
    12000,
    12096,
    12096,
    12288
};

const uint32_t HardwareLayer::nBuffElems12K[8] =
{
    6144,
    6144,
    6048,
    6144,
    5760,
    6048,
    6048,
    6144
};

unique_ptr<HardwareLayer> HardwareLayer::Create(const Layer& softwareLayer, XNN_LYR *layerDescriptor, void *descriptorBuffer, uint32_t hwInBufferSize)
{
    switch (OperationsMap.at(softwareLayer.Config.Kind))
    {
    //case NN_CNN:
        //    return new HardwareLayerCnn();
    case NN_COPY:
        return make_unique<HardwareLayerCopy>(softwareLayer, layerDescriptor, descriptorBuffer, hwInBufferSize);
    //case NN_GMM:
        //    return new HwGmmLayer();
    case NN_RNN:
        return make_unique<HardwareLayerRnn>(softwareLayer, layerDescriptor, descriptorBuffer, hwInBufferSize);
    default:
        return make_unique<HardwareLayerAffDiagTrans>(softwareLayer, layerDescriptor, descriptorBuffer, hwInBufferSize);
    }
}

HardwareLayer::HardwareLayer(const Layer &swLayer, XNN_LYR *layerDesc, void *descBuffer, uint32_t hwInBuffSize) :
    softwareLayer(swLayer),
    layerDescriptor(layerDesc),
    descriptorBuffer(descBuffer)
{
    Expect::NotNull(layerDescriptor);
    Expect::NotNull(descriptorBuffer);

    if (12 == hwInBuffSize)
    {
        nBuffElems = static_cast<const uint32_t*>(nBuffElems12K);
    }
    else
    {
        nBuffElems = static_cast<const uint32_t*>(nBuffElems24K);
    }
}

HardwareLayerExt::HardwareLayerExt(const Layer& swLayer, XNN_LYR *layerDesc, void *descBuffer,
    uint32_t hwInBufferSize, uint32_t effectiveGrouping) :
    HardwareLayer(swLayer, layerDesc, descBuffer, hwInBufferSize),
    iterationGrouping(effectiveGrouping),
    bufferElementCount(nBuffElems[iterationGrouping - 1])
{
    Expect::InRange(iterationGrouping, 1, XNN_N_GROUP_MAX, XNN_ERR_GROUPING);
    // Calculates number of iterations and elements in last iteration
     //#groups for calculation(can be different than network grouping)
    auto elementsTimesGrouping = softwareLayer.Input.ElementCount * iterationGrouping;
    nIters = ((elementsTimesGrouping - 1) / bufferElementCount) + 1;
    nLast = ((elementsTimesGrouping) - ((nIters - 1) * bufferElementCount)) / iterationGrouping;
}

HardwareLayerAffDiagTrans::HardwareLayerAffDiagTrans(const Layer& swLayer, XNN_LYR *layerDesc,
    void *descBuffer, uint32_t hwInBuffSize) :
    HardwareLayerExt(swLayer, layerDesc, descBuffer, hwInBuffSize, softwareLayer.Input.VectorCount)
{
    switch (softwareLayer.Config.Kind)
    {
    case INTEL_AFFINE:
    case INTEL_AFFINE_DIAGONAL:
    case INTEL_AFFINE_MULTIBIAS:
        auto& aff = static_cast<const AffineLayer&>(softwareLayer);
        affine = aff.Affine.get();
        activation = const_cast<ActivationFunction*>(&aff.Activation);
        break;
    }
    validate();
    save();
}

HardwareLayerCopy::HardwareLayerCopy(const Layer& swLayer, XNN_LYR *layerDesc, void *descBuffer,
    uint32_t hwInBuffSize) :
    HardwareLayer(swLayer, layerDesc, descBuffer, hwInBuffSize)
{
    validate();
    save();
}

HardwareLayerRnn::HardwareLayerRnn(const Layer& swLayer, XNN_LYR *layerDesc, void *descBuffer,
    uint32_t hwInBuffSize) :
    HardwareLayerExt(swLayer, layerDesc, descBuffer, hwInBuffSize, 1),
    nFbIters(0),
    nFbFirst(0),
    nFbLast(0)
{
    auto& rnn = static_cast<const RnnLayer&>(softwareLayer);
    affine = rnn.Affine.get();
    activation = const_cast<ActivationFunction*>(&rnn.Activation);
    convert();
    validate();
    save();
};

void HardwareLayerRnn::convert()
{
    auto elementCount = softwareLayer.Input.ElementCount;

    nFbFirst = min((bufferElementCount - nLast), elementCount);
    Expect::True(nFbFirst <= bufferElementCount, XNN_ERR_LYR_CFG);

    nFbIters = (elementCount - nFbFirst) / bufferElementCount;
    if ((elementCount - nFbFirst) % bufferElementCount)
    {
        nFbIters++;
    }
    if (nFbFirst > 0)
    {
        nFbIters++;
    }
    Expect::InRange(nFbIters, 1, UINT8_MAX, XNN_ERR_LYR_CFG);

    if (nFbFirst && 1 == nFbIters)
    {
        nFbLast = nFbFirst;
    }
    else if (0 == nFbFirst)
    {
        nFbLast = elementCount - nFbFirst - (nFbIters - 1) * bufferElementCount;
    }
    else
    {
        nFbLast = elementCount - nFbFirst - (nFbIters - 2) * bufferElementCount;
    }
    Expect::InRange(nFbLast, 1, bufferElementCount, XNN_ERR_LYR_CFG);
}

//void HardwareLayerCnn::init(
//    nn_layer*		lyr,
//    XNN_LYR*        layerDescriptor,
//    const void*     buffer,
//    uint32_t        hwInBuffSize,
//    Layer*		bLayerIn) {
//
//    HardwareLayerExt::init(lyr, layerDescriptor, buffer, hwInBuffSize, bLayerIn);
//    cnnLayer = (CnnLayer*)baseLayer;
//}


void HardwareLayerCopy::validate() {}

//void HardwareLayerCnn::convert()
//{
//    HardwareLayerExt::convert();
//
//    uint32_t nFlts = cnnLayer->cnn->nFilters;
//    uint32_t nFltSize = cnnLayer->cnn->nFilterCoefficients;
//    uint32_t maxNCOE = (cnnLayer->ElementCount - nFltSize) / cnnLayer->fltStrideSz + 1;
//    nFltsPerIter =
//        min(
//        nFlts,
//        (nFltSize <= bufferElementCount / 6 / 3) ?
//        16 :
//        (nFltSize <= bufferElementCount / 6 / 2) ?
//        12 :
//        (nFltSize <= bufferElementCount / 6) ?
//        4 :
//        0);
//    Validate::IsTrue(0 == nFltsPerIter, CNN_ERR_FLT_COUNT);
//    nFltIters = (nFlts - 1) / nFltsPerIter + 1;
//    nFltsLast = nFlts - ((nFltIters - 1) * nFltsPerIter);
//    fltBuffSz = nFltsPerIter * nFltSize;
//    fltBuffSzLast = nFltsLast * nFltSize;
//
//    validate();
//    save();
//}

void HardwareLayer::convertAL(ActiveList& activeList)
{
    if (activeList.Enabled)
    {
        layerDescriptor->act_list_n_elems = static_cast<uint16_t>(activeList.IndicesCount);
        layerDescriptor->act_list_buffer = getOffset(activeList.Indices);
        if (NN_AFFINE == layerDescriptor->op)
        {
            layerDescriptor->op = NN_AFF_AL;
        }
        else if (NN_GMM == layerDescriptor->op)
        {
            layerDescriptor->op = NN_GMM_ACTIVE_LIST;
        }
    }
}

void HardwareLayerExt::validate()
{
    Expect::InRange(nIters, 1, UINT8_MAX, XNN_ERR_LYR_CFG);
    Expect::InRange(nLast, 1, bufferElementCount, XNN_ERR_LYR_CFG);
    Expect::MultiplicityOf(nLast, XNN_N_IN_ELEMS_MPLY);
}

void HardwareLayerRnn::validate()
{
    HardwareLayerExt::validate();
}

//void HardwareLayerCnn::validate()
//{
//    Expect::True(nFltsPerIter      < CNN_N_FLT_COEFF_MPLY, XNN_ERR_LYR_CFG);
//    Expect::True(nFltsPerIter      > CNN_N_FLT_ITER_MAX, XNN_ERR_LYR_CFG);
//    Expect::MultiplicityOf(nFltsPerIter, CNN_N_FLT_COEFF_MPLY);
//    Expect::True(nFltsLast         < CNN_N_FLT_COEFF_MPLY, XNN_ERR_LYR_CFG);
//    Expect::True(nFltsLast         > CNN_N_FLT_ITER_MAX, XNN_ERR_LYR_CFG);
//    Expect::MultiplicityOf(nFltsLast, CNN_N_FLT_COEFF_MPLY);
//    Expect::True(fltBuffSz         < 1, XNN_ERR_LYR_CFG);
//    Expect::True(fltBuffSz         > bufferElementCount, XNN_ERR_LYR_CFG);
//    Expect::True(fltBuffSzLast     < 1, XNN_ERR_LYR_CFG);
//    Expect::True(fltBuffSzLast     > bufferElementCount, XNN_ERR_LYR_CFG);
//}

void HardwareLayer::save()
{
    layerDescriptor->op = OperationsMap.at(softwareLayer.Config.Kind);
    layerDescriptor->n_in_elems = static_cast<uint16_t>(softwareLayer.Input.ElementCount);
    layerDescriptor->n_out_elems = static_cast<uint16_t>(softwareLayer.Output.ElementCount);
    layerDescriptor->n_groups = static_cast<uint8_t>(softwareLayer.Input.VectorCount);
    if (INTEL_GMM == softwareLayer.Config.Type)
    {
        layerDescriptor->gmm_descriptor = 0; // TODO:KJ: set actual gmm descriptor address
    }
    // TODO:KJ: add I/O configuration conversion to Request config
    else
    {
        layerDescriptor->in_buffer = getOffset(softwareLayer.Input.Buffer);
    }
    layerDescriptor->out_act_fn_buffer = getOffset(softwareLayer.Output.Buffer);
    layerDescriptor->out_sum_buffer = getOffset(softwareLayer.Output.ScratchPad);
}

void HardwareLayerExt::save()
{
    HardwareLayer::save();
    layerDescriptor->n_iters = static_cast<uint8_t>(nIters);
    layerDescriptor->n_elems_last = static_cast<uint16_t>(nLast);

    if (affine)
    {
        layerDescriptor->flags.weight_size = (affine->GetWeightMode() == GNA_WEIGHT_2B) ? 0 : 1;
        layerDescriptor->aff_weight_buffer = getOffset(affine->GetWeights());
        layerDescriptor->aff_const_buffer = getOffset(affine->GetBiases());

    }
    if (activation && activation->Enabled)
    {
        layerDescriptor->flags.act_fn_en = 1;
        layerDescriptor->pwl_n_segs = static_cast<uint8_t>(activation->SegmentCount);
        layerDescriptor->pwl_seg_def_buffer = getOffset(activation->Segments);
    }
    else
    {
        layerDescriptor->flags.act_fn_en = 0;
    }
}

void HardwareLayerCopy::save()
{
    HardwareLayer::save();
    auto& copy = static_cast<const CopyLayer&>(softwareLayer);
    layerDescriptor->cpy_n_elems = static_cast<uint16_t>(copy.CopyElementsCount);
}

void HardwareLayerRnn::save()
{
    HardwareLayerExt::save();
    auto& rnn = static_cast<const RnnLayer&>(softwareLayer);
    layerDescriptor->rnn_n_fb_iters = nFbIters;
    layerDescriptor->rnn_n_elems_first = nFbFirst;
    layerDescriptor->rnn_n_elems_last = nFbLast;
    // will be 0 for hidden layers
    layerDescriptor->rnn_out_fb_buffer = calculateFeedbackBuffer(softwareLayer.Output.Buffer);
}

const uint32_t HardwareLayerRnn::calculateFeedbackBuffer(const void * const outputBuffer) const
{
    auto& rnn = static_cast<const RnnLayer&>(softwareLayer);
    return getOffset(rnn.CalculateFeedbackBuffer(outputBuffer));
}

//void HardwareLayerCnn::save()
//{
//    HardwareLayerExt::save();
//    // some fields saved by HardwareLayerExt will be overwritten
//    layerDescriptor->flags.pool_param = static_cast<uint8_t>(cnnLayer->cnn->poolType);
//    layerDescriptor->cnn_flt_bf_sz_iter = fltBuffSz;
//    layerDescriptor->cnn_flt_bf_sz_last = fltBuffSzLast;
//    layerDescriptor->cnn_flt_buffer = getOffset(cnnLayer->cnn->pFilters);;
//    layerDescriptor->cnn_flt_size = cnnLayer->cnn->nFilterCoefficients;
//    layerDescriptor->cnn_n_flts = cnnLayer->cnn->nFilters;
//    layerDescriptor->cnn_n_flts_iter = nFltsPerIter;
//    layerDescriptor->cnn_n_flt_iters = nFltIters;
//    layerDescriptor->cnn_n_flt_last = nFltsLast;
//    layerDescriptor->cnn_n_flt_outs = cnnLayer->nFltOutElems;
//    layerDescriptor->cnn_n_flt_stride = cnnLayer->fltStrideSz;
//    layerDescriptor->cnn_n_out_p_flt = cnnLayer->ElementCount;
//    layerDescriptor->cnn_pool_size = cnnLayer->cnn->nPoolSize;
//    layerDescriptor->cnn_pool_stride = cnnLayer->cnn->nPoolStride;
//    layerDescriptor->aff_const_buffer = getOffset(cnnLayer->cnn->pBiases);
//}