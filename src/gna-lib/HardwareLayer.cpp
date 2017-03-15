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

const std::map<const nn_layer_kind, const NN_OP_TYPE> HardwareLayer::OperationsMap =
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
    case NN_AFFINE:
        return make_unique<HardwareLayerAffDiagTrans>(softwareLayer, layerDescriptor, descriptorBuffer, hwInBufferSize);
    //case NN_RNN:
    //    return new HardwareLayerRnn();
    //case NN_CNN:
    //    return new HardwareLayerCnn();
    //case NN_GMM:
    //    return new HwGmmLayer();
    case NN_COPY:
        return make_unique<HardwareLayerCopy>(softwareLayer, layerDescriptor, descriptorBuffer, hwInBufferSize);
    default:
        throw GnaException(XNN_ERR_LYR_CFG);
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

HardwareLayerExt::HardwareLayerExt(const Layer& swLayer, XNN_LYR *layerDesc, void *descBuffer, uint32_t hwInBuffSize) :
    HardwareLayer(swLayer, layerDesc, descBuffer, hwInBuffSize) {}

HardwareLayerAffDiagTrans::HardwareLayerAffDiagTrans(const Layer& swLayer, XNN_LYR *layerDesc, void *descBuffer, uint32_t hwInBuffSize) :
    HardwareLayerExt(swLayer, layerDesc, descBuffer, hwInBuffSize),
    affineLayer(static_cast<const AffineLayer&>(softwareLayer))
{
    convert();
}

HardwareLayerCopy::HardwareLayerCopy(const Layer& swLayer, XNN_LYR *layerDesc, void *descBuffer, uint32_t hwInBuffSize) :
    HardwareLayer(swLayer, layerDesc, descBuffer, hwInBuffSize),
    copyLayer(static_cast<const CopyLayer&>(softwareLayer))
{
    convert();
}

//void HardwareLayerRnn::init(
//    nn_layer*		lyr,
//    XNN_LYR*        layerDescriptor,
//    const void*     buffer,
//    uint32_t        hwInBuffSize,
//    Layer*		bLayerIn) {
//
//    HardwareLayerExt::init(lyr, layerDescriptor, buffer, hwInBuffSize, bLayerIn);
//    rnnLayer = (RnnLayer*)baseLayer;
//}
//
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

void HardwareLayerAffDiagTrans::convert()
{
    calcIterations(softwareLayer.Input.VectorCount);
    validate();
    save();
}

void HardwareLayerCopy::convert()
{
    validate();
    save();
}

void HardwareLayerCopy::validate() {}

//void HardwareLayerRnn::convert()
//{
//    HardwareLayerExt::convert();
//    calcIterations(1);
//
//    nFbFirst = min((nBuffElems[0] - nLast), rnnLayer->ElementCount);
//    Validate::IsTrue(nFbFirst > nBuffElems[0], XNN_ERR_LYR_CFG);
//
//    nFbIters = (rnnLayer->ElementCount - nFbFirst) / nBuffElems[0];
//    if ((rnnLayer->ElementCount - nFbFirst) % nBuffElems[0])
//    {
//        nFbIters++;
//    }
//    if (nFbFirst > 0)
//    {
//        nFbIters++;
//    }
//    Validate::IsTrue(nFbIters < 1, XNN_ERR_LYR_CFG);
//    Validate::IsTrue(nFbIters > UINT8_MAX, XNN_ERR_LYR_CFG);
//
//    if (nFbFirst && 1 == nFbIters)
//    {
//        nFbLast = nFbFirst;
//    }
//    else if (0 == nFbFirst)
//    {
//        nFbLast = rnnLayer->ElementCount - nFbFirst - (nFbIters - 1) * nBuffElems[0];
//    }
//    else
//    {
//        nFbLast = rnnLayer->ElementCount - nFbFirst - (nFbIters - 2) * nBuffElems[0];
//    }
//    Validate::IsTrue(nFbLast < 1, XNN_ERR_LYR_CFG);
//    Validate::IsTrue(nFbLast > nBuffElems[0], XNN_ERR_LYR_CFG);
//
//    validate();
//    save();
//}
//
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
//        (nFltSize <= nBuffElems[0] / 6 / 3) ?
//        16 :
//        (nFltSize <= nBuffElems[0] / 6 / 2) ?
//        12 :
//        (nFltSize <= nBuffElems[0] / 6) ?
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

void HardwareLayerExt::calcIterations(uint32_t nGrIn)
{
    nGr = nGrIn;
    nIters = ((softwareLayer.Input.ElementCount * nGr - 1) / nBuffElems[nGr - 1]) + 1;
    nLast = ((softwareLayer.Input.ElementCount * nGr) - ((nIters - 1) * nBuffElems[nGr - 1])) / nGr;
}

void HardwareLayer::convertAL(ActiveList& activeList)
{
    if (activeList.Enabled)
    {
        Expect::AlignedTo64(activeList.Indices);
        Expect::True(activeList.IndicesCount > XNN_N_IN_ELEMS_MAX, XNN_ERR_LYR_CFG);
        layerDescriptor->act_list_n_elems = static_cast<uint16_t>(activeList.IndicesCount);
        layerDescriptor->act_list_buffer = Hw::getAddrOffset(activeList.Indices, descriptorBuffer);
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
    Expect::False(nIters < 1, XNN_ERR_LYR_CFG);
    Expect::False(nIters > UINT8_MAX, XNN_ERR_LYR_CFG);
    Expect::False(nLast < 1, XNN_ERR_LYR_CFG);
    Expect::False(nLast > nBuffElems[nGr - 1], XNN_ERR_LYR_CFG);
    Expect::MultiplicityOf(nLast, XNN_N_IN_ELEMS_MPLY);
}

//void HardwareLayerRnn::validate()
//{
//    HardwareLayerExt::validate();
//}
//
//void HardwareLayerCnn::validate()
//{
//    Expect::True(nFltsPerIter      < CNN_N_FLT_COEFF_MPLY, XNN_ERR_LYR_CFG);
//    Expect::True(nFltsPerIter      > CNN_N_FLT_ITER_MAX, XNN_ERR_LYR_CFG);
//    Expect::MultiplicityOf(nFltsPerIter, CNN_N_FLT_COEFF_MPLY);
//    Expect::True(nFltsLast         < CNN_N_FLT_COEFF_MPLY, XNN_ERR_LYR_CFG);
//    Expect::True(nFltsLast         > CNN_N_FLT_ITER_MAX, XNN_ERR_LYR_CFG);
//    Expect::MultiplicityOf(nFltsLast, CNN_N_FLT_COEFF_MPLY);
//    Expect::True(fltBuffSz         < 1, XNN_ERR_LYR_CFG);
//    Expect::True(fltBuffSz         > nBuffElems[0], XNN_ERR_LYR_CFG);
//    Expect::True(fltBuffSzLast     < 1, XNN_ERR_LYR_CFG);
//    Expect::True(fltBuffSzLast     > nBuffElems[0], XNN_ERR_LYR_CFG);
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
        layerDescriptor->in_buffer = Hw::getAddrOffset(softwareLayer.Input.Buffer, descriptorBuffer);
    }
    layerDescriptor->out_act_fn_buffer = Hw::getAddrOffset(softwareLayer.Output.Buffer, descriptorBuffer);
    layerDescriptor->out_sum_buffer = Hw::getAddrOffset(softwareLayer.Output.ScratchPad, descriptorBuffer);
}

void HardwareLayerExt::save()
{
    HardwareLayer::save();
    layerDescriptor->n_iters = static_cast<uint8_t>(nIters);
    layerDescriptor->n_elems_last = static_cast<uint16_t>(nLast);

    const ActivationFunction *pwl = nullptr;
    if (INTEL_AFFINE == softwareLayer.Config.Kind
        || INTEL_AFFINE_DIAGONAL == softwareLayer.Config.Kind)
    {
        auto& affineLayer = static_cast<const AffineLayer&>(softwareLayer);
        auto affineFunction = affineLayer.Affine.get();

        auto weightMode = affineLayer.Affine->GetWeightMode();
        layerDescriptor->flags.weight_size = (weightMode == GNA_WEIGHT_2B) ? 0 : 1;
        layerDescriptor->aff_weight_buffer = Hw::getAddrOffset(affineFunction->GetWeights(), descriptorBuffer);
        layerDescriptor->aff_const_buffer = Hw::getAddrOffset(affineFunction->GetBiases(), descriptorBuffer);

        pwl = &affineLayer.Activation;
    }
    // TODO: (1) uncomment after layers are enabled, hopefully it will work
    //       (2) consider another virtual method for getting Activation out of Layer
    //if (INTEL_RECURRENT == softwareLayer.Config.Kind)
    //{
    //    auto& rnnLayer = static_cast<const RecurrentLayer&>(softwareLayer);
    //    pwl = &rnnLayer.Activation;
    //}
    //if(INTEL_CONVOLUTIONAL == softwareLayer.Config.Kind)
    //{
    //    auto& cnnLayer = static_cast<const ConvolutionalLayer&>(softwareLayer);
    //    pwl = &cnnLayer.Activation;
    //}
    if (pwl && pwl->Enabled)
    {
        layerDescriptor->flags.act_fn_en = 1;
        layerDescriptor->pwl_n_segs = static_cast<uint8_t>(pwl->SegmentCount);
        layerDescriptor->pwl_seg_def_buffer = Hw::getAddrOffset(pwl->Segments, descriptorBuffer);
    }
    else
    {
        layerDescriptor->flags.act_fn_en = 0;
    }
}

void HardwareLayerCopy::save()
{
    HardwareLayer::save();
    layerDescriptor->cpy_n_elems = static_cast<uint16_t>(copyLayer.CopyElementsCount);
}

//void HardwareLayerRnn::save()
//{
//    HardwareLayerExt::save();
//    layerDescriptor->rnn_n_fb_iters = nFbIters;
//    layerDescriptor->rnn_n_elems_first = nFbFirst;
//    layerDescriptor->rnn_n_elems_last = nFbLast;
//    layerDescriptor->rnn_out_fb_buffer = Hw::getAddrOffset(rnnLayer->rnn->pFeedbackBuffer, buffer);
//}
//
//void HardwareLayerCnn::save()
//{
//    HardwareLayerExt::save();
//    // some fields saved by HardwareLayerExt will be overwritten
//    layerDescriptor->flags.pool_param = static_cast<uint8_t>(cnnLayer->cnn->poolType);
//    layerDescriptor->cnn_flt_bf_sz_iter = fltBuffSz;
//    layerDescriptor->cnn_flt_bf_sz_last = fltBuffSzLast;
//    layerDescriptor->cnn_flt_buffer = Hw::getAddrOffset(cnnLayer->cnn->pFilters, buffer);;
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
//    layerDescriptor->aff_const_buffer = Hw::getAddrOffset(cnnLayer->cnn->pBiases, buffer);
//}