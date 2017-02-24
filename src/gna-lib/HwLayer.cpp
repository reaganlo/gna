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

#include "HwLayer.h"

using namespace GNA;

const uint32_t HwLayer::nBuffElems24K[8] =
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

const uint32_t HwLayer::nBuffElems12K[8] =
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

HwLayer* HwLayer::create(NN_OP_TYPE kind)
{
    switch (kind)
    {
    case NN_RESERVED:
        throw GnaException(XNN_ERR_LYR_CFG);
 /*   case NN_RNN:
        return new HwLayerRnn();
    case NN_CNN:
        return new HwLayerCnn();
    case NN_COPY:
        return new HwLayerCopy();*/
    default:
        return new HwLayerAffDiagTrans();
    }
}

void HwLayer::init(nn_layer* lyrIn, XNN_LYR* hwLyrIn, const void* bufferIn, uint32_t hwInBuffSize, Layer* bLayerIn)
{
    Validate::IsNull(bLayerIn);
    baseLayer = bLayerIn;

    Validate::IsNull(bufferIn);
    Validate::IsNull(hwLyrIn);

    buffer = (void*)bufferIn;
    hwLyr = hwLyrIn;

    if (12 == hwInBuffSize)
    {
        nBuffElems = (uint32_t*)nBuffElems12K;
    }
    else
    {
        nBuffElems = (uint32_t*)nBuffElems24K;
    }
}

void HwLayerExt::init(
    nn_layer*		lyr,
    XNN_LYR*        hwLyr,
    const void*     buffer,
    uint32_t        hwInBuffSize,
    Layer*		bLayerIn) {

    HwLayer::init(lyr, hwLyr, buffer, hwInBuffSize, bLayerIn);
    baseLayerExt = (Layer*)baseLayer;
}

void HwLayerAffDiagTrans::init(
    nn_layer*		lyr,
    XNN_LYR*        hwLyr,
    const void*     buffer,
    uint32_t        hwInBuffSize,
    Layer*		bLayerIn) {

    HwLayerExt::init(lyr, hwLyr, buffer, hwInBuffSize, bLayerIn);
}
//
//void HwLayerCopy::init(
//    nn_layer*		lyr,
//    XNN_LYR*        hwLyr,
//    const void*     buffer,
//    uint32_t        hwInBuffSize,
//    Layer*		bLayerIn) {
//
//    HwLayer::init(lyr, hwLyr, buffer, hwInBuffSize, bLayerIn);
//    copyLayer = (CopyLayer*)baseLayer;
//}
//
//void HwLayerRnn::init(
//    nn_layer*		lyr,
//    XNN_LYR*        hwLyr,
//    const void*     buffer,
//    uint32_t        hwInBuffSize,
//    Layer*		bLayerIn) {
//
//    HwLayerExt::init(lyr, hwLyr, buffer, hwInBuffSize, bLayerIn);
//    rnnLayer = (RnnLayer*)baseLayer;
//}
//
//void HwLayerCnn::init(
//    nn_layer*		lyr,
//    XNN_LYR*        hwLyr,
//    const void*     buffer,
//    uint32_t        hwInBuffSize,
//    Layer*		bLayerIn) {
//
//    HwLayerExt::init(lyr, hwLyr, buffer, hwInBuffSize, bLayerIn);
//    cnnLayer = (CnnLayer*)baseLayer;
//}

void HwLayer::convert()
{
}

void HwLayerAffDiagTrans::convert()
{
    calcIterations(baseLayer->Input.VectorCount);
    HwLayerExt::convert();
    save();
}
//
//void HwLayerCopy::convert()
//{
//    HwLayer::convert();
//    validate();
//    save();
//}
//
//void HwLayerRnn::convert()
//{
//    HwLayerExt::convert();
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
//void HwLayerCnn::convert()
//{
//    HwLayerExt::convert();
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

void HwLayerExt::convert()
{
    HwLayer::convert();
}

void HwLayerExt::calcIterations(uint32_t nGrIn)
{
    nGr = nGrIn;
    Validate::IsTrue(nGr < 1, XNN_ERR_GROUPING);
    Validate::IsTrue(nGr > XNN_N_GROUP_MAX, XNN_ERR_GROUPING);
    nIters = ((baseLayer->Input.ElementCount * nGr - 1) / nBuffElems[nGr - 1]) + 1;
    nLast = ((baseLayer->Input.ElementCount * nGr) - ((nIters - 1) * nBuffElems[nGr - 1])) / nGr;
}

void HwLayer::convertAL(ActiveList* al)
{
    Validate::IsNull(hwLyr);
    Validate::IsNull(al);
    if (al->enabled)
    {
        Validate::IsAlignedTo64(al->indices);
        Validate::IsTrue(al->indicesCount > XNN_N_IN_ELEMS_MAX, XNN_ERR_LYR_CFG);
        hwLyr->act_list_n_elems = (uint16_t)al->indicesCount;
        hwLyr->act_list_buffer = Hw::getAddrOffset(al->indices, buffer);
        if (NN_AFFINE == hwLyr->op)
        {
            hwLyr->op = NN_AFF_AL;
        }
        else if (NN_GMM == hwLyr->op)
        {
            hwLyr->op = NN_GMM_ACTIVE_LIST;
        }
    }
}

void HwLayer::validate()
{
}

void HwLayerExt::validate()
{
    Validate::IsTrue(nIters < 1, XNN_ERR_LYR_CFG);
    Validate::IsTrue(nIters > UINT8_MAX, XNN_ERR_LYR_CFG);
    Validate::IsTrue(nLast < 1, XNN_ERR_LYR_CFG);
    Validate::IsTrue(nLast > nBuffElems[nGr - 1], XNN_ERR_LYR_CFG);
    Validate::IsMultiplicityOf(nLast, XNN_N_IN_ELEMS_MPLY);
}
//
//void HwLayerCopy::validate()
//{
//}
//
//void HwLayerRnn::validate()
//{
//    HwLayerExt::validate();
//}
//
//void HwLayerCnn::validate()
//{
//    Validate::IsTrue(nFltsPerIter      < CNN_N_FLT_COEFF_MPLY, XNN_ERR_LYR_CFG);
//    Validate::IsTrue(nFltsPerIter      > CNN_N_FLT_ITER_MAX, XNN_ERR_LYR_CFG);
//    Validate::IsMultiplicityOf(nFltsPerIter, CNN_N_FLT_COEFF_MPLY);
//    Validate::IsTrue(nFltsLast         < CNN_N_FLT_COEFF_MPLY, XNN_ERR_LYR_CFG);
//    Validate::IsTrue(nFltsLast         > CNN_N_FLT_ITER_MAX, XNN_ERR_LYR_CFG);
//    Validate::IsMultiplicityOf(nFltsLast, CNN_N_FLT_COEFF_MPLY);
//    Validate::IsTrue(fltBuffSz         < 1, XNN_ERR_LYR_CFG);
//    Validate::IsTrue(fltBuffSz         > nBuffElems[0], XNN_ERR_LYR_CFG);
//    Validate::IsTrue(fltBuffSzLast     < 1, XNN_ERR_LYR_CFG);
//    Validate::IsTrue(fltBuffSzLast     > nBuffElems[0], XNN_ERR_LYR_CFG);
//}

void HwLayer::save()
{
    hwLyr->op = (NN_OP_TYPE)baseLayer->Config.Operation;
    hwLyr->n_in_elems = (uint16_t)baseLayer->Input.ElementCount;
    hwLyr->n_out_elems = (uint16_t)baseLayer->Output.ElementCount;
    hwLyr->n_groups = (uint8_t)baseLayer->Input.VectorCount;
    if (INTEL_GMM == baseLayer->Config.Type)
    {
        hwLyr->gmm_descriptor = 0; // TODO:KJ: set actual gmm descriptor address
    }
    // TODO:KJ: add I/O configuration conversion to Request config
    else
    {
        hwLyr->in_buffer = Hw::getAddrOffset(baseLayer->Input.Buffer, buffer);
    }
    hwLyr->out_act_fn_buffer = Hw::getAddrOffset(baseLayer->Output.Buffer, buffer);
    hwLyr->out_sum_buffer = Hw::getAddrOffset(baseLayer->Output.BufferIntermediate, buffer);
}

void HwLayerExt::save()
{
    HwLayer::save();
    //hwLyr->flags.act_fn_en = baseLayerExt->pwl ? 1 : 0;
    //hwLyr->n_iters = (uint8_t)nIters;
    //hwLyr->n_elems_last = (uint16_t)nLast;
    //if (baseLayerExt->aff)    // affine layers
    //{
    //    hwLyr->flags.weight_size = (baseLayerExt->aff->nBytesPerWeight == 2) ? 0 : 1;
    //    hwLyr->aff_weight_buffer = Hw::getAddrOffset(baseLayerExt->aff->pWeights, buffer);
    //    hwLyr->aff_const_buffer = Hw::getAddrOffset(baseLayerExt->aff->pBiases, buffer);
    //}
    //if (baseLayerExt->pwl)    // pwl enabled layers
    //{
    //    hwLyr->pwl_n_segs = (uint8_t)baseLayerExt->pwl->nSegments;
    //    hwLyr->pwl_seg_def_buffer = Hw::getAddrOffset(baseLayerExt->pwl->pSegments, buffer);
    //}
}

void HwLayerAffDiagTrans::save()
{
    HwLayerExt::save();
}
//
//void HwLayerCopy::save()
//{
//    HwLayer::save();
//    hwLyr->cpy_n_elems = (uint16_t)copyLayer->cpy->nCopyCols;
//}
//
//void HwLayerRnn::save()
//{
//    HwLayerExt::save();
//    hwLyr->rnn_n_fb_iters = nFbIters;
//    hwLyr->rnn_n_elems_first = nFbFirst;
//    hwLyr->rnn_n_elems_last = nFbLast;
//    hwLyr->rnn_out_fb_buffer = Hw::getAddrOffset(rnnLayer->rnn->pFeedbackBuffer, buffer);
//}
//
//void HwLayerCnn::save()
//{
//    HwLayerExt::save();
//    // some fields saved by HwLayerExt will be overwritten
//    hwLyr->flags.pool_param = (uint8_t)cnnLayer->cnn->poolType;
//    hwLyr->cnn_flt_bf_sz_iter = fltBuffSz;
//    hwLyr->cnn_flt_bf_sz_last = fltBuffSzLast;
//    hwLyr->cnn_flt_buffer = Hw::getAddrOffset(cnnLayer->cnn->pFilters, buffer);;
//    hwLyr->cnn_flt_size = cnnLayer->cnn->nFilterCoefficients;
//    hwLyr->cnn_n_flts = cnnLayer->cnn->nFilters;
//    hwLyr->cnn_n_flts_iter = nFltsPerIter;
//    hwLyr->cnn_n_flt_iters = nFltIters;
//    hwLyr->cnn_n_flt_last = nFltsLast;
//    hwLyr->cnn_n_flt_outs = cnnLayer->nFltOutElems;
//    hwLyr->cnn_n_flt_stride = cnnLayer->fltStrideSz;
//    hwLyr->cnn_n_out_p_flt = cnnLayer->ElementCount;
//    hwLyr->cnn_pool_size = cnnLayer->cnn->nPoolSize;
//    hwLyr->cnn_pool_stride = cnnLayer->cnn->nPoolStride;
//    hwLyr->aff_const_buffer = Hw::getAddrOffset(cnnLayer->cnn->pBiases, buffer);
//}
