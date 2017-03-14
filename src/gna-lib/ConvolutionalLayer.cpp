///*
// INTEL CONFIDENTIAL
// Copyright 2017 Intel Corporation.
//
// The source code contained or described herein and all documents related
// to the source code ("Material") are owned by Intel Corporation or its suppliers
// or licensors. Title to the Material remains with Intel Corporation or its suppliers
// and licensors. The Material may contain trade secrets and proprietary
// and confidential information of Intel Corporation and its suppliers and licensors,
// and is protected by worldwide copyright and trade secret laws and treaty provisions.
// No part of the Material may be used, copied, reproduced, modified, published,
// uploaded, posted, transmitted, distributed, or disclosed in any way without Intel's
// prior express written permission.
//
// No license under any patent, copyright, trade secret or other intellectual
// property right is granted to or conferred upon you by disclosure or delivery
// of the Materials, either expressly, by implication, inducement, estoppel
// or otherwise. Any license under such intellectual property rights must
// be express and approved by Intel in writing.
//
// Unless otherwise agreed by Intel in writing, you may not remove or alter this notice
// or any other notice embedded in Materials by Intel or Intel's suppliers or licensors
// in any way.
//*/
//
//#include "ConvolutionalLayer.h"
//
//using namespace GNA;
//
//void CnnLayer::convert()
//{
//    uint32_t nFlts = 0;    // number of all filters
//    uint32_t nFltSize = 0;    // number of conv. filter elements
//    uint32_t maxNCOE = 0;    // max number of conv. out elems.
//
//    BaseLayerExt::convert();
//    
//    Expect::NotNull(lyr->pLayerStruct);
//    cnn = (nn_layer_conv*)lyr->pLayerStruct;
//    pwl = (cnn->pwl.nSegments > 0) ? &cnn->pwl : nullptr;
//    ElementCount = lyr->ColumnCount;// FLAT input matrix
//    fltStrideSz = cnn->nFeatureMaps * cnn->nFeatureMapColumns; // always move 1 "row"
//    Expect::True(0 != fltStrideSz, CNN_ERR_FLT_STRIDE);
//
//    nFltSize = cnn->nFilterCoefficients;
//    maxNCOE = (ElementCount - nFltSize) / fltStrideSz + 1;
//    nFltOutElems = maxNCOE;
//    if (INTEL_NO_POOLING == cnn->poolType)// FLAT input matrix, conv. outputs per filter
//    {
//        ElementCount = maxNCOE;
//    }
//    else // FLAT input matrix, pooled outputs per filter
//    {
//        Expect::True(0 != fltStrideSz, CNN_ERR_POOL_STRIDE);
//        ElementCount = ((maxNCOE - 1) / cnn->nPoolStride + 1);
//    }
//    validate();
//}
//
//void CnnLayer::validate()
//{
//    BaseLayerExt::validate();
//    
//    Expect::True(nGroup == 1, XNN_ERR_GROUPING);
//    Expect::True(nGroup == lyr->RowCount, XNN_ERR_GROUPING);
//    Expect::True(nGroup == lyr->RowCount, XNN_ERR_GROUPING);
//
//    Expect::NotNull(lyr->Buffer);
//    Expect::NotNull(lyr->ScratchPad);
//
//    Expect::ValidBuffer(cnn->pFilters);
//
//    Expect::ValidBuffer(cnn->pBiases);
//
//    Expect::False(cnn->nFilterCoefficients < CNN_N_FLT_COEFF_MIN, XNN_ERR_LYR_CFG);
//    Expect::False(cnn->nFilterCoefficients > CNN_N_FLT_COEFF_MAX, XNN_ERR_LYR_CFG);
//    Expect::False(cnn->nFilterCoefficients > ElementCount, XNN_ERR_LYR_CFG);
//    Expect::MultiplicityOf(cnn->nFilterCoefficients, XNN_N_IN_ELEMS_MPLY);
//
//    Expect::False(cnn->nFilters     < CNN_N_FLT_COEFF_MPLY, XNN_ERR_LYR_CFG);
//    Expect::False(cnn->nFilters     > CNN_N_FLT_MAX, XNN_ERR_LYR_CFG);
//    Expect::MultiplicityOf(cnn->nFilters, CNN_N_FLT_COEFF_MPLY);
//
//    Expect::False(cnn->nFilterRows > CNN_N_FLT_COEFF_MAX, XNN_ERR_LYR_CFG);
//    Expect::False(cnn->nFilterRows < 1, XNN_ERR_LYR_CFG);
//
//    uint32_t nFeatures = cnn->nFeatureMapRows * fltStrideSz;
//    Expect::False(nFeatures < CNN_N_FLT_COEFF_MIN, XNN_ERR_LYR_CFG);
//
//    Expect::True(lyr->ColumnCount == cnn->nFilters * ElementCount, XNN_ERR_LYR_CFG);
//
//    Expect::False(fltStrideSz       < 1, XNN_ERR_LYR_CFG);
//    Expect::False(fltStrideSz       > CNN_N_FLT_COEFF_MAX, XNN_ERR_LYR_CFG);
//    Expect::True(cnn->poolType < NUM_POOLING_TYPES, XNN_ERR_LYR_CFG);
//    if (INTEL_NO_POOLING != cnn->poolType)
//    {
//        Expect::False(cnn->nPoolSize    < CNN_POOL_SIZE_MIN, XNN_ERR_LYR_CFG);
//        Expect::False(cnn->nPoolSize    > CNN_POOL_SIZE_MAX, XNN_ERR_LYR_CFG);
//
//        Expect::False(cnn->nPoolStride  < CNN_POOL_SIZE_MIN, XNN_ERR_LYR_CFG);
//        Expect::False(cnn->nPoolStride  > CNN_POOL_SIZE_MAX, XNN_ERR_LYR_CFG); 
//    }
//
//    Expect::True(cnn->nBytesFilterCoefficient == 2, XNN_ERR_WEIGHT_BYTES);
//    Expect::True(cnn->nBytesBias == 4, XNN_ERR_BIAS_BYTES);
//}
