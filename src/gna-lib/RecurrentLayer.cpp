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
//#include "RecurrentLayer.h"
//
//using namespace GNA;
//
//void RnnLayer::convert()
//{
//    BaseLayerExt::convert();
//
//    Validate::IsNull(lyr->pLayerStruct);
//    ElementCount = lyr->ColumnCount;      // FLAT input matrix
//    ElementCount = lyr->ColumnCount;     // - || -
//
//    rnn = (nn_layer_reccurent*)lyr->pLayerStruct;
//    aff = &rnn->affine;
//    pwl = (rnn->pwl.nSegments > 0) ? &rnn->pwl : nullptr;
//    validate();
//}
//
//void RnnLayer::validate()
//{
//    BaseLayerExt::validate();
//
//    // must be multiple 32 to keep 64B output buffer alignment
//    Validate::IsMultiplicityOf(ElementCount, RNN_N_OUT_ELEMS_MPLY);
//    Validate::IsTrue(ElementCount < RNN_N_OUT_ELEMS_MPLY, XNN_ERR_LYR_CFG);
//    Validate::IsTrue(ElementCount > XNN_N_IN_ELEMS_MAX, XNN_ERR_LYR_CFG);
//
//    Validate::IsNull(lyr->BufferIntermediate); // intermediate output buffer must be set always
//
//    Validate::IsTrue(nGroup != lyr->RowCount, XNN_ERR_LYR_CFG);
//    Validate::IsTrue(nGroup != lyr->RowCount, XNN_ERR_LYR_CFG);
//
//    int32_t  delay = 0;            // helper delay counter
//
//    Validate::IsNull(pwl); // RNN must have pwl enabled
//    Validate::IsTrue(NULL == rnn->pFeedbackBuffer, XNN_ERR_NO_FEEDBACK);
//    Validate::IsAlignedTo64(rnn->pFeedbackBuffer);
//    // pFeedbackBuffer K-delay offset check
//    delay = (int32_t)((uint8_t*)lyr->pOutputs - (uint8_t*)rnn->pFeedbackBuffer);
//    Validate::IsTrue(delay < 2 * ElementCount, XNN_ERR_NO_FEEDBACK); // FB must be before output buffer
//    Validate::IsMultiplicityOf(delay, 2 * ElementCount);
//    delay /= 2 * ElementCount; // delay in terms of FV
//    Validate::IsTrue(delay < 1, XNN_ERR_NO_FEEDBACK);
//    Validate::IsTrue(delay > XNN_N_GROUP_MAX, XNN_ERR_NO_FEEDBACK);
//}
