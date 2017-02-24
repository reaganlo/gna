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
//#pragma once
//
//#include "Layer.h"
//
//namespace GNA
//{
//
//// Transpose Layer descriptor converter
//class TransposeLayer : public BaseLayerExt
//{
//public:
//    friend class HwLayerAffDiag;
//    void convert() override;
//
//    TransposeLayer(NN_OP_TYPE hwLyrKind) : BaseLayerExt(hwLyrKind) {};
//
//    virtual ~TransposeLayer() {};
//
//protected:
//    void validate() override;
//
//};
//
//// Copy Layer descriptor converter
//class CopyLayer : public Layer
//{
//public:
//    friend class HwLayerCopy;
//    void convert() override;
//
//    CopyLayer() : Layer(NN_COPY), cpy(nullptr) {};
//
//    virtual ~CopyLayer() {};
//
//    /**
//     * Deleted functions to prevent from being defined or called
//     * @see: https://msdn.microsoft.com/en-us/library/dn457344.aspx
//     */
//    CopyLayer(const CopyLayer &) = delete;
//    CopyLayer& operator=(const CopyLayer&) = delete;
//
//protected:
//    void validate() override;
//
//private:
//    nn_layer_copy*   cpy;            // copy layer details
//};
//
//}
