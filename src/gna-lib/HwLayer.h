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

#include "ActiveList.h"
#include "AffineLayers.h"
#include "ConvolutionalLayer.h"
#include "GmmLayer.h"
#include "Layer.h"
#include "RecurrentLayer.h"
#include "SimpleLayers.h"
#include "SwHw.h"

namespace GNA
{

/** CNN maximum number of filters per iteration */
#define CNN_N_FLT_ITER_MAX          16

/**
* Auxiliary Hardware Layer descriptor converter
*/
class HwLayer
{
public:
    /**
    * Creates HwLayer converter based on layer kind
    *
    * @kind        (in)    API layer kind
    * @return      layer specific converter instance
    */
    static HwLayer* create(NN_OP_TYPE kind);

    /**
    * Initializes converter
    *
    * @lyr         (in)    input API layer descriptor pointer
    * @hwLyr       (out)   converted HW single layer descriptor
    * @buffer      (in)    Accelerator memory buffer base address
    * @hwInBuffSize(in)    hw input buffer size in KB
    * @bLayer	   (in)	   common converter and validator
    */
    virtual void init(
        nn_layer*		lyr,
        XNN_LYR*        hwLyr,
        const void*     buffer,
        uint32_t        hwInBuffSize,
        Layer*		bLayerIn);

    /**
    * Converts API layer to hardware layer and stores to hwLyr
    */
    virtual void convert();

    /**
    * Converts API layer active list to hardware layer and stores to hwLyr
    *
    * @NOTE:   Convert must be called earlier
    * @al          (in)    active list parameters
    */
    void convertAL(ActiveList* al);

    /**
    * Empty virtual destructor enables derived classes destructor calls
    */
    virtual ~HwLayer() {};

protected:
    XNN_LYR*            hwLyr;      // converted HW single layer descriptor
    void*               buffer;     // Accelerator memory buffer base address
    Layer*			baseLayer;	// common sw/hw converter and validator
    /**
    * Number of data elements that may be stored in hw with 12KB buffer
    */
    const static uint32_t nBuffElems12K[8];

    /**
    * Number of data elements that may be stored in hw with 24KB buffer
    */
    const static uint32_t nBuffElems24K[8];

    /**
    * Effective number of data elements that may be stored in hw
    */
    uint32_t* nBuffElems;

    /**
    * Validates layer parameters
    */
    virtual void validate();

    /**
    * Stores hardware layer parameters
    */
    virtual void save();

    /**
    * Creates uninitialized converter
    */
    HwLayer() : hwLyr(nullptr), buffer(nullptr), baseLayer(nullptr), nBuffElems((uint32_t*)nBuffElems24K) {};
};

/**
* Extended Hardware Layer descriptor converter
*/
class HwLayerExt : public HwLayer
{
public:
    void convert() override;

    ~HwLayerExt() {};

    /**
     * Deleted functions to prevent from being defined or called
     * @see: https://msdn.microsoft.com/en-us/library/dn457344.aspx
     */
    HwLayerExt(const HwLayerExt &) = delete;
    HwLayerExt& operator=(const HwLayerExt&) = delete;

protected:
    uint32_t            nGr;        // grouping for iteration calculation
    uint32_t            nIters;     // number of iterations = data chunks/parts
    uint32_t            nLast;      // number of elements in last iteration

    /**
    * Calculates number of iterations and elements in last iteration
    *
    * @nGr #groups for calculation (can be different than network grouping)
    */
    void calcIterations(uint32_t nGr);

    void validate() override;

    void save() override;

    HwLayerExt() : HwLayer(), nGr(0), nIters(0), nLast(0), baseLayerExt(nullptr) {};

    void init(
        nn_layer*		lyr,
        XNN_LYR*        hwLyr,
        const void*     buffer,
        uint32_t        hwInBuffSize,
        Layer*		bLayerIn) override;

private:
    Layer* baseLayerExt;
};

/**
* Affine, Diagonal and transpose layers Layer descriptor converter
*/
class HwLayerAffDiagTrans : public HwLayerExt
{
public:
    void convert() override final;

    HwLayerAffDiagTrans() : HwLayerExt() {};

    virtual ~HwLayerAffDiagTrans() {};

    void init(
        nn_layer*		lyr,
        XNN_LYR*        hwLyr,
        const void*     buffer,
        uint32_t        hwInBuffSize,
        Layer*		bLayerIn) override;

protected:
    void save() override final;
};
//
///**
//* Hardware Copy Layer descriptor converter
//*/
//class HwLayerCopy : public HwLayer
//{
//public:
//    void convert() override final;
//
//    HwLayerCopy() : HwLayer(), copyLayer(nullptr) {};
//
//    /**
//     * Deleted functions to prevent from being defined or called
//     * @see: https://msdn.microsoft.com/en-us/library/dn457344.aspx
//     */
//    HwLayerCopy(const HwLayerCopy &) = delete;
//    HwLayerCopy& operator=(const HwLayerCopy&) = delete;
//
//    void init(
//        nn_layer*		lyr,
//        XNN_LYR*        hwLyr,
//        const void*     buffer,
//        uint32_t        hwInBuffSize,
//        Layer*		bLayerIn) override;
//
//    virtual ~HwLayerCopy() {};
//
//protected:
//    void validate() override final;
//
//    void save() override final;
//
//private:
//    CopyLayer* copyLayer;
//};
//
///**
//* Recurrent Layer descriptor converter
//*/
//class HwLayerRnn : public HwLayerExt
//{
//public:
//    void convert() override final;
//
//    HwLayerRnn() : HwLayerExt(),
//        nFbIters(0), nFbFirst(0), nFbLast(0), rnnLayer(nullptr) {};
//
//    /**
//     * Deleted functions to prevent from being defined or called
//     * @see: https://msdn.microsoft.com/en-us/library/dn457344.aspx
//     */
//    HwLayerRnn(const HwLayerRnn &) = delete;
//    HwLayerRnn& operator=(const HwLayerRnn&) = delete;
//
//    void init(
//        nn_layer*		lyr,
//        XNN_LYR*        hwLyr,
//        const void*     buffer,
//        uint32_t        hwInBuffSize,
//        Layer*		bLayerIn) override;
//
//    virtual ~HwLayerRnn() {};
//
//protected:
//    void validate() override final;
//
//    void save() override final;
//
//private:
//    uint32_t        nFbIters;       // number of iterations for feedback data
//    uint32_t        nFbFirst;       // number of el. in first feedback data iter.
//    uint32_t        nFbLast;        // number of el. in last feedback data iter.
//    RnnLayer*       rnnLayer;
//};
//
///**
//* Convolutional Layer descriptor converter
//*/
//class HwLayerCnn : public HwLayerExt
//{
//public:
//    void convert() override final;
//
//    HwLayerCnn() : HwLayerExt(),
//        nFltIters(0), nFltsLast(0), nFltsPerIter(0), fltBuffSz(0), fltBuffSzLast(0), cnnLayer(nullptr) {};
//
//    /**
//     * Deleted functions to prevent from being defined or called
//     * @see: https://msdn.microsoft.com/en-us/library/dn457344.aspx
//     */
//    HwLayerCnn(const HwLayerCnn &) = delete;
//    HwLayerCnn& operator=(const HwLayerCnn&) = delete;
//
//    void init(
//        nn_layer*		lyr,
//        XNN_LYR*        hwLyr,
//        const void*     buffer,
//        uint32_t        hwInBuffSize,
//        Layer*		bLayerIn) override;
//
//    virtual ~HwLayerCnn() {};
//
//protected:
//    void validate() override final;
//
//    void save() override final;
//
//private:
//    uint32_t        nFltIters;      // number of iterations  to process all flts.
//    uint32_t        nFltsLast;      // number of filters in last iter.
//    uint32_t        nFltsPerIter;   // number of filters in buffer in full iterations
//    uint32_t        fltBuffSz;      // size of filter in non-last iter. (elems)
//    uint32_t        fltBuffSzLast;  // size of filter in last iter. (elems)
//    CnnLayer*		cnnLayer;
//};
//
}
