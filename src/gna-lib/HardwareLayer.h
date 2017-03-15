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

#include <map>

#include "ActiveList.h"
#include "AffineLayers.h"
#include "ConvolutionalLayer.h"
#include "GmmLayer.h"
#include "Layer.h"
#include "RecurrentLayer.h"
#include "SimpleLayers.h"
#include "SwHw.h"

using std::map;

namespace GNA
{

/** CNN maximum number of filters per iteration */
#define CNN_N_FLT_ITER_MAX          16

/**
* Auxiliary Hardware Layer descriptor converter
*/
class HardwareLayer
{
public:
    static const map<const nn_layer_kind, const NN_OP_TYPE> OperationsMap;

    /**
    * Creates HardwareLayer converter based on layer kind
    *
    * @type        (in)    API layer kind
    * @return      layer specific converter instance
    */
    static std::unique_ptr<HardwareLayer> Create(const Layer& softwareLayer, XNN_LYR *layerDescriptor, void *descriptorBuffer, uint32_t hwInBufferSize);

    /**
    * Empty virtual destructor enables derived classes destructor calls
    */
    virtual ~HardwareLayer() {};

protected:
    const Layer&        softwareLayer;
    XNN_LYR*            layerDescriptor;    // single layer descriptor
    void*               descriptorBuffer;   // hardware descriptor buffer

    /**
    * Number of data elements that may be stored in hw with 12KB buffer
    */
    const static uint32_t nBuffElems12K[8];

    /**
    * Number of data elements that may be stored in hw with 24KB buffer
    */
    const static uint32_t nBuffElems24K[8];

    /**
    * Converts API layer active list to hardware layer and stores to hwLyr
    *
    * @NOTE:   Convert must be called earlier
    * @al          (in)    active list parameters
    */
    void convertAL(ActiveList &activeList);

    /**
    * Effective number of data elements that may be stored in hw
    */
    const uint32_t* nBuffElems;

    virtual void convert() = 0;

    /**
    * Validates layer parameters
    */
    virtual void validate() = 0;

    /**
    * Stores hardware layer parameters
    */
    virtual void save();

    /**
    * Creates uninitialized converter
    */
    HardwareLayer(const Layer& swLayer, XNN_LYR *layerDesc, void *descBuffer, uint32_t hwInBufferSize);
};

/**
* Extended Hardware Layer descriptor converter
*/
class HardwareLayerExt : public HardwareLayer
{
public:
    ~HardwareLayerExt() {};

    /**
     * Deleted functions to prevent from being defined or called
     * @see: https://msdn.microsoft.com/en-us/library/dn457344.aspx
     */
    HardwareLayerExt(const HardwareLayerExt &) = delete;
    HardwareLayerExt& operator=(const HardwareLayerExt&) = delete;

protected:
    HardwareLayerExt(const Layer& swLayer, XNN_LYR *layerDesc, void *descBuffer, uint32_t hwInBuffSize);

    /**
    * Calculates number of iterations and elements in last iteration
    *
    * @nGr #groups for calculation (can be different than network grouping)
    */
    void calcIterations(uint32_t nGr);

    void validate() override;

    void save() override;

    uint32_t nGr;    // grouping for iteration calculation
    uint32_t nIters; // number of iterations = data chunks/parts
    uint32_t nLast;  // number of elements in last iteration
};

/**
* Affine, Diagonal and transpose layers Layer descriptor converter
*/
class HardwareLayerAffDiagTrans : public HardwareLayerExt
{
public:
    HardwareLayerAffDiagTrans(const Layer& swLayer, XNN_LYR *layerDesc, void *descBuffer, uint32_t hwInBuffSize);

    virtual ~HardwareLayerAffDiagTrans() {};

protected:
    void convert() override final;

private:
    const AffineLayer& affineLayer;
};

/**
* Hardware Copy Layer descriptor converter
*/
class HardwareLayerCopy : public HardwareLayer
{
public:
    HardwareLayerCopy(const Layer& swLayer, XNN_LYR *layerDesc, void *descBuffer, uint32_t hwInBuffSize);

    /**
     * Deleted functions to prevent from being defined or called
     * @see: https://msdn.microsoft.com/en-us/library/dn457344.aspx
     */
    HardwareLayerCopy(const HardwareLayerCopy &) = delete;
    HardwareLayerCopy& operator=(const HardwareLayerCopy&) = delete;

    virtual ~HardwareLayerCopy() {};

protected:
    void convert() override final;

    void validate() override final;

    void save() override final;

private:
    const CopyLayer& copyLayer;
};

///**
//* Recurrent Layer descriptor converter
//*/
//class HardwareLayerRnn : public HardwareLayerExt
//{
//public:
//    void convert() override final;
//
//    HardwareLayerRnn() : HardwareLayerExt(),
//        nFbIters(0), nFbFirst(0), nFbLast(0), rnnLayer(nullptr) {};
//
//    /**
//     * Deleted functions to prevent from being defined or called
//     * @see: https://msdn.microsoft.com/en-us/library/dn457344.aspx
//     */
//    HardwareLayerRnn(const HardwareLayerRnn &) = delete;
//    HardwareLayerRnn& operator=(const HardwareLayerRnn&) = delete;
//
//    void init(
//        nn_layer*		lyr,
//        XNN_LYR*        hwLyr,
//        const void*     buffer,
//        uint32_t        hwInBuffSize,
//        Layer*		bLayerIn) override;
//
//    virtual ~HardwareLayerRnn() {};
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
//class HardwareLayerCnn : public HardwareLayerExt
//{
//public:
//    void convert() override final;
//
//    HardwareLayerCnn() : HardwareLayerExt(),
//        nFltIters(0), nFltsLast(0), nFltsPerIter(0), fltBuffSz(0), fltBuffSzLast(0), cnnLayer(nullptr) {};
//
//    /**
//     * Deleted functions to prevent from being defined or called
//     * @see: https://msdn.microsoft.com/en-us/library/dn457344.aspx
//     */
//    HardwareLayerCnn(const HardwareLayerCnn &) = delete;
//    HardwareLayerCnn& operator=(const HardwareLayerCnn&) = delete;
//
//    void init(
//        nn_layer*		lyr,
//        XNN_LYR*        hwLyr,
//        const void*     buffer,
//        uint32_t        hwInBuffSize,
//        Layer*		bLayerIn) override;
//
//    virtual ~HardwareLayerCnn() {};
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
