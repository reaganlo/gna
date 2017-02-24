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

#include "SwHw.h"
#include "HwLayer.h"

using namespace GNA;

Sw::Sw()
{
}

const std::map<const gna_gmm_mode, const GMM_MODE_CTRL> Hw::GmmModes = {
    //{ gna_gmm_mode, { read_elimination, calculation_mode, __res_03} },
    { GNA_MAXMIX8, {  0,  0,  0 } },
    { GNA_MAXMIX16,{ 0,  0,  0 } },
    { GNA_LINF, { 0,  2,  0 } },
    { GNA_L1, { 0,  1,  0 } },
    { GNA_L2, { 0,  0,  0 } },
};

Hw::Hw(const void*   baseIn, uint32_t inBuffSizeIn) :
    base(baseIn),
    gmmLayerDescriptors((GMM_CONFIG*)(((uint8_t*)baseIn) + (XNN_LAYERS_MAX_COUNT * sizeof(XNN_LYR)))),// TODO:KJ: support variable size models
    inData(nullptr),
    xnnLayerDescriptors(nullptr),
    dataSize(0),
    inBuffSize(inBuffSizeIn)
{
    memset(&io_handle, 0, sizeof(io_handle_t));
}

Hw::~Hw()
{
#ifdef _WIN32
    if (nullptr != io_handle.hEvent) CloseHandle(io_handle.hEvent);
#endif
    if (inData)
    {
        free(inData);
    }
}

// TODO:KJ:move to converter
void Hw::GmmLayerDescriptorSetup(const GmmLayer *layer, GMM_CONFIG* descriptor)
{
    // can be updated per request
    descriptor->fvaddr      = getAddrOffset(layer->Input.Buffer);
    descriptor->gmmscradd   = getAddrOffset(layer->Output.Buffer);

    // GMM Model configuration, will be constant over time for model
    descriptor->gmmscrlen   = GMM_SCORE_SIZE * layer->Input.VectorCount * layer->Config.stateCount;; // will be updated when ActiveList is used
    descriptor->fvoffset    = layer->Input.ElementCount;
    
    descriptor->numfv       = layer->Input.VectorCount;
    descriptor->vlength     = layer->Input.ElementCount;

    descriptor->mode        = GmmModes.at(layer->Config.mode);
    descriptor->gcaddr      = getAddrOffset(layer->Data.gaussianConstants);
    descriptor->mvaddr      = getAddrOffset(layer->Data.meanValues);
    descriptor->vvaddr      = getAddrOffset(layer->Data.inverseCovariancesForMaxMix16);

    descriptor->gcsoffset   = layer->Params.GaussConstSetOffsetSize;
    descriptor->mvsoffset   = layer->Params.MeanSetOffsetSize;
    descriptor->vvsoffset   = layer->Params.VarSetOffsetSize;
    descriptor->vvwidth     = layer->Params.VarianceSize;
    descriptor->gmmtelst    = layer->Config.mixtureComponentCount * layer->Input.ElementCount;
    descriptor->maxlsscore  = layer->Config.maximumScore;
    descriptor->numgmms     = layer->Config.stateCount;
    descriptor->nummcpg     = layer->Config.mixtureComponentCount;

    descriptor->fvwidth     = GMM_FV_ELEMENT_SIZE;
    descriptor->gcwidth     = GMM_CONSTANTS_SIZE;
    descriptor->gmmscrwdth  = GMM_SCORE_SIZE;
    descriptor->maxlswidth  = GMM_SCORE_SIZE;
    descriptor->mvwidth     = GMM_MEAN_VALUE_SIZE;
    // other fields left zeroed intentionally  
}

// TODO:KJ:move to converter
void GNA::Hw::GmmLayerDescriptorUpdateRequestConfig(const uint32_t layerIndex, const GmmLayer *gmm, 
    const RequestConfiguration &configuration, GMM_CONFIG* descriptor)
{
    // TODO:KJ: consider caching reusable RequestConfigurations data after conversion to reduce latency of update during scoring
    // this part of code will be run in library but updated descriptor parts will be written to driver and driver will handle
    // actual descriptors update per request
    // for each request we will send update list with modified layers
    //  for each modified layer we will send updated parts (or whole) of XNN Descriptor and GMM Descriptor it applicable

    // always update configuration if present in RequestConfiguration for given layer

    //if (layerIndex == configuration.inputBuffer.layerIndex)
    //{
    //    GmmLayerDescriptorUpdateInput(configuration.inputBuffer, descriptor);
    //}
    //if (layerIndex == configuration.outputBuffer.layerIndex)
    //{
    //    GmmLayerDescriptorUpdateOutput(configuration.outputBuffer, descriptor);
    //}
    //if (layerIndex == configuration.activeList->layerIndex)
    //{
    //    // update active list configuration even if it is disabled to revert possible enabling
    //    GmmLayerDescriptorUpdateActiveList(gmm, *configuration.activeList.get(), descriptor);
    //}
}

void Hw::GmmLayerDescriptorUpdateInput(const ConfigurationBuffer &inputBuffer, GMM_CONFIG* descriptor)
{
    descriptor->fvaddr = getAddrOffset(inputBuffer.address);
}

void Hw::GmmLayerDescriptorUpdateOutput(const ConfigurationBuffer &outputBuffer, GMM_CONFIG* descriptor)
{
    descriptor->gmmscradd = getAddrOffset(outputBuffer.address);
}

void Hw::GmmLayerDescriptorUpdateActiveList(const GmmLayer *gmm, const ActiveList &activeList, GMM_CONFIG* descriptor)
{
    // Active list setup
    uint32_t scoreElementsCount = GMM_SCORE_SIZE * gmm->Input.VectorCount * gmm->Config.stateCount;
    uint32_t activeListIndices = 0;
    uint32_t activeListIndicesCount = 0;
    if (true == activeList.enabled)
    {
        scoreElementsCount = GMM_SCORE_SIZE * gmm->Input.VectorCount * activeList.indicesCount;
        activeListIndices = getAddrOffset(activeList.indices);
        activeListIndicesCount = activeList.indicesCount;
    }
    inData->ctrlFlags.activeListOn = static_cast<uint32_t>(activeList.enabled);
    descriptor->gmmscrlen = scoreElementsCount;
    descriptor->asladdr = activeListIndices;
    descriptor->astlistlen = activeListIndicesCount;
}

void Hw::Fill(SoftwareModel* model)
{
    uint32_t i = 0;
    HwLayer *hwLayer = nullptr;     // hw layer converter
    uint32_t xnnSize = 0;
    xnnSize = XNN_LYR_DSC_SIZE * model->layerCount;
    dataSize= REQUEST_SIZE;
    init();

    // fill data structure that will be sent to kernel
    inData->ctrlFlags.gnaMode       = 1; // GNA2 default mode of operation
    //inData->ctrlFlags.activeListOn  = (uint32_t)model->activeList.enabled; // TODO: RequestConfig handler for ioctls
    inData->ctrlFlags.layerCount = model->layerCount;
    inData->ctrlFlags.layerNo = 0;
    inData->modelId = 0;

    // initial layer descriptor at the beginning of model buffer
    xnnLayerDescriptors = (XNN_LYR*) base;
    
    // set descriptor for all layers
    for (i = 0; i < model->layerCount; i++)
    {
        try
        {
            hwLayer = HwLayer::create(model->Layers[i]->Config.Operation);
            hwLayer->init(const_cast<nn_layer*>(&model->Layers[i]->sourceLayer), &xnnLayerDescriptors[i], base,
                inBuffSize, const_cast<Layer*>(model->Layers[i].get()));
            hwLayer->convert();

            // TODO:KJ: add GmmHwLayer and handle XNN layer common part of GMM layer
            if (INTEL_GMM == model->Layers[i]->Config.Type)
            {
                GmmLayerDescriptorSetup(static_cast<const GmmLayer*>(model->Layers[i].get()), &gmmLayerDescriptors[i]);
            }
            
            
           /* if (i == model->layerCount - 1)
            {
                hwLayer->convertAL(&model->activeList);
            }*/
            delete hwLayer;
            hwLayer = nullptr;
        }
        catch (GnaException& e)
        {
            if (hwLayer) delete hwLayer;
            ERR("Layer descriptor conversion error: LYR[%u]: %s\n", i, GnaStatusToString(e.getStatus()));
            throw e;
        }  
    }
}

void Hw::init()
{
#ifdef _WIN32
    io_handle.hEvent = CreateEvent(nullptr, false, false, nullptr);
    Validate::IsTrue(nullptr == io_handle.hEvent, GNA_ERR_RESOURCES);
#endif
    inData = (hw_calc_in_t*)calloc(1, sizeof(hw_calc_in_t));
    Validate::IsTrue(nullptr == inData, GNA_ERR_RESOURCES);
    inData->status = GNA_NULLARGNOTALLOWED;
    
}
