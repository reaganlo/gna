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

#include <cstring>
#include <cstdlib>
#include <iostream>

#include "gna-api.h"

#include "SetupGmmModel.h"

SetupGmmModel::SetupGmmModel(DeviceController & deviceCtrl, bool activeListEn)
    : deviceController{deviceCtrl},
      activeListEnabled{activeListEn}
{
    sampleGmmLayer(nnet);

    deviceController.ModelCreate(&nnet, &modelId);

    configId = deviceController.ConfigAdd(modelId);

    deviceController.BufferAdd(configId, GNA_IN, 0, inputBuffer);
    deviceController.BufferAdd(configId, GNA_OUT, 0, outputBuffer);

    if (activeListEnabled)
    {
        deviceController.ActiveListAdd(configId, 0, indicesCount, indices);
    }
}

SetupGmmModel::~SetupGmmModel()
{
    deviceController.Free();

    free(nnet.pLayers->pLayerStruct);
    free(nnet.pLayers);
}

void SetupGmmModel::checkReferenceOutput(int modelIndex, int configIndex) const
{
    unsigned int ref_output_size = refSize[configIndex];
    const int32_t* ref_output = refOutputAssign[configIndex];
    for (unsigned int i = 0; i < ref_output_size; ++i)
    {
        int32_t outElemVal = static_cast<const int32_t*>(outputBuffer)[i];
        if (ref_output[i] != outElemVal)
        {
            // TODO: how it should notified? return or throw
            throw std::runtime_error("Wrong output");
        }
    }
}


void SetupGmmModel::sampleGmmLayer(intel_nnet_type_t& hNnet)
{
    hNnet.nGroup = groupingNum;
    hNnet.nLayers = layersNum;
    hNnet.pLayers = (intel_nnet_layer_t*)calloc(hNnet.nLayers, sizeof(intel_nnet_layer_t));

    uint32_t stateCount = 8;
    const int elementSize = sizeof(int32_t);

    int buf_size_weights = ALIGN64(sizeof(variance)); // note that buffer alignment to 64-bytes is required by GNA HW
    int buf_size_inputs = ALIGN64(sizeof(feature_vector));
    int buf_size_biases = ALIGN64(sizeof(Gconst));
    int buf_size_outputs = ALIGN64(outVecSz * elementSize);
    int buf_size_tmp_outputs = ALIGN64(outVecSz * elementSize);

                                                              // prepare params for GNAAlloc
    uint32_t bytes_requested = buf_size_weights + buf_size_inputs + buf_size_biases + buf_size_outputs + buf_size_tmp_outputs;
    if (activeListEnabled)
    {
        indicesCount = stateCount / 2;
        uint32_t buf_size_indices = indicesCount * sizeof(uint32_t);
        bytes_requested += buf_size_indices;
    }
    uint32_t bytes_granted;

    // call GNAAlloc (obtains pinned memory shared with the device)
    uint8_t* pinned_mem_ptr = deviceController.Alloc(bytes_requested, hNnet.nLayers, 1, &bytes_granted);

    int16_t *pinned_weights = (int16_t*)pinned_mem_ptr;
    memcpy(pinned_weights, variance, sizeof(variance));   // puts the weights into the pinned memory
    pinned_mem_ptr += buf_size_weights;                 // fast-forwards current pinned memory pointer to the next free block

    inputBuffer = (int16_t*)pinned_mem_ptr;
    memcpy(inputBuffer, feature_vector, sizeof(feature_vector));      // puts the inputs into the pinned memory
    pinned_mem_ptr += buf_size_inputs;                  // fast-forwards current pinned memory pointer to the next free block

    int32_t *pinned_biases = (int32_t*)pinned_mem_ptr;
    memcpy(pinned_biases, Gconst, sizeof(Gconst));      // puts the biases into the pinned memory
    pinned_mem_ptr += buf_size_biases;                  // fast-forwards current pinned memory pointer to the next free block

    outputBuffer = ((int16_t*)pinned_mem_ptr) + 32;
    pinned_mem_ptr += buf_size_outputs;                 // fast-forwards the current pinned memory pointer by the space needed for outputs

    pinned_mem_ptr += buf_size_tmp_outputs;

    if (activeListEnabled)
    {
        size_t indicesSize = indicesCount * sizeof(uint32_t);
        indices = (uint32_t*)pinned_mem_ptr;
        memcpy(indices, alIndices, indicesSize);
        pinned_mem_ptr += indicesSize;
    }

    gna_gmm_layer *gmm = (gna_gmm_layer*)calloc(1, sizeof(gna_gmm_layer));
    gmm->data.gaussianConstants = (uint32_t*)pinned_biases;
    gmm->data.inverseCovariancesForMaxMix16 = (uint16_t*)pinned_weights;
    gmm->data.meanValues = (uint8_t*)pinned_weights;
    gmm->config.layout = GMM_LAYOUT_FLAT;
    gmm->config.maximumScore = UINT32_MAX;
    gmm->config.mixtureComponentCount = 1;
    //gmm->config.mode = GNA_MAXMIX16;
    gmm->config.mode = GNA_MAXMIX8;
    gmm->config.stateCount = stateCount;

    hNnet.pLayers[0].nInputColumns = inVecSz;
    hNnet.pLayers[0].nInputRows = hNnet.nGroup;
    hNnet.pLayers[0].nOutputColumns = outVecSz;
    hNnet.pLayers[0].nOutputRows = hNnet.nGroup;
    hNnet.pLayers[0].nBytesPerInput = 1;
    hNnet.pLayers[0].nBytesPerOutput = 4;             // 4 bytes since we are not using PWL (would be 2 bytes otherwise)
    hNnet.pLayers[0].nBytesPerIntermediateOutput = 4; // this is always 4 bytes
    hNnet.pLayers[0].type = INTEL_INPUT_OUTPUT;
    hNnet.pLayers[0].nLayerKind = INTEL_GMM;
    hNnet.pLayers[0].pLayerStruct = gmm;
    hNnet.pLayers[0].pInputs = nullptr;
    hNnet.pLayers[0].pOutputsIntermediate = nullptr;
    hNnet.pLayers[0].pOutputs = nullptr;
}
