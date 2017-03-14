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

#include "SoftwareModel.h"

#include "AffineLayers.h"
#include "Validator.h"

using std::make_unique;

using namespace GNA;

SoftwareModel::SoftwareModel(const gna_model* network)
    :layerCount(network->nLayers),
    inputVectorCount(network->nGroup)
{
#ifndef NO_ERRCHECK
    Expect::InRange(inputVectorCount, 1, XNN_N_GROUP_MAX, XNN_ERR_LYR_CFG);
    Expect::InRange(layerCount, 1, XNN_LAYERS_MAX_COUNT, XNN_ERR_NET_LYR_NO);
    Expect::NotNull(network->pLayers);
#endif
    build(network->pLayers);
}

void SoftwareModel::ValidateConfiguration(const RequestConfiguration& configuration)
{
    Expect::True(inputLayerCount == configuration.InputBuffersCount, XNN_ERR_NETWORK_INPUTS);
    Expect::True(outputLayerCount == configuration.OutputBuffersCount, XNN_ERR_NETWORK_OUTPUTS);
}

void SoftwareModel::build(const nn_layer* layers)
{
    try
    {
        for (int i = 0; i < layerCount; i++)
        {
            auto layer = layers + i;
            switch (layer->type)
            {
            case INTEL_INPUT:
                ++inputLayerCount;
                break;
            case INTEL_OUTPUT:
                ++outputLayerCount;
                break;
            case INTEL_INPUT_OUTPUT:
                ++inputLayerCount;
                ++outputLayerCount;
                break;
            case INTEL_HIDDEN:
                break;
            default:
                throw GnaException(XNN_ERR_LYR_TYPE);
            }
            Layers.push_back(Layer::Create(const_cast<const nn_layer*>(layer), inputVectorCount));
        }
    }
    catch (const GnaException& e)
    {
        // re-throws internal exceptions
        throw e;
    }
    catch (...)
    {
        // catch null dereference // TODO:review
        throw GnaException(GNA_NULLARGNOTALLOWED);
    }

}