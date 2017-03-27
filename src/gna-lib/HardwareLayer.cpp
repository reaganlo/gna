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

using std::make_unique;
using std::map;
using std::unique_ptr;

using namespace GNA;

const map<const nn_layer_kind, const NN_OP_TYPE> HardwareLayer::OperationsMap =
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

XNN_LYR HardwareLayer::layerDescriptor;

XNN_LYR HardwareLayer::Convert(const Layer& softwareLayer, void * const memoryBase, 
    const uint32_t hardwareInternalBufferSize)
{
    auto converter = unique_ptr<HardwareLayer>();
    switch (OperationsMap.at(softwareLayer.Config.Kind))
    {
    case NN_CNN:
        converter = make_unique<HardwareLayerCnn>(softwareLayer, memoryBase, hardwareInternalBufferSize);
        break;
    case NN_COPY:
        converter = make_unique<HardwareLayerCopy>(softwareLayer, memoryBase);
        break;
    //case NN_GMM:
        //converter = make_unique<HwGmmLayer>();
        break;
    case NN_RNN:
        converter = make_unique<HardwareLayerRnn>(softwareLayer, memoryBase, hardwareInternalBufferSize);
        break;
    default:
        converter = make_unique<HardwareLayerAffDiagTrans>(softwareLayer, memoryBase, hardwareInternalBufferSize);
        break;
    }
    return converter->layerDescriptor;
}

HardwareLayer::HardwareLayer(const Layer &swLayer, void * const memoryBase) :
    softwareLayer(swLayer),
    memoryBaseAddress(memoryBase)
{
    layerDescriptor = {};
    Expect::ValidBuffer(memoryBaseAddress);
}

void HardwareLayer::convertAL(ActiveList& activeList)
{
    if (activeList.Enabled)
    {
        layerDescriptor.act_list_n_elems = static_cast<uint16_t>(activeList.IndicesCount);
        layerDescriptor.act_list_buffer = getOffset(activeList.Indices);
        if (NN_AFFINE == layerDescriptor.op)
        {
            layerDescriptor.op = NN_AFF_AL;
        }
        else if (NN_GMM == layerDescriptor.op)
        {
            layerDescriptor.op = NN_GMM_ACTIVE_LIST;
        }
    }
}

void HardwareLayer::save()
{
    layerDescriptor.op = OperationsMap.at(softwareLayer.Config.Kind);
    layerDescriptor.n_in_elems = static_cast<uint16_t>(softwareLayer.Input.ElementCount);
    layerDescriptor.n_out_elems = static_cast<uint16_t>(softwareLayer.Output.ElementCount);
    layerDescriptor.n_groups = static_cast<uint8_t>(softwareLayer.Input.VectorCount);
    if (INTEL_GMM == softwareLayer.Config.Type)
    {
        layerDescriptor.gmm_descriptor = 0; // TODO:KJ: set actual gmm descriptor address
    }
    // TODO:KJ: add I/O configuration conversion to Request config
    else
    {
        layerDescriptor.in_buffer = getOffset(softwareLayer.Input.Buffer);
    }
    layerDescriptor.out_act_fn_buffer = getOffset(softwareLayer.Output.Buffer);
    layerDescriptor.out_sum_buffer = getOffset(softwareLayer.Output.ScratchPad);
}

const map<const uint32_t, std::array<const uint32_t, XNN_N_GROUP_MAX>> HardwareLayerExt::bufferElementsMap
{
    { 12,{ 12288, 12288, 12096, 12288, 12000, 12096, 12096, 12288 } },
    { 24,{ 6144, 6144, 6048, 6144, 5760, 6048, 6048, 6144 } }
};

HardwareLayerExt::HardwareLayerExt(const Layer& swLayer, void * const memoryBase,
    const uint32_t bufferSize, const uint32_t effectiveGrouping) :
    HardwareLayer(swLayer, memoryBase),
    iterationGrouping(effectiveGrouping),
    bufferElementCount(bufferElementsMap.at(bufferSize).at(effectiveGrouping - 1))
{
    Expect::InRange(iterationGrouping, 1, XNN_N_GROUP_MAX, XNN_ERR_GROUPING);
    // Calculates number of iterations and elements in last iteration
     //#groups for calculation(can be different than network grouping)
    auto elementsTimesGrouping = softwareLayer.Input.ElementCount * iterationGrouping;
    iterationCount = ((elementsTimesGrouping - 1) / bufferElementCount) + 1;
    Expect::InRange(iterationCount, 1, UINT8_MAX, XNN_ERR_LYR_CFG);

    lastIterationElementCount = ((elementsTimesGrouping) - ((iterationCount - 1) * bufferElementCount)) / iterationGrouping;
    Expect::InRange(lastIterationElementCount, 1, bufferElementCount, XNN_ERR_LYR_CFG);
    Expect::MultiplicityOf(lastIterationElementCount, XNN_N_IN_ELEMS_MPLY);
}

void HardwareLayerExt::save()
{
    HardwareLayer::save();
    layerDescriptor.n_iters = static_cast<uint8_t>(iterationCount);
    layerDescriptor.n_elems_last = static_cast<uint16_t>(lastIterationElementCount);

    if (affine)
    {
        layerDescriptor.flags.weight_size = (affine->GetWeightMode() == GNA_WEIGHT_2B) ? 0 : 1;
        layerDescriptor.aff_weight_buffer = getOffset(affine->GetWeights());
        layerDescriptor.aff_const_buffer = getOffset(affine->GetBiases());
    }
    if (activation && activation->Enabled)
    {
        layerDescriptor.flags.act_fn_en = 1;
        layerDescriptor.pwl_n_segs = static_cast<uint8_t>(activation->SegmentCount);
        layerDescriptor.pwl_seg_def_buffer = getOffset(activation->Segments);
    }
    else
    {
        layerDescriptor.flags.act_fn_en = 0;
    }
}

HardwareLayerAffDiagTrans::HardwareLayerAffDiagTrans(const Layer& swLayer,
    void * const memoryBase, uint32_t hwInBuffSize) :
    HardwareLayerExt(swLayer, memoryBase, hwInBuffSize, softwareLayer.Input.VectorCount)
{
    switch (softwareLayer.Config.Kind)
    {
    case INTEL_AFFINE:
    case INTEL_AFFINE_DIAGONAL:
    case INTEL_AFFINE_MULTIBIAS:
        auto& aff = static_cast<const AffineLayer&>(softwareLayer);
        affine = aff.Affine.get();
        activation = aff.Activation.get();
        break;
    }
    save();
}

HardwareLayerCopy::HardwareLayerCopy(const Layer& swLayer, void * const memoryBase) :
    HardwareLayer(swLayer, memoryBase)
{
    save();
}

void HardwareLayerCopy::save()
{
    HardwareLayer::save();
    auto& copy = static_cast<const CopyLayer&>(softwareLayer);
    layerDescriptor.cpy_n_elems = static_cast<uint16_t>(copy.CopyElementsCount);
}

HardwareLayerRnn::HardwareLayerRnn(const Layer& swLayer, void * const memoryBase, const uint32_t hwInBuffSize) :
    HardwareLayerExt(swLayer, memoryBase, hwInBuffSize, 1),
    feedbackIterationsCount(0),
    feedbackFirstIterElementCount(0),
    feedbackLastIterElementCount(0)
{
    auto& rnn = static_cast<const RnnLayer&>(softwareLayer);
    affine = rnn.Affine.get();
    activation = rnn.Activation.get();
    convert();
    save();
};

void HardwareLayerRnn::convert()
{
    auto elementCount = softwareLayer.Input.ElementCount;

    feedbackFirstIterElementCount = min((bufferElementCount - lastIterationElementCount), elementCount);
    Expect::True(feedbackFirstIterElementCount <= bufferElementCount, XNN_ERR_LYR_CFG);

    feedbackIterationsCount = (elementCount - feedbackFirstIterElementCount) / bufferElementCount;
    if ((elementCount - feedbackFirstIterElementCount) % bufferElementCount)
    {
        feedbackIterationsCount++;
    }
    if (feedbackFirstIterElementCount > 0)
    {
        feedbackIterationsCount++;
    }
    Expect::InRange(feedbackIterationsCount, 1, UINT8_MAX, XNN_ERR_LYR_CFG);

    if (feedbackFirstIterElementCount && 1 == feedbackIterationsCount)
    {
        feedbackLastIterElementCount = feedbackFirstIterElementCount;
    }
    else if (0 == feedbackFirstIterElementCount)
    {
        feedbackLastIterElementCount = 
            elementCount - feedbackFirstIterElementCount - (feedbackIterationsCount - 1) * bufferElementCount;
    }
    else
    {
        feedbackLastIterElementCount = 
            elementCount - feedbackFirstIterElementCount - (feedbackIterationsCount - 2) * bufferElementCount;
    }
    Expect::InRange(feedbackLastIterElementCount, 1, bufferElementCount, XNN_ERR_LYR_CFG);
}

void HardwareLayerRnn::save()
{
    HardwareLayerExt::save();
    layerDescriptor.rnn_n_fb_iters = feedbackIterationsCount;
    layerDescriptor.rnn_n_elems_first = feedbackFirstIterElementCount;
    layerDescriptor.rnn_n_elems_last = feedbackLastIterElementCount;
    // will be 0 for hidden layers
    if (INTEL_INPUT == softwareLayer.Config.Type || INTEL_HIDDEN == softwareLayer.Config.Type)
    {
        layerDescriptor.rnn_out_fb_buffer = CalculateFeedbackBuffer(softwareLayer.Output.Buffer);
    }
}

const uint32_t HardwareLayerRnn::CalculateFeedbackBuffer(const void * const outputBuffer) const
{
    auto& rnn = static_cast<const RnnLayer&>(softwareLayer);
    return getOffset(rnn.CalculateFeedbackBuffer(outputBuffer));
}

HardwareLayerCnn::HardwareLayerCnn(const Layer & swLayer, void * const memoryBase, uint32_t hwInBuffSize) :
    HardwareLayerExt(swLayer, memoryBase, hwInBuffSize, 1)
{
    auto& cnn = static_cast<const CnnLayer&>(softwareLayer);
    auto fitlerCount = cnn.Convolution.Filters.Count;
    auto fitlerSize = cnn.Convolution.Filters.CoefficientCount;
    filtersCountInFullIteration =
        min(
            fitlerCount,
            (fitlerSize <= bufferElementCount / 6 / 3) ?
                16 :
                (fitlerSize <= bufferElementCount / 6 / 2) ?
                    12 :
                    (fitlerSize <= bufferElementCount / 6) ?
                        4 :
                        0);
    Expect::InRange(filtersCountInFullIteration, CNN_N_FLT_COEFF_MPLY, CNN_N_FLT_ITER_MAX, XNN_ERR_LYR_CFG);
    Expect::MultiplicityOf(filtersCountInFullIteration, CNN_N_FLT_COEFF_MPLY);

    filtersIterationCount = (fitlerCount - 1) / filtersCountInFullIteration + 1;

    filtersCountInLastIteration = fitlerCount - ((filtersIterationCount - 1) * filtersCountInFullIteration);
    Expect::InRange(filtersCountInLastIteration, CNN_N_FLT_COEFF_MPLY, CNN_N_FLT_ITER_MAX, XNN_ERR_LYR_CFG);
    Expect::MultiplicityOf(filtersCountInLastIteration, CNN_N_FLT_COEFF_MPLY);


    filtersElementCountInFullIteration = filtersCountInFullIteration * fitlerSize;
    Expect::InRange(filtersElementCountInFullIteration, 1, bufferElementCount, XNN_ERR_LYR_CFG);

    filtersElementCountInLastIteration = filtersCountInLastIteration * fitlerSize;
    Expect::InRange(filtersElementCountInLastIteration, 1, bufferElementCount, XNN_ERR_LYR_CFG);

    save();
}

void HardwareLayerCnn::save()
{
    HardwareLayerExt::save();
    // some fields saved by HardwareLayerExt will be overwritten
    auto& cnn = static_cast<const CnnLayer&>(softwareLayer);
    layerDescriptor.flags.pool_param = static_cast<uint8_t>(cnn.Pooling.Type);
    layerDescriptor.cnn_flt_bf_sz_iter = filtersElementCountInFullIteration;
    layerDescriptor.cnn_flt_bf_sz_last = filtersElementCountInLastIteration;
    layerDescriptor.cnn_flt_buffer = getOffset(cnn.Convolution.Filters.Data);;
    layerDescriptor.cnn_flt_size = cnn.Convolution.Filters.CoefficientCount;
    layerDescriptor.cnn_n_flts = cnn.Convolution.Filters.Count;
    layerDescriptor.cnn_n_flts_iter = filtersCountInFullIteration;
    layerDescriptor.cnn_n_flt_iters = filtersIterationCount;
    layerDescriptor.cnn_n_flt_last = filtersCountInLastIteration;
    layerDescriptor.cnn_n_flt_outs = cnn.Convolution.OutputElementsCount;
    layerDescriptor.cnn_n_flt_stride = cnn.Pooling.Stride;
    layerDescriptor.cnn_n_out_p_flt = softwareLayer.Output.ElementCount;
    layerDescriptor.cnn_pool_size = cnn.Pooling.Size;
    layerDescriptor.cnn_pool_stride = cnn.Pooling.Stride;
    layerDescriptor.aff_const_buffer = getOffset(cnn.Convolution.Filters.Biases);
}