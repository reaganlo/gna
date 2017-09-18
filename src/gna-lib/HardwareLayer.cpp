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

#include "ActiveList.h"
#include "ConvolutionalLayer.h"
#include "GmmLayer.h"
#include "LayerConfiguration.h"
#include "RecurrentLayer.h"
#include "RequestConfiguration.h"
#include "SimpleLayers.h"
#include "Validator.h"

using std::array;
using std::make_unique;
using std::map;
using std::unique_ptr;

using namespace GNA;

DescriptorParameters::DescriptorParameters(const Layer* softwareLayer, const BaseAddressC& memoryBase,
    const AddrXnnLyr& xnnDescriptor, const AddrGmmCfgC& gmmDescriptor, const uint32_t hardwareInternalBufferSize) :
        SoftwareLayer{softwareLayer},
        MemoryBase{memoryBase},
        XnnDescriptor{xnnDescriptor},
        GmmDescriptor{gmmDescriptor},
        HardwareInternalBufferSize{hardwareInternalBufferSize}
{
    Expect::ValidBuffer(MemoryBase);
    Expect::ValidBuffer(XnnDescriptor);
    Expect::AlignedTo(XnnDescriptor, sizeof(XNN_LYR));
    if (INTEL_GMM == softwareLayer->Config.Kind)
    {
        Expect::ValidBuffer(GmmDescriptor);
        Expect::AlignedTo(GmmDescriptor, sizeof(GMM_CONFIG));
    }
};

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

unique_ptr<HardwareLayer> HardwareLayer::Create(const DescriptorParameters& parameters)
{
    switch (OperationsMap.at(parameters.SoftwareLayer->Config.Kind))
    {
    case NN_CNN:
        return make_unique<HardwareLayerCnn>(parameters);
    case NN_COPY:
        return make_unique<HardwareLayerCopy>(parameters);
    case NN_GMM:
        return make_unique<HardwareLayerGmm>(parameters);
    case NN_RNN:
        return make_unique<HardwareLayerRnn>(parameters);
    case NN_AFF_MB:
        return make_unique<HardwareLayerAffineMBias>(parameters);
    default:
        return make_unique<HardwareLayerAffDiagTrans>(parameters);
    }
}

HardwareLayer::HardwareLayer(const DescriptorParameters& parameters) :
    DescriptorParameters{parameters}
{
}

void HardwareLayer::WriteInputBuffer(PGNA_BUFFER_DESCR& lyrsCfg, const ConfigurationBuffer * const buffer) const
{
    lyrsCfg->offset = getOffset(XnnDescriptor) + offsetof(XNN_LYR, in_buffer);
    lyrsCfg->value = getOffset(buffer->Get());

    ++lyrsCfg;
}

void HardwareLayer::WriteOutputBuffer(PGNA_BUFFER_DESCR& lyrsCfg, const ConfigurationBuffer * const buffer) const
{
    if (LayerOutput::ActivatedOutput == SoftwareLayer->Output.GetOutputMode()
        || INTEL_CONVOLUTIONAL == SoftwareLayer->Config.Kind
        || INTEL_INTERLEAVE == SoftwareLayer->Config.Kind
        || INTEL_DEINTERLEAVE == SoftwareLayer->Config.Kind
        || INTEL_COPY == SoftwareLayer->Config.Kind)
    {
        lyrsCfg->offset = getOffset(XnnDescriptor) + offsetof(XNN_LYR, out_act_fn_buffer);
    }
    else
    {
        lyrsCfg->offset = getOffset(XnnDescriptor) + offsetof(XNN_LYR, out_sum_buffer);
    }
    lyrsCfg->value = getOffset(buffer->Get());

    ++lyrsCfg;
}

void HardwareLayer::WriteNnopType(PNNOP_TYPE_DESCR, bool) const
{
    throw GnaException(XNN_ERR_LYR_CFG);
}

void HardwareLayer::WriteActiveList(HardwareActiveListDescriptor & descriptor) const
{
    auto& config = descriptor.Config.Xnn;
    config->act_list_buffer_offset = getOffset(XnnDescriptor) + offsetof(XNN_LYR, act_list_buffer);
    config->act_list_buffer_value = getOffset(descriptor.List->Indices);
    config->act_list_n_elems_offset = getOffset(XnnDescriptor) + offsetof(XNN_LYR, act_list_n_elems);
    config->act_list_n_elems_value = descriptor.List->IndicesCount;
}

void HardwareLayer::save()
{
    XnnDescriptor->op = OperationsMap.at(SoftwareLayer->Config.Kind);
    XnnDescriptor->n_in_elems = static_cast<uint16_t>(SoftwareLayer->Input.ElementCount);
    XnnDescriptor->n_out_elems = static_cast<uint16_t>(SoftwareLayer->Output.ElementCount);
    XnnDescriptor->n_groups = static_cast<uint8_t>(SoftwareLayer->Input.VectorCount);
    XnnDescriptor->in_buffer = getOffset(SoftwareLayer->Input.Buffer);
    XnnDescriptor->out_act_fn_buffer = getOffset(SoftwareLayer->Output.Buffer);
    if (LayerOutput::OutputMode::NonActivatedOutput == SoftwareLayer->Output.GetOutputMode())
    {
        XnnDescriptor->out_sum_buffer = getOffset(SoftwareLayer->Output.Buffer);
    }
    else
    {
        XnnDescriptor->out_sum_buffer = getOffset(SoftwareLayer->Output.ScratchPad);
    }
}

const map<const uint32_t, const array<const uint32_t, XNN_N_GROUP_MAX>> HardwareLayerExt::bufferElementsMap
{
    { 24,{ 12288, 12288, 12096, 12288, 12000, 12096, 12096, 12288 } },
    { 12,{ 6144, 6144, 6048, 6144, 5760, 6048, 6048, 6144 } },
    { 6,{ 3072, 3072, 2880, 3072, 2880, 2880, 3024, 3072 } }
};

HardwareLayerExt::HardwareLayerExt(const DescriptorParameters& parameters, const uint32_t effectiveGrouping) :
    HardwareLayer(parameters),
    bufferElementCount{bufferElementsMap.at(HardwareInternalBufferSize).at(effectiveGrouping - 1)},
    iterationGrouping{effectiveGrouping}
{
    Expect::InRange(iterationGrouping, 1, XNN_N_GROUP_MAX, XNN_ERR_GROUPING);
    // Calculates number of iterations and elements in last iteration
     //#groups for calculation(can be different than network grouping)
    auto elementsTimesGrouping = SoftwareLayer->Input.ElementCount * iterationGrouping;
    iterationCount = ((elementsTimesGrouping - 1) / bufferElementCount) + 1;
    Expect::InRange(iterationCount, 1, UINT8_MAX, XNN_ERR_LYR_CFG);

    lastIterationElementCount = ((elementsTimesGrouping) - ((iterationCount - 1) * bufferElementCount)) / iterationGrouping;
    Expect::InRange(lastIterationElementCount, 1, bufferElementCount, XNN_ERR_LYR_CFG);
    Expect::MultiplicityOf(lastIterationElementCount, XNN_N_IN_ELEMS_MPLY);
}

void HardwareLayerExt::save()
{
    HardwareLayer::save();
    XnnDescriptor->n_iters = static_cast<uint8_t>(iterationCount);
    XnnDescriptor->n_elems_last = static_cast<uint16_t>(lastIterationElementCount);

    if (affine)
    {
        XnnDescriptor->flags.weight_size = affine->Mode;
        XnnDescriptor->aff_weight_buffer = getOffset(affine->Weights);
        XnnDescriptor->aff_const_buffer = getOffset(affine->Biases);
    }
    if (activation)
    {
        XnnDescriptor->flags.act_fn_en = 1;
        XnnDescriptor->pwl_n_segs = static_cast<uint8_t>(activation->SegmentCount);
        XnnDescriptor->pwl_seg_def_buffer = getOffset(activation->Segments);
    }
    else
    {
        XnnDescriptor->flags.act_fn_en = 0;
    }
}

HardwareLayerAffDiagTrans::HardwareLayerAffDiagTrans(const DescriptorParameters& parameters) :
    HardwareLayerExt(parameters, parameters.SoftwareLayer->Input.VectorCount)
{
    switch (SoftwareLayer->Config.Kind)
    {
    case INTEL_AFFINE:
    case INTEL_AFFINE_DIAGONAL:
        auto aff = SoftwareLayer->Get<const AffineLayer>();
        affine = aff->Affine.get();
        activation = aff->Activation.get();
        break;
    }
    save();
}

void HardwareLayerAffDiagTrans::WriteNnopType(PNNOP_TYPE_DESCR nnopCfg, bool actListEnabled) const
{
    if (INTEL_AFFINE == SoftwareLayer->Config.Kind)
    {
        nnopCfg->offset = getOffset(XnnDescriptor) + offsetof(XNN_LYR, op);
        nnopCfg->value = actListEnabled ? NN_AFF_AL : NN_AFFINE;
    }
}

HardwareLayerCopy::HardwareLayerCopy(const DescriptorParameters& parameters) :
    HardwareLayer(parameters)
{
    save();
}

void HardwareLayerCopy::save()
{
    HardwareLayer::save();
    auto copy = SoftwareLayer->Get<const CopyLayer>();
    XnnDescriptor->cpy_n_elems = static_cast<uint16_t>(copy->ColumnCount);
    XnnDescriptor->n_groups = static_cast<uint8_t>(copy->RowCount);
}

HardwareLayerRnn::HardwareLayerRnn(const DescriptorParameters& parameters) :
    HardwareLayerExt(parameters, 1),
    feedbackIterationsCount{0},
    feedbackFirstIterElementCount{0},
    feedbackLastIterElementCount{0}
{
    auto rnn = SoftwareLayer->Get<const RnnLayer>();
    affine = rnn->Affine.get();
    activation = rnn->Activation.get();
    convert();
    save();
};

void HardwareLayerRnn::WriteOutputBuffer(PGNA_BUFFER_DESCR& lyrsCfg, const ConfigurationBuffer * buffer) const
{
    HardwareLayer::WriteOutputBuffer(lyrsCfg, buffer);

    lyrsCfg->offset = getOffset(XnnDescriptor) + offsetof(XNN_LYR, rnn_out_fb_buffer);
    lyrsCfg->value = CalculateFeedbackBuffer(*buffer);

    ++lyrsCfg;
}

void HardwareLayerRnn::convert()
{
    auto elementCount = SoftwareLayer->Output.ElementCount;

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
    XnnDescriptor->rnn_n_fb_iters = feedbackIterationsCount;
    XnnDescriptor->rnn_n_elems_first = feedbackFirstIterElementCount;
    XnnDescriptor->rnn_n_elems_last = feedbackLastIterElementCount;
    // will be 0 for hidden layers
    if (INTEL_INPUT == SoftwareLayer->Config.Type || INTEL_HIDDEN == SoftwareLayer->Config.Type)
    {
        XnnDescriptor->rnn_out_fb_buffer = CalculateFeedbackBuffer(SoftwareLayer->Output.Buffer);
    }
}

const uint32_t HardwareLayerRnn::CalculateFeedbackBuffer(const OutputBuffer& outputBuffer) const
{
    auto rnn = SoftwareLayer->Get<const RnnLayer>();
    return getOffset(rnn->CalculateFeedbackBuffer(outputBuffer));
}

HardwareLayerCnn::HardwareLayerCnn(const DescriptorParameters& parameters) :
    HardwareLayerExt(parameters, 1)
{
    auto cnn = SoftwareLayer->Get<const CnnLayer>();
    activation = cnn->Activation.get();

    auto fitlerCount = cnn->Convolution.Filters.Count;
    auto fitlerSize = cnn->Convolution.Filters.CoefficientCount;
    filtersCountInFullIteration =
        min(
            fitlerCount,
            (fitlerSize <= bufferElementCount / 6 / 3) ?
                16ui32 :
                (fitlerSize <= bufferElementCount / 6 / 2) ?
                    12ui32 :
                    (fitlerSize <= bufferElementCount / 6) ?
                        4ui32 :
                        0ui32);
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

    // No pooling
    outputElementCount = cnn->Convolution.OutputElementsCount;
    convOutputElementCount = cnn->Convolution.OutputElementsCount;
    // Pooling enabled
    if (INTEL_NO_POOLING != cnn->Pooling.Type) // use pooled outputs per filter
    {
        outputElementCount = ((convOutputElementCount - 1) / cnn->Pooling.Stride + 1);
    }

    save();
}

void HardwareLayerCnn::save()
{
    HardwareLayerExt::save();
    // some fields saved by HardwareLayerExt will be overwritten
    auto cnn = SoftwareLayer->Get<const CnnLayer>();
    XnnDescriptor->flags.pool_param = static_cast<uint8_t>(cnn->Pooling.Type);
    XnnDescriptor->cnn_flt_bf_sz_iter = filtersElementCountInFullIteration;
    XnnDescriptor->cnn_flt_bf_sz_last = filtersElementCountInLastIteration;
    XnnDescriptor->cnn_flt_buffer = getOffset(cnn->Convolution.Filters.Data);
    XnnDescriptor->cnn_flt_size = cnn->Convolution.Filters.CoefficientCount;
    XnnDescriptor->cnn_n_flts = cnn->Convolution.Filters.Count;
    XnnDescriptor->cnn_n_flts_iter = filtersCountInFullIteration;
    XnnDescriptor->cnn_n_flt_iters = filtersIterationCount;
    XnnDescriptor->cnn_n_flt_last = filtersCountInLastIteration;
    XnnDescriptor->cnn_n_flt_outs = convOutputElementCount;
    XnnDescriptor->cnn_n_flt_stride = cnn->Convolution.FeatureMaps.Stride;
    XnnDescriptor->cnn_n_out_p_flt = outputElementCount;
    XnnDescriptor->cnn_pool_size = cnn->Pooling.Size;
    XnnDescriptor->cnn_pool_stride = cnn->Pooling.Stride;
    XnnDescriptor->aff_const_buffer = getOffset(cnn->Convolution.Filters.Biases);
}

HardwareLayerAffineMBias::HardwareLayerAffineMBias(const DescriptorParameters& parameters) :
    HardwareLayerExt(parameters, parameters.SoftwareLayer->Input.VectorCount)
{
    auto mbiasLayer = SoftwareLayer->Get<const AffineLayer>();
    auto affineMulti = static_cast<const AffineFunctionMulti*>(mbiasLayer->Affine.get());
    activation = mbiasLayer->Activation.get();

    save();

    XnnDescriptor->aff_weight_buffer = getOffset(affineMulti->Weights);
    XnnDescriptor->flags.weight_size = affineMulti->Mode;

    XnnDescriptor->bias_grp_cnt = affineMulti->BiasVectorCount;
    XnnDescriptor->bias_grp_ptr = getOffset(affineMulti->Biases);
    XnnDescriptor->bias_grp_value = affineMulti->BiasVectorIndex;

    if (affineMulti->Mode == GNA_WEIGHT_1B)
    {
        XnnDescriptor->aff_const_buffer = 
            getOffset((static_cast<const AffineFunctionMulti1B*>(affineMulti)->WeightScaleFactors));
    }
}

const std::map<const gna_gmm_mode, const GMM_MODE_CTRL> HardwareLayerGmm::GmmModes =
{
    //{ gna_gmm_mode, { read_elimination, calculation_mode, __res_03} },
    { GNA_MAXMIX8, { 0, 0, 0 } },
    { GNA_MAXMIX16,{ 0, 0, 0 } },
};

HardwareLayerGmm::HardwareLayerGmm(const DescriptorParameters& parameters) :
    HardwareLayer(parameters)
{
    save();
}

void HardwareLayerGmm::save()
{
    XnnDescriptor->op = OperationsMap.at(SoftwareLayer->Config.Kind);
    XnnDescriptor->gmm_descriptor = getOffset(GmmDescriptor);
    auto gmm = SoftwareLayer->Get<const GmmLayer>();
    // can be updated per request
    GmmDescriptor->fvaddr      = getOffset(gmm->Input.Buffer);
    GmmDescriptor->gmmscradd   = getOffset(gmm->Output.Buffer);

    // GMM Model configuration, will be constant over time for model
    GmmDescriptor->gmmscrlen   = GMM_SCORE_SIZE * gmm->Input.VectorCount * gmm->Config.stateCount;; // will be updated when ActiveList is used
    GmmDescriptor->fvoffset    = ALIGN64(gmm->Input.ElementCount * GMM_FV_ELEMENT_SIZE);

    GmmDescriptor->numfv       = gmm->Input.VectorCount;
    GmmDescriptor->vlength     = gmm->Input.ElementCount;

    GmmDescriptor->mode        = GmmModes.at(gmm->Config.mode);
    GmmDescriptor->gcaddr      = getOffset(gmm->Data.gaussianConstants);
    GmmDescriptor->mvaddr      = getOffset(gmm->Data.meanValues);
    GmmDescriptor->vvaddr      = getOffset(gmm->Data.inverseCovariancesForMaxMix16);

    GmmDescriptor->gcsoffset   = gmm->Params.GaussConstSetOffsetSize;
    GmmDescriptor->mvsoffset   = gmm->Params.MeanSetOffsetSize;
    GmmDescriptor->vvsoffset   = gmm->Params.VarSetOffsetSize;
    GmmDescriptor->vvwidth     = gmm->Params.VarianceSize;
    GmmDescriptor->gmmtelst    = gmm->Config.mixtureComponentCount * gmm->Input.ElementCount;
    GmmDescriptor->maxlsscore  = gmm->Config.maximumScore;
    GmmDescriptor->numgmms     = gmm->Config.stateCount;
    GmmDescriptor->nummcpg     = gmm->Config.mixtureComponentCount;

    GmmDescriptor->fvwidth     = GMM_FV_ELEMENT_SIZE;
    GmmDescriptor->gcwidth     = GMM_CONSTANTS_SIZE;
    GmmDescriptor->gmmscrwdth  = GMM_SCORE_SIZE;
    GmmDescriptor->maxlswidth  = GMM_SCORE_SIZE;
    GmmDescriptor->mvwidth     = GMM_MEAN_VALUE_SIZE;
}

void HardwareLayerGmm::WriteInputBuffer(PGNA_BUFFER_DESCR& lyrsCfg, const ConfigurationBuffer * const buffer) const
{
    lyrsCfg->offset = getOffset(GmmDescriptor) + offsetof(GMM_CONFIG, fvaddr);
    lyrsCfg->value = getOffset(buffer->Get());

    ++lyrsCfg;
}

void HardwareLayerGmm::WriteOutputBuffer(PGNA_BUFFER_DESCR& lyrsCfg, const ConfigurationBuffer * const buffer) const
{
    lyrsCfg->offset = getOffset(GmmDescriptor) + offsetof(GMM_CONFIG, gmmscradd);
    lyrsCfg->value = getOffset(buffer->Get());

    ++lyrsCfg;
}

void HardwareLayerGmm::WriteNnopType(PNNOP_TYPE_DESCR nnopCfg, bool actListEnabled) const
{
    nnopCfg->offset = getOffset(XnnDescriptor) + offsetof(XNN_LYR, op);
    nnopCfg->value = actListEnabled ? NN_GMM_ACTIVE_LIST : NN_GMM;
}

void HardwareLayerGmm::WriteActiveList(HardwareActiveListDescriptor & descriptor) const
{
    auto gmm = SoftwareLayer->Get<const GmmLayer>();

    auto scoreElementsCount = GMM_SCORE_SIZE * gmm->Input.VectorCount * gmm->Config.stateCount;
    auto activeListIndices = 0ui32;
    auto activeListIndicesCount = 0ui32;
    if (descriptor.List)
    {
        scoreElementsCount = GMM_SCORE_SIZE * gmm->Input.VectorCount * descriptor.List->IndicesCount;
        activeListIndices = getOffset(descriptor.List->Indices);
        activeListIndicesCount = descriptor.List->IndicesCount;
    }

    auto& config = descriptor.Config.Gmm;
    config->asladdr_offset = getOffset(GmmDescriptor) + offsetof(GMM_CONFIG, asladdr);
    config->asladdr_value = activeListIndices;
    config->astlistlen_offset = getOffset(GmmDescriptor) + offsetof(GMM_CONFIG, astlistlen);
    config->astlistlen_value = activeListIndicesCount;
    config->gmmscrlen_offset = getOffset(GmmDescriptor) + offsetof(GMM_CONFIG, gmmscrlen);
    config->gmmscrlen_value = scoreElementsCount;
}
