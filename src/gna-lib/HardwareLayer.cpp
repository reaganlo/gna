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

#include <algorithm>
#include <cmath>

#include "Cnn2DuArch.h"
#include "ConvolutionalFunctions2D.h"
#include "ConvolutionalLayer.h"
#include "CopyLayer.h"
#include "Expect.h"
#include "GmmLayer.h"
#include "HardwareCapabilities.h"
#include "Macros.h"
#include "PoolingFunctions2D.h"
#include "RecurrentLayer.h"
#include "RequestConfiguration.h"
#include "TransposeLayer.h"

using std::array;
using std::make_unique;
using std::map;
using std::unique_ptr;

using namespace GNA;

DescriptorParameters::DescriptorParameters(
        const Layer* softwareLayer,
        const LayerDescriptor& xnnDescriptor) :
    SoftwareLayer{ softwareLayer },
    XnnDescriptor{ xnnDescriptor },
    GmmDescriptor{ xnnDescriptor.GmmDescriptor }
{
};

const map<const nn_operation, const NN_OP_TYPE> HardwareLayer::OperationsMap =
{
    { INTEL_AFFINE, NN_AFFINE },
    { INTEL_AFFINE_DIAGONAL, NN_DIAG },
    { INTEL_AFFINE_MULTIBIAS, NN_AFF_MB },
    { INTEL_CONVOLUTIONAL, NN_CNN },
    { GNA_LAYER_CNN_2D_ADDITION, NN_CNN2D_ADDITION },
    { GNA_LAYER_CNN_2D_CONVERSION, NN_CNN2D_CONVERTION },
    { GNA_LAYER_CNN_2D_POOLING, NN_CNN2D_POOLING },
    { INTEL_CONVOLUTIONAL_2D, NN_CNN2D_FUSED },
    { INTEL_COPY, NN_COPY },
    { INTEL_DEINTERLEAVE, NN_DEINT },
    { INTEL_GMM, NN_GMM },
    { INTEL_INTERLEAVE, NN_INTER },
    { INTEL_RECURRENT, NN_RNN }
};

unique_ptr<HardwareLayer> HardwareLayer::Create(const DescriptorParameters& parameters)
{
    switch (OperationsMap.at(parameters.SoftwareLayer->Operation))
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
    case NN_CNN2D_FUSED:
        return make_unique<HardwareLayerCnn2D>(parameters);
    default:
        return make_unique<HardwareLayerAffDiagTrans>(parameters);
    }
}

HardwareLayer::HardwareLayer(const DescriptorParameters& parameters) :
    DescriptorParameters{parameters}
{
}

NN_OP_TYPE HardwareLayer::GetNnopType(bool) const
{
    throw GnaException { Gna2StatusXnnErrorLyrCfg };
}

NN_OP_TYPE HardwareLayerAffDiagTrans::GetNnopType(bool hasActiveList) const
{
    return hasActiveList ? NN_AFF_AL : NN_AFFINE;
}

NN_OP_TYPE HardwareLayerGmm::GetNnopType(bool hasActiveList) const
{
    return hasActiveList ? NN_GMM_ACTIVE_LIST : NN_GMM;
}

uint32_t HardwareLayer::GetXnnDescriptorOffset() const
{
    return XnnDescriptor.GetOffset();
}

uint32_t HardwareLayer::GetGmmDescriptorOffset() const
{
    throw GnaException { Gna2StatusXnnErrorLyrCfg };
}

uint32_t HardwareLayerGmm::GetGmmDescriptorOffset() const
{
    return XnnDescriptor[gmm_descriptor].Get();
}

uint32_t HardwareLayerGmm::GetLdOutputOffset() const
{
    return XnnDescriptor[gmmscradd].GetOffset();
}

uint32_t HardwareLayerGmm::GetLdInputOffset() const
{
    return XnnDescriptor[fvaddr].GetOffset();
}

uint32_t HardwareLayerGmm::GetScrlen(uint32_t indicesCount) const
{
    auto gmm = SoftwareLayer->Get<const GmmLayer>();
    return GMM_SCORE_SIZE * gmm->Input.at(GNA_DIM_N) * indicesCount;
}

uint32_t HardwareLayerGmm::GetLdScrlenOffset() const
{
    return XnnDescriptor[gmmscrlen].GetOffset();
}

uint32_t HardwareLayerGmm::GetLdActlenOffset() const
{
    return XnnDescriptor[astlistlen].GetOffset();
}

uint32_t HardwareLayerGmm::GetLdActlistOffset() const
{
    return XnnDescriptor[asladdr].GetOffset();
}

uint32_t HardwareLayer::GetScrlen(uint32_t) const
{
    throw GnaException(Gna2StatusXnnErrorLyrCfg);
}

uint32_t HardwareLayer::GetLdScrlenOffset() const
{
    throw GnaException(Gna2StatusXnnErrorLyrCfg);
}

uint32_t HardwareLayer::GetLdGmmMeanOffset() const
{
    throw GnaException(Gna2StatusXnnErrorLyrCfg);
}

uint32_t HardwareLayer::GetLdGmmInverseCovarianceOffset() const
{
    throw GnaException(Gna2StatusXnnErrorLyrCfg);
}

uint32_t HardwareLayer::GetLdGaussianConstantOffset() const
{
    throw GnaException(Gna2StatusXnnErrorLyrCfg);
}

uint32_t HardwareLayer::GetLdWeightOffset() const
{
    return XnnDescriptor[weight_buffer].GetOffset();
}

uint32_t HardwareLayer::GetLdBiasOffset() const
{
    return XnnDescriptor[bias_buffer].GetOffset();
}

uint32_t HardwareLayer::GetLdFilterOffset() const
{
    return XnnDescriptor[weight_buffer].GetOffset();
}

uint32_t HardwareLayer::GetLdIntermediateOutputOffset() const
{
    return XnnDescriptor[out_sum_buffer].GetOffset();
}

uint32_t HardwareLayer::GetLdWeightScaleFactorOffset() const
{
    if (INTEL_AFFINE_MULTIBIAS == SoftwareLayer->Operation)
    {
        return XnnDescriptor[bias_buffer].GetOffset();
    }
    throw GnaException(Gna2StatusXnnErrorLyrCfg);
}

uint32_t HardwareLayer::GetLdPwlOffset() const
{
    return XnnDescriptor[pwl_seg_def_buffer].GetOffset();
}

uint32_t HardwareLayer::GetLdActlistOffset() const
{
    return XnnDescriptor[act_list_buffer].GetOffset();
}

uint32_t HardwareLayer::GetLdActlenOffset() const
{
    return XnnDescriptor[act_list_n_elems].GetOffset();
}

uint32_t HardwareLayer::GetLdNnopOffset() const
{
    return XnnDescriptor[op].GetOffset();
}

uint32_t HardwareLayer::GetLdInputOffset() const
{
    return XnnDescriptor[in_buffer].GetOffset();
}

uint32_t HardwareLayer::GetLdOutputOffset() const
{
    if (GNA_DATA_ACTIVATION_DISABLED != SoftwareLayer->Output.Mode
        || INTEL_CONVOLUTIONAL == SoftwareLayer->Operation
        || INTEL_INTERLEAVE == SoftwareLayer->Operation
        || INTEL_DEINTERLEAVE == SoftwareLayer->Operation
        || INTEL_COPY == SoftwareLayer->Operation)
    {
        return XnnDescriptor[out_buffer].GetOffset();
    }
    else
    {
        return XnnDescriptor[out_sum_buffer].GetOffset();
    }
}

uint32_t HardwareLayer::GetLdFeedbackOffset() const
{
    //auto& config = descriptor.Config.Xnn;
    //config->act_list_buffer_offset = XnnDescriptor[act_list_buffer].GetOffset();
    //config->act_list_buffer_value = XnnDescriptor[act_list_buffer].GetBufferOffset(descriptor.List->Indices);
    //config->act_list_n_elems_offset = XnnDescriptor[act_list_n_elems].GetOffset();
    //config->act_list_n_elems_value = descriptor.List->IndicesCount;
    //if (descriptor.List->IndicesCount != 0) XnnDescriptor[AL_bit] = (uint8_t)1; //FOR Debug purposes only
    //XnnDescriptor[act_list_buffer] = XnnDescriptor[act_list_buffer].GetBufferOffset(descriptor.List->Indices);
    //XnnDescriptor[act_list_n_elems] = descriptor.List->IndicesCount;
    throw GnaException { Gna2StatusXnnErrorLyrCfg };
}

void HardwareLayer::saveCommonPart()
{
    XnnDescriptor[op] =static_cast<uint8_t>(OperationsMap.at(SoftwareLayer->Operation));
    XnnDescriptor[n_groups] = SoftwareLayer->Input.at(GNA_DIM_N);
    XnnDescriptor[in_buffer] = SoftwareLayer->Input;
    if(XnnDescriptor.HasParameter(input_element_precision))
    {
        XnnDescriptor[input_element_precision] = SoftwareLayer->Input.Mode;
    }
    XnnDescriptor[out_buffer] = SoftwareLayer->Output;
}

void HardwareLayer::save()
{
    saveCommonPart();

    XnnDescriptor[n_in_elems] = SoftwareLayer->Input.at(GNA_DIM_W);
    XnnDescriptor[n_out_elems] = SoftwareLayer->Output.at(GNA_DIM_H);
    if (GNA_DATA_ACTIVATION_DISABLED == SoftwareLayer->Output.Mode)
    {
        XnnDescriptor[out_sum_buffer] = SoftwareLayer->Output;
    }
    else
    {
        XnnDescriptor[out_sum_buffer] = SoftwareLayer->Output.ScratchPad;
    }
}

void HardwareLayer::saveActivation(const ActivationFunction* activationIn)
{
    if (activationIn)
    {
        if (XnnDescriptor.HasParameter(act_fn_precision))
        {
            XnnDescriptor[act_fn_precision] = SoftwareLayer->Output.Mode;
        }
        XnnDescriptor[pwl_n_segs] = activationIn->Segments->Count;
        XnnDescriptor[pwl_seg_def_buffer] = *activationIn->Segments;
    }
}

HardwareLayerExt::HardwareLayerExt(const DescriptorParameters& parameters, const uint32_t effectiveGrouping) :
    HardwareLayer(parameters),
    bufferElementCount { parameters.XnnDescriptor.HwCapabilities.GetBufferElementCount(
                            effectiveGrouping, SoftwareLayer->Input.Mode) },
    iterationGrouping{effectiveGrouping}
{
    Expect::InRange(iterationGrouping, ui32_1, XNN_N_GROUP_MAX, Gna2StatusXnnErrorGrouping);
    // Calculates number of iterations and elements in last iteration
     //#groups for calculation(can be different than network grouping)
    auto elementsTimesGrouping = SoftwareLayer->Input.at(GNA_DIM_W) * iterationGrouping;
    iterationCount = ((elementsTimesGrouping - 1) / bufferElementCount) + 1;
    Expect::InRange(iterationCount, ui32_1, ui32_UINT8_MAX, Gna2StatusXnnErrorLyrCfg);

    lastIterationElementCount = ((elementsTimesGrouping) - ((iterationCount - 1) * bufferElementCount)) / iterationGrouping;
    Expect::InRange(lastIterationElementCount, ui32_1, bufferElementCount, Gna2StatusXnnErrorLyrCfg);
    Expect::MultiplicityOf(lastIterationElementCount, XNN_N_IN_ELEMS_MPLY);
}

void HardwareLayerExt::save()
{
    HardwareLayer::save();
    XnnDescriptor[n_iters] = iterationCount;
    XnnDescriptor[n_elems_last] = lastIterationElementCount;

    if (affine)
    {
        XnnDescriptor[weight_size] = affine->Weights->Mode;
        XnnDescriptor[weight_buffer] = *affine->Weights;
        XnnDescriptor[bias_buffer] = *affine->Biases;
        if(XnnDescriptor.HasParameter(bias_precision))
        {
            XnnDescriptor[bias_precision] = affine->Biases->Mode;
        }
    }
}

HardwareLayerAffDiagTrans::HardwareLayerAffDiagTrans(const DescriptorParameters& parameters) :
    HardwareLayerExt(parameters, parameters.SoftwareLayer->Input.at(GNA_DIM_N))
{
    const ActivationFunction* act = nullptr;
    switch (SoftwareLayer->Operation)
    {
        case INTEL_AFFINE:
        case INTEL_AFFINE_DIAGONAL:
            {
                auto aff = SoftwareLayer->Get<const AffineLayer>();
                affine = aff->Affine.get();
                act = aff->Activation.get();
                break;
            }
        case INTEL_INTERLEAVE:
        case INTEL_DEINTERLEAVE:
            break;
        default:
            throw GnaException { Gna2StatusXnnErrorLyrOperation };
    }
    save();
    saveActivation(act);
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
    XnnDescriptor[cpy_n_elems] = copy->ColumnCount;
    XnnDescriptor[n_groups] = copy->RowCount;
}

HardwareLayerRnn::HardwareLayerRnn(const DescriptorParameters& parameters) :
    HardwareLayerExt(parameters, 1),
    feedbackIterationsCount{0},
    feedbackFirstIterElementCount{0},
    feedbackLastIterElementCount{0}
{
    auto rnn = SoftwareLayer->Get<const RnnLayer>();
    affine = rnn->Affine.get();
    convert();
    save();
    saveActivation(rnn->Activation.get());
};

void HardwareLayerRnn::convert()
{
    auto elementCount = SoftwareLayer->Output.at(GNA_DIM_H);

    feedbackFirstIterElementCount = (std::min)((bufferElementCount - lastIterationElementCount), elementCount);
    Expect::True(feedbackFirstIterElementCount <= bufferElementCount, Gna2StatusXnnErrorLyrCfg);

    feedbackIterationsCount = (elementCount - feedbackFirstIterElementCount) / bufferElementCount;
    if ((elementCount - feedbackFirstIterElementCount) % bufferElementCount)
    {
        feedbackIterationsCount++;
    }
    if (feedbackFirstIterElementCount > 0)
    {
        feedbackIterationsCount++;
    }
    Expect::InRange(feedbackIterationsCount, ui32_1, ui32_UINT8_MAX, Gna2StatusXnnErrorLyrCfg);

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
    Expect::InRange(feedbackLastIterElementCount, ui32_1, bufferElementCount, Gna2StatusXnnErrorLyrCfg);
}

void HardwareLayerRnn::save()
{
    HardwareLayerExt::save();
    XnnDescriptor[rnn_n_fb_iters] = feedbackIterationsCount;
    XnnDescriptor[rnn_n_elems_first] = feedbackFirstIterElementCount;
    XnnDescriptor[rnn_n_elems_last] = feedbackLastIterElementCount;
    // can be negative for non-output layers (if user provides nullptr as output buffer)
    auto rnn = SoftwareLayer->Get<const RnnLayer>();
    XnnDescriptor[rnn_out_fb_buffer] = rnn->CalculateFeedbackBuffer(SoftwareLayer->Output);
}

uint32_t HardwareLayerRnn::GetLdFeedbackOffset() const
{
    return XnnDescriptor[rnn_out_fb_buffer].GetOffset();
}

HardwareLayerCnn::HardwareLayerCnn(const DescriptorParameters& parameters) :
    HardwareLayerExt(parameters, 1)
{
    auto cnn = SoftwareLayer->Get<const CnnLayer>();

    auto filterCount = cnn->Convolution->Filters->Count;
    auto filterSize = cnn->Convolution->Filters->CoefficientCount;
    filtersCountInFullIteration =
        (std::min)(
            filterCount,
            (filterSize <= bufferElementCount / 6 / 3) ?
                uint32_t{16} :
                (filterSize <= bufferElementCount / 6 / 2) ?
                    uint32_t{12} :
                    (filterSize <= bufferElementCount / 6) ?
                        uint32_t{4} : uint32_t{0});

    Expect::InRange(filtersCountInFullIteration, CNN_N_FLT_COEFF_MPLY, CNN_N_FLT_ITER_MAX, Gna2StatusXnnErrorLyrCfg);
    Expect::MultiplicityOf(filtersCountInFullIteration, CNN_N_FLT_COEFF_MPLY);

    filtersIterationCount = (filterCount - 1) / filtersCountInFullIteration + 1;

    filtersCountInLastIteration = filterCount - ((filtersIterationCount - 1) * filtersCountInFullIteration);
    Expect::InRange(filtersCountInLastIteration, CNN_N_FLT_COEFF_MPLY, CNN_N_FLT_ITER_MAX, Gna2StatusXnnErrorLyrCfg);
    Expect::MultiplicityOf(filtersCountInLastIteration, CNN_N_FLT_COEFF_MPLY);


    filtersElementCountInFullIteration = filtersCountInFullIteration * filterSize;
    Expect::InRange(filtersElementCountInFullIteration, ui32_1, bufferElementCount, Gna2StatusXnnErrorLyrCfg);

    filtersElementCountInLastIteration = filtersCountInLastIteration * filterSize;
    Expect::InRange(filtersElementCountInLastIteration, ui32_1, bufferElementCount, Gna2StatusXnnErrorLyrCfg);

    // No pooling
    outputElementCount = cnn->Convolution->OutputsPerFilterCount;
    convOutputElementCount = cnn->Convolution->OutputsPerFilterCount;
    // Pooling enabled
    if (NULL != cnn->Pooling && INTEL_NO_POOLING != cnn->Pooling->Type) // use pooled outputs per filter
    {
        outputElementCount = cnn->Pooling->OutputsPerFilterCount;
    }

    save();
    saveActivation(cnn->Activation.get());

}

void HardwareLayerCnn::save()
{
    HardwareLayerExt::save();
    // some fields saved by HardwareLayerExt will be overwritten
    auto cnn = SoftwareLayer->Get<const CnnLayer>();

    if (NULL != cnn->Pooling)
    {
        XnnDescriptor[pool_param] = static_cast<uint8_t>(cnn->Pooling->Type);
        XnnDescriptor[cnn_pool_size] = cnn->Pooling->Window.at(GNA_DIM_W);
        XnnDescriptor[cnn_pool_stride] = cnn->Pooling->Stride.at(GNA_DIM_W);
    }

    if(XnnDescriptor.HasParameter(bias_precision))
    {
        XnnDescriptor[bias_precision] = cnn->Convolution->Biases->Mode;
    }
    XnnDescriptor[weight_size] = cnn->Convolution->Filters->Mode;
    XnnDescriptor[weight_buffer] = *cnn->Convolution->Filters;
    XnnDescriptor[cnn_flt_bf_sz_iter] = filtersElementCountInFullIteration;
    XnnDescriptor[cnn_flt_bf_sz_last] = filtersElementCountInLastIteration;
    XnnDescriptor[cnn_flt_size] = cnn->Convolution->Filters->CoefficientCount;
    XnnDescriptor[cnn_n_flts] = cnn->Convolution->Filters->Count;
    XnnDescriptor[cnn_n_flts_iter] = filtersCountInFullIteration;
    XnnDescriptor[cnn_n_flt_iters] = filtersIterationCount;
    XnnDescriptor[cnn_n_flt_last] = filtersCountInLastIteration;
    XnnDescriptor[cnn_n_flt_outs] = convOutputElementCount;
    XnnDescriptor[bias_buffer] = *cnn->Convolution->Biases;
    XnnDescriptor[cnn_n_flt_stride] = cnn->Convolution->Stride->at(GNA_DIM_W);
    XnnDescriptor[cnn_n_out_p_flt] = outputElementCount;
    XnnDescriptor[bias_buffer] = *cnn->Convolution->Biases;
}

uint32_t HardwareLayerCnn2D::GetKernelWorkGroupSize(DeviceVersion hw,
    ConvolutionFunction2D const * cnnIn, PoolingFunction2D const * poolingIn,
    const DataMode& outputMode)
{
    UNREFERENCED_PARAMETER(hw);
    convolutional_fused_configuration validationResults;
    auto status = GNA3_PopulateLD(cnnIn, poolingIn, outputMode, &validationResults);
    Expect::True(status, Gna2StatusXnnErrorLyrCfg);
    Expect::True(validationResults.Valid, Gna2StatusXnnErrorLyrCfg);
    return validationResults.KWG;
}

uint32_t HardwareLayerCnn2D::GetKernelMemorySize(DeviceVersion hw,
    FiltersTensor const * filter)
{
    UNREFERENCED_PARAMETER(hw);
    return ALIGN(filter->Size / filter->Count, 16);
}

uint32_t HardwareLayerCnn2D::GetConvolutionMemorySize(DeviceVersion hw,
    ConvolutionFunction2D const * cnnIn)
{
    UNREFERENCED_PARAMETER(hw);
    return cnnIn->Filters->at(GNA_DIM_H) * cnnIn->Output->at(GNA_DIM_W) * 8;
}

uint32_t HardwareLayerCnn2D::GetPoolingMemorySize(DeviceVersion hw,
    PoolingFunction2D const * poolingIn, const DataMode& outputMode)
{
    UNREFERENCED_PARAMETER(hw);
    if(poolingIn)
    {
        auto elementSize = (GNA_INT32 == outputMode) ? 4 : outputMode.Size * 2;
        auto div = (float)poolingIn->Window->at(GNA_DIM_H) / (float)poolingIn->Stride->at(GNA_DIM_H);
        auto ceil = (uint32_t)std::ceil(div);
        return elementSize * poolingIn->Output->at(GNA_DIM_W) * (1 + ceil);
    }
    else
    {
        return 0;
    }
}

HardwareLayerCnn2D::HardwareLayerCnn2D(const DescriptorParameters& parameters) :
    HardwareLayer(parameters),
    cnn{SoftwareLayer->Get()->Transforms.Get<ConvolutionFunction2D>(ConvolutionalTransform2D)},
    pooling{SoftwareLayer->Get()->Transforms.Get<PoolingFunction2D>(PoolingTransform2D)}
{
    kernelWorkGroupSize = GetKernelWorkGroupSize(parameters.XnnDescriptor.HwCapabilities.GetDeviceVersion(),
        cnn, pooling, SoftwareLayer->GetOutputTransform()->Output->Mode);

    kernelWorkGroupIterationCount = GnaCeilDiv(cnn->Filters->Count, kernelWorkGroupSize);

    save();
    saveActivation(
        SoftwareLayer->Get()->Transforms.Get<ActivationFunction>(ActivationTransform));
}

void HardwareLayerCnn2D::save()
{
    saveCommonPart();
    XnnDescriptor[weight_size] = cnn->Filters->Mode;
    XnnDescriptor[weight_buffer] = *cnn->Filters;
    XnnDescriptor[cnn_n_flts] = cnn->Filters->Count;
    XnnDescriptor[bias_buffer] = *cnn->Biases;
    XnnDescriptor[bias_precision] = cnn->Biases->Mode;
    XnnDescriptor[cnn2d_bias_mode] = cnn->Biases->BiasMode;
    XnnDescriptor[cnn2d_in_dim_d] = cnn->Input->Dimensions.at(GNA_DIM_D);
    XnnDescriptor[cnn2d_in_dim_w] = cnn->Input->Dimensions.at(GNA_DIM_W);
    XnnDescriptor[cnn2d_in_dim_h] = cnn->Input->Dimensions.at(GNA_DIM_H);
    XnnDescriptor[cnn2d_padding_w] = cnn->Padding->Dimensions.at(GNA_DIM_W);
    XnnDescriptor[cnn2d_padding_h] = cnn->Padding->Dimensions.at(GNA_DIM_H);
    XnnDescriptor[cnn2d_conv_stride_w] = cnn->Stride->Dimensions.at(GNA_DIM_W);
    XnnDescriptor[cnn2d_conv_stride_h] = cnn->Stride->Dimensions.at(GNA_DIM_H);
    XnnDescriptor[cnn2d_conv_out_w] = cnn->Output->Dimensions.at(GNA_DIM_W);
    XnnDescriptor[cnn2d_conv_out_h] = cnn->Output->Dimensions.at(GNA_DIM_H);
    XnnDescriptor[cnn2d_conv_kernel_w] = cnn->Filters->Dimensions.at(GNA_DIM_W);
    XnnDescriptor[cnn2d_conv_kernel_h] = cnn->Filters->Dimensions.at(GNA_DIM_H);
    XnnDescriptor[cnn2d_kernel_iter] = kernelWorkGroupIterationCount;
    XnnDescriptor[cnn2d_kernel_wg] = kernelWorkGroupSize;
    XnnDescriptor[cnn2d_addaptive] = (uint32_t)0;

    if (pooling)
    {
        XnnDescriptor[cnn2d_pool_stride_w] = pooling->Stride->Dimensions.at(GNA_DIM_H);
        XnnDescriptor[cnn2d_pool_stride_h] = pooling->Stride->Dimensions.at(GNA_DIM_H);
        XnnDescriptor[cnn2d_pool_out_w] = pooling->Output->Dimensions.at(GNA_DIM_W);
        XnnDescriptor[cnn2d_pool_out_h] = pooling->Output->Dimensions.at(GNA_DIM_H);
        XnnDescriptor[cnn2d_pool_window_w] = pooling->Window->Dimensions.at(GNA_DIM_W);
        XnnDescriptor[cnn2d_pool_window_h] = pooling->Window->Dimensions.at(GNA_DIM_H);
        XnnDescriptor[pool_param] = static_cast<uint8_t>(pooling->Type);
    }
}

HardwareLayerAffineMBias::HardwareLayerAffineMBias(const DescriptorParameters& parameters) :
    HardwareLayerExt(parameters, parameters.SoftwareLayer->Input.at(GNA_DIM_N))
{
    auto mbiasLayer = SoftwareLayer->Get<const AffineLayer>();
    auto affineMulti = static_cast<const AffineFunctionMulti*>(mbiasLayer->Affine.get());

    save();
    saveActivation(mbiasLayer->Activation.get());

    XnnDescriptor[weight_buffer] = *affineMulti->Weights;
    XnnDescriptor[weight_size] = affineMulti->Weights->Mode;

    if (XnnDescriptor.HasParameter(bias_precision))
    {
        XnnDescriptor[bias_precision] = affineMulti->Biases->Mode;
    }

    XnnDescriptor[bias_grp_cnt] = affineMulti->Biases->at(GNA_DIM_N);
    XnnDescriptor[bias_grp_buffer] = *affineMulti->Biases;
    XnnDescriptor[bias_grp_value] = affineMulti->Biases->VectorIndex;

    if (affineMulti->Weights->Mode == GNA_INT8 && mbiasLayer->Input.Mode == GNA_INT16)
    {
        XnnDescriptor[bias_buffer] = *affineMulti->WeightScaleFactors;
    }
}

const std::map<const gna_gmm_mode, const GMM_MODE_CTRL> HardwareLayerGmm::GmmModes =
{
    //{ gna_gmm_mode, { read_elimination, calculation_mode, __res_03} },
    { GNA_MAXMIX8,  { 0, 0, 0 } },
    { GNA_MAXMIX16, { 0, 0, 0 } },
};

HardwareLayerGmm::HardwareLayerGmm(const DescriptorParameters& parameters) :
    HardwareLayer(parameters)
{
    save();
}

void HardwareLayerGmm::save()
{
    XnnDescriptor[op] = static_cast<uint8_t>(OperationsMap.at(SoftwareLayer->Operation));
    XnnDescriptor[gmm_descriptor] = XnnDescriptor.GmmDescriptor;

    auto gmm = SoftwareLayer->Get<const GmmLayer>();
    // can be updated per request
    XnnDescriptor[fvaddr]    = gmm->Input;
    XnnDescriptor[gmmscradd] = gmm->Output;

    // GMM Model configuration, will be constant over time for model
    XnnDescriptor[gmmscrlen] = GMM_SCORE_SIZE * gmm->Input.at(GNA_DIM_N) * gmm->Config.stateCount;; // will be updated when ActiveList is used
    XnnDescriptor[fvoffset]  = ALIGN64(gmm->Input.at(GNA_DIM_W) * GMM_FV_ELEMENT_SIZE);

    XnnDescriptor[numfv]     = gmm->Input.at(GNA_DIM_N);
    XnnDescriptor[vlength]   = gmm->Input.at(GNA_DIM_W);

    XnnDescriptor[mode]      = GmmModes.at(gmm->Config.mode)._value;
    XnnDescriptor[gcaddr]    = gmm->Data.gaussianConstants;
    XnnDescriptor[mvaddr]    = gmm->Data.meanValues;
    XnnDescriptor[vvaddr]    = gmm->Data.inverseCovariances.inverseCovariancesForMaxMix16;

    XnnDescriptor[gcsoffset] = gmm->Params.GaussConstSetOffsetSize;
    XnnDescriptor[mvsoffset] = gmm->Params.MeanSetOffsetSize;
    XnnDescriptor[vvsoffset] = gmm->Params.VarSetOffsetSize;
    XnnDescriptor[vvwidth]   = gmm->Params.VarianceSize;
    XnnDescriptor[gmmtelst]  = gmm->Config.mixtureComponentCount * gmm->Input.at(GNA_DIM_W);
    XnnDescriptor[maxlsscore] = gmm->Config.maximumScore;
    XnnDescriptor[numgmms]   = gmm->Config.stateCount;
    XnnDescriptor[nummcpg]   = gmm->Config.mixtureComponentCount;

    XnnDescriptor[fvwidth]   = GMM_FV_ELEMENT_SIZE;
    XnnDescriptor[gcwidth]   = GMM_CONSTANTS_SIZE;
    XnnDescriptor[gmmscrwdth] = GMM_SCORE_SIZE;
    XnnDescriptor[maxlswidth] = GMM_SCORE_SIZE;
    XnnDescriptor[mvwidth]   = GMM_MEAN_VALUE_SIZE;
}


