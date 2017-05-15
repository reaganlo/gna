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

#include "AcceleratorSw.h"

#include "common.h"
#include "AffineLayers.h"
#include "RecurrentLayer.h"
#include "GnaException.h"
#include "Validator.h"

using namespace GNA;

GmmScoreContext::GmmScoreContext(const GmmLayer* gmm, const LayerConfiguration * const layerConfiguration)
{
    if (nullptr != layerConfiguration && layerConfiguration->InputBuffer)
    {
        Input = *layerConfiguration->InputBuffer;
    }
    else
    {
        Input = gmm->Input.Buffer;
    }

    if (nullptr != layerConfiguration && layerConfiguration->OutputBuffer)
    {
        Output = *layerConfiguration->OutputBuffer;
    }
    else
    {
        Output = gmm->Output.Buffer;

    }

    if (nullptr != layerConfiguration && layerConfiguration->ActiveList)
    {
        ActiveList = layerConfiguration->ActiveList.get();
        gmm->ValidateActiveList(ActiveList);
        StateCount = ActiveList->IndicesCount;
    }
    else
    {
        ActiveList = nullptr;
        StateCount = gmm->Config.stateCount;
    }
}

AcceleratorSw::AcceleratorSw(acceleration acceleration_mode) :
    IAccelerator{acceleration_mode}
{
    switch(accel){
    case GNA_AVX2_FAST:
        gmmKernel = &gmmKernel_avx2;
        xnnKernel = &xnnKernel_avx2;
        break;
    case GNA_AVX2_SAT:
        gmmKernel = &gmmKernel_avx2;
        xnnKernel = &xnnKernel_avx2_sat;
        break;
    case GNA_AVX1_FAST:
        gmmKernel = &gmmKernel_avx1;
        xnnKernel = &xnnKernel_avx1;
        break;
    case GNA_AVX1_SAT:
        gmmKernel = &gmmKernel_avx1;
        xnnKernel = &xnnKernel_avx1_sat;
        break;
    case GNA_SSE4_2_FAST:
        gmmKernel = &gmmKernel_sse4;
        xnnKernel = &xnnKernel_sse4;
        break;
    case GNA_SSE4_2_SAT:
        gmmKernel = &gmmKernel_sse4;
        xnnKernel = &xnnKernel_sse4_sat;
        break;
    case GNA_GEN_FAST:
        gmmKernel = &gmmKernel_generic;
        xnnKernel = &xnnKernel_generic;
        break;
    case GNA_GEN_SAT:
        gmmKernel = &gmmKernel_generic;
        xnnKernel = &xnnKernel_generic_sat;
        break;
    default:
        throw GnaException(GNA_CPUTYPENOTSUPPORTED);
    }
}

status_t AcceleratorSw::Score(
    uint32_t layerIndex,
    uint32_t layerCount,
    const RequestConfiguration& requestConfiguration,
    RequestProfiler *profiler,
    KernelBuffers *fvBuffers)
{
    profilerDTscAStart(&profiler->scoring);

    requestConfiguration.Model.ValidateConfiguration(requestConfiguration);

    auto nOuts = uint32_t{ 0 }; // number of outputs
    const uint32_t* activeIndices = nullptr; // active list pointer
    auto sat = uint32_t{ 0 };   // scoring saturation counter
    
    // TODO: refactor to remove dependency on software model
    auto iter = requestConfiguration.Model.GetLayers().begin() + layerIndex;
    auto end = iter + layerCount;
    for (; iter < end; ++iter)
    {
        const auto& layer = *iter;
        auto* sourceLayer = const_cast<nn_layer*>(&layer->sourceLayer);
        nOuts = layer->Output.ElementCount; // regular output (all)
        const LayerConfiguration* layerConfiguration = nullptr;

        if (INTEL_HIDDEN != layer->Config.Type)
        {
            auto found = requestConfiguration.LayerConfigurations.find(layerIndex);
            if (found != requestConfiguration.LayerConfigurations.end())
            {
                layerConfiguration = found->second.get();
                if (INTEL_GMM != layer->Config.Kind)
                {
                    applyRequestBuffersToLayer(*layerConfiguration, layer.get(), *sourceLayer, nOuts, activeIndices);
                }
            }
        }

        switch (layer->Config.Kind)
        {
        case INTEL_AFFINE:
        {
            auto affineLayer = layer->Get<const AffineLayer>();
            auto& activation = affineLayer->Activation;

            xnnKernel->affine(sourceLayer, activeIndices, nOuts, &sat, fvBuffers);

            // apply piecewise linear function if enabled
            if (activation)
            {
                xnnKernel->pwl(sourceLayer, 0, nOuts - 1, 0,
                    sourceLayer->nInputColumns - 1, &sat, fvBuffers->pwl);
            }
            break;
        }
        case INTEL_AFFINE_MULTIBIAS:
        {
            auto affineLayer = layer->Get<const AffineMultiBiasLayer>();
            auto& activation = affineLayer->Activation;

            xnnKernel->affineMbias(sourceLayer, activeIndices, nOuts, &sat, fvBuffers);

            if (activation)
            {
                xnnKernel->pwl(sourceLayer, 0, sourceLayer->nOutputRows - 1, 0,
                    sourceLayer->nInputColumns - 1, &sat, fvBuffers->pwl);
            }
            break;
        }
        case INTEL_AFFINE_DIAGONAL:
        {
            xnnKernel->diagonal(sourceLayer, &sat);

            auto affineLayer = layer->Get<const AffineDiagonalLayer>();
            xnnKernel->diagonal(sourceLayer, &sat);

            auto& activation = affineLayer->Activation;
            if (activation)
            {
                xnnKernel->pwl(sourceLayer, 0, sourceLayer->nOutputRows - 1, 0,
                    sourceLayer->nInputColumns - 1, &sat, fvBuffers->pwl);
            }
            break;
        }
        case INTEL_RECURRENT:
        {
            xnnKernel->recurrent(sourceLayer, &sat, fvBuffers->pwl);
            break;
        }
        case INTEL_INTERLEAVE:
        case INTEL_DEINTERLEAVE:
        {
            xnnKernel->transpose(sourceLayer);
            break;
        }
        case INTEL_COPY:
        {
            xnnKernel->copy(sourceLayer);
            break;
        }
        case INTEL_CONVOLUTIONAL:
        {
            xnnKernel->conv(sourceLayer, &sat, fvBuffers->pwl, fvBuffers->pool);
            break;
        }
        case INTEL_GMM:
        {
            gmmSoftwareKernel(accel, layer->Get<const GmmLayer>(), layerConfiguration, sat);
            break;
        }
        default:
        {
            return XNN_ERR_LYR_TYPE;
        }
        }

        ++layerIndex;
    }

    profilerDTscStop(&profiler->scoring);
    profilerDTscStop(&profiler->total);

    return (sat > 0) ? GNA_SSATURATE : GNA_SUCCESS;
}

void AcceleratorSw::applyRequestBuffersToLayer(
    const LayerConfiguration& layerConfiguration,
    Layer* layer,
    nn_layer& sourceLayer,
    uint32_t &nOuts,
    const uint32_t * &activeIndices)
{
    if (layerConfiguration.InputBuffer)
    {
        sourceLayer.pInputs = *layerConfiguration.InputBuffer;
    }

    if (layerConfiguration.OutputBuffer)
    {
        sourceLayer.pOutputs = *layerConfiguration.OutputBuffer;
        if (INTEL_RECURRENT == layer->Config.Kind)
        {
            auto rnn = layer->Get<RnnLayer>();
            rnn->SetFeedbackBuffer(*layerConfiguration.OutputBuffer);
            // TODO: move to XnnKernel when kernels are refactored
        }
    }

    if (layerConfiguration.ActiveList)
    {
        if (INTEL_AFFINE == layer->Config.Kind
            || INTEL_GMM == layer->Config.Kind)
        {
            nOuts = layerConfiguration.ActiveList->IndicesCount; // active list outputs
            activeIndices = layerConfiguration.ActiveList->Indices;
        }
    }
}

void AcceleratorSw::checkScoresSaturation(const uint32_t& nGMMs, const uint32_t& nVectors, const uint32_t * pS,
    const uint32_t& maxScore, uint32_t& nSaturated)
{
    for(auto i = 0ui32; i < nGMMs * nVectors; i++)
    {
        if (maxScore == *pS)
        {
            nSaturated++;
            return;
        }
        pS++;
    }
}

void AcceleratorSw::gmmSoftwareKernel(const acceleration accel, const GmmLayer* gmm, const LayerConfiguration * const layerConfiguration,
    uint32_t& nSaturated)
{
    const auto context = GmmScoreContext(gmm, layerConfiguration);
    const gna_gmm_data* data = &gmm->Data;
    const uint32_t fvCount = gmm->Input.VectorCount;
    const uint32_t fvLength = gmm->Input.ElementCount;
    const uint32_t mixCount = gmm->Config.mixtureComponentCount;
    const uint32_t meanSetOffsetSize = gmm->Params.MeanSetOffsetSize;
    const uint32_t varSetOffsetSize = gmm->Params.VarSetOffsetSize;
    const uint32_t gConstSetOffsetSize = gmm->Params.GaussConstSetOffsetSize;
    const uint32_t maxScore = gmm->Config.maximumScore;

    uint32_t i, j, k;
    // auxiliary pointers
    uint8_t const *fv;
    uint8_t *means;
    uint32_t *consts;
    uint32_t *scores;

    if (!context.ActiveList)
    {
        const uint32_t meanOffset = meanSetOffsetSize / GMM_MEAN_VALUE_SIZE;;
        const uint32_t varOffset = varSetOffsetSize / (gmm->Config.mode + 1);;
        const uint32_t gConstOffset = gConstSetOffsetSize / GMM_CONSTANTS_SIZE;

        if (gmm->Config.mode == GNA_MAXMIX8)
        {
            uint8_t *vars; // auxiliary pointer
            if(GNA_GEN_FAST == accel || GNA_GEN_SAT == accel)
            {
                for(j = 0; j < context.StateCount; j++)
                {
                    fv = context.Input;
                    means = data->meanValues + j*meanOffset;
                    vars = data->inverseCovariancesForMaxMix8 + j*varOffset;
                    consts = data->gaussianConstants + j*gConstOffset;
                    scores = context.Output + j*fvCount;

                    for(i = 0; i < fvCount; i++)
                    {
                        *scores = gmmKernel->GMM8(fv, means, vars, consts, maxScore, fvLength, mixCount);
                        scores++;
                        fv += fvLength;
                    }
                }
            }
            else //if(GNA_SW_SSE4_2 == accel || GNA_SW_AVX1 == accel || GNA_SW_AVX2 == accel)
            {
                uint8_t pFeatureBuffer[GMM_FV_ELEMENT_COUNT_MAX*GMM_FV_COUNT_MAX+GMM_FV_MEM_ALIGN]; // max data size
                fv = (uint8_t*)(((unsigned long long)pFeatureBuffer+GMM_FV_MEM_ALIGN) & 0xffffffffffffffc0ull); // aligned to GMM_FV_MEM_ALIGN bytes
                uint32_t n;
                uint32_t g;

                // pack feature vectors by 8 features
                // v0[0..7]v1[0..7]vj[0..7]v0[8..15]v1[8..15]...
                if (fvCount > 1)
                {
                    for (n = 0; n < fvLength; n += GMM_FV_COUNT_MAX)
                    {
                        for (g = 0; g < fvCount; g++)
                        {
                            *((uint64_t*)fv) = *((uint64_t*)((context.Input) + g * fvLength + n));
                            fv += GMM_FV_COUNT_MAX;
                        }
                    }
                    fv = (uint8_t*)(((unsigned long long)pFeatureBuffer+GMM_FV_MEM_ALIGN) & 0xffffffffffffffc0ull);
                }
                else
                {
                    fv = context.Input;
                }
                switch (fvCount)
                {
                case 1:
                    for(j = 0; j < context.StateCount; j++)
                    {
                        means = data->meanValues + j*meanOffset;
                        vars = data->inverseCovariancesForMaxMix8 + j*varOffset;
                        consts = data->gaussianConstants + j*gConstOffset;
                        scores = context.Output + j*fvCount;

                        gmmKernel->GMM8_MAXMIX_G1(fv, means, vars, consts, maxScore, fvLength, mixCount, scores);
                    }
                    break;
                case 2:
                    for(j = 0; j < context.StateCount; j++)
                    {
                        means = data->meanValues + j*meanOffset;
                        vars = data->inverseCovariancesForMaxMix8 + j*varOffset;
                        consts = data->gaussianConstants + j*gConstOffset;
                        scores = context.Output + j*fvCount;

                        gmmKernel->GMM8_MAXMIX_G2(fv, means, vars, consts, maxScore, fvLength, mixCount, scores);
                    }
                    break;
                case 3:

                    for(j = 0; j < context.StateCount; j++)
                    {
                        means = data->meanValues + j*meanOffset;
                        vars = data->inverseCovariancesForMaxMix8 + j*varOffset;
                        consts = data->gaussianConstants + j*gConstOffset;
                        scores = context.Output + j*fvCount;

                        gmmKernel->GMM8_MAXMIX_G3(fv, means, vars, consts, maxScore, fvLength, mixCount, scores);
                        scores += fvCount;

                        means += meanOffset;
                        vars += varOffset;
                        consts += gConstOffset;
                    }
                    break;
                case 4:

                    for(j = 0; j < context.StateCount; j++)
                    {
                        means = data->meanValues + j*meanOffset;
                        vars = data->inverseCovariancesForMaxMix8 + j*varOffset;
                        consts = data->gaussianConstants + j*gConstOffset;
                        scores = context.Output + j*fvCount;

                        gmmKernel->GMM8_MAXMIX_G4(fv, means, vars, consts, maxScore, fvLength, mixCount, scores);
                        scores += fvCount;

                        means += meanOffset;
                        vars += varOffset;
                        consts += gConstOffset;
                    }
                    break;
                case 5:

                    for(j = 0; j < context.StateCount; j++)
                    {
                        means = data->meanValues + j*meanOffset;
                        vars = data->inverseCovariancesForMaxMix8 + j*varOffset;
                        consts = data->gaussianConstants + j*gConstOffset;
                        scores = context.Output + j*fvCount;

                        gmmKernel->GMM8_MAXMIX_G5(fv, means, vars, consts, maxScore, fvLength, mixCount, scores);
                    }
                    break;
                case 6:

                    for(j = 0; j < context.StateCount; j++)
                    {
                        means = data->meanValues + j*meanOffset;
                        vars = data->inverseCovariancesForMaxMix8 + j*varOffset;
                        consts = data->gaussianConstants + j*gConstOffset;
                        scores = context.Output + j*fvCount;

                        gmmKernel->GMM8_MAXMIX_G6(fv, means, vars, consts, maxScore, fvLength, mixCount, scores);
                    }
                    break;
                case 7:

                    for(j = 0; j < context.StateCount; j++)
                    {
                        means = data->meanValues + j*meanOffset;
                        vars = data->inverseCovariancesForMaxMix8 + j*varOffset;
                        consts = data->gaussianConstants + j*gConstOffset;
                        scores = context.Output + j*fvCount;

                        gmmKernel->GMM8_MAXMIX_G7(fv, means, vars, consts, maxScore, fvLength, mixCount, scores);
                    }
                    break;
                case 8:

                    for(j = 0; j < context.StateCount; j++)
                    {
                        means = data->meanValues + j*meanOffset;
                        vars = data->inverseCovariancesForMaxMix8 + j*varOffset;
                        consts = data->gaussianConstants + j*gConstOffset;
                        scores = context.Output + j*fvCount;

                        gmmKernel->GMM8_MAXMIX_G8(fv, means, vars, consts, maxScore, fvLength, mixCount, scores);
                    }
                    break;
                }
            } //else //if(GNA_SW_SSE4_2 == accel || GNA_SW_AVX1 == accel || GNA_SW_AVX2 == accel)
        }//else if (gmm.mode == GNA_MAXMIX8)
        else if (gmm->Config.mode == GNA_MAXMIX16)
        {
            // auxiliary pointers
            uint16_t *vars;

            for(j = 0; j < context.StateCount; j++)
            {
                fv = context.Input;
                means = data->meanValues + j*meanOffset;
                vars = data->inverseCovariancesForMaxMix16 + j*varOffset;
                consts = data->gaussianConstants + j*gConstOffset;
                scores = context.Output + j*fvCount;

                for(i = 0; i < fvCount; i++)
                {
                    *scores = gmmKernel->GMM16(fv, means, vars, consts, maxScore, fvLength, mixCount);
                    scores++;
                    fv += fvLength;
                }
            }
        }//else if (gmm.mode == MAXMIX16)
    }
    else // has active list
    {
        if (gmm->Config.mode == GNA_MAXMIX8)
        {
            uint8_t *vars; // auxiliary pointer
            if(GNA_GEN_FAST == accel || GNA_GEN_SAT == accel)
            {
                for(j = 0; j < context.StateCount; j++)
                {
                    k = context.ActiveList->Indices[j];
                    means = data->meanValues + k * meanSetOffsetSize;
                    vars = data->inverseCovariancesForMaxMix8 + k * varSetOffsetSize;
                    consts = (uint32_t*)((uint8_t*)data->gaussianConstants + k * gConstSetOffsetSize);
                    fv = context.Input;
                    scores = context.Output + j*fvCount;

                    for(i = 0; i < fvCount; i++)
                    {
                        *scores = gmmKernel->GMM8(fv, means, vars, consts, maxScore, fvLength, mixCount);
                        scores++;
                        fv += fvLength;
                    }
                }
            }
            else //if(GNA_SW_SSE4_2 == accel || GNA_SW_AVX1 == accel || GNA_SW_AVX2 == accel)
            {
                uint8_t  pFeatureBuffer[GMM_FV_ELEMENT_COUNT_MAX*GMM_FV_COUNT_MAX+GMM_FV_MEM_ALIGN]; // max data size
                fv = (uint8_t*)(((unsigned long long)pFeatureBuffer+GMM_FV_MEM_ALIGN) & 0xffffffffffffffc0ull);
                uint32_t n = 0;
                uint32_t g = 0;

                // pack feature vectors by 8 features
                // v0[0..7]v1[0..7]vj[0..7]v0[8..15]v1[8..15]...
                if (fvCount > 1)
                {
                    for (n = 0; n < fvLength; n += GMM_FV_COUNT_MAX)
                    {
                        for (g = 0; g < fvCount; g++)
                        {
                            *((uint64_t*)fv) = *((uint64_t*)((context.Input) + g * fvLength + n));
                            fv += GMM_FV_COUNT_MAX;
                        }
                    }
                    fv = (uint8_t*)(((unsigned long long)pFeatureBuffer+GMM_FV_MEM_ALIGN) & 0xffffffffffffffc0ull);
                }
                else
                {
                    fv = context.Input;
                }

                switch(fvCount)
                {
                case 1:

                    for(j = 0; j < context.StateCount; j++)
                    {
                        k = context.ActiveList->Indices[j];
                        means = data->meanValues + k * meanSetOffsetSize;
                        vars = data->inverseCovariancesForMaxMix8 + k * varSetOffsetSize;
                        consts = (uint32_t*)((uint8_t*)data->gaussianConstants + k * gConstSetOffsetSize);
                        scores = context.Output + j*fvCount;

                        gmmKernel->GMM8_MAXMIX_G1(fv, means, vars, consts, maxScore, fvLength, mixCount, scores);
                    }
                    break;
                case 2:

                    for(j = 0; j < context.StateCount; j++)
                    {
                        k = context.ActiveList->Indices[j];
                        means = data->meanValues + k * meanSetOffsetSize;
                        vars = data->inverseCovariancesForMaxMix8 + k * varSetOffsetSize;
                        consts = (uint32_t*)((uint8_t*)data->gaussianConstants + k * gConstSetOffsetSize);
                        scores = context.Output + j*fvCount;

                        gmmKernel->GMM8_MAXMIX_G2(fv, means, vars, consts, maxScore, fvLength, mixCount, scores);
                    }
                    break;
                case 3:

                    for(j = 0; j < context.StateCount; j++)
                    {
                        k = context.ActiveList->Indices[j];
                        means = data->meanValues + k * meanSetOffsetSize;
                        vars = data->inverseCovariancesForMaxMix8 + k * varSetOffsetSize;
                        consts = (uint32_t*)((uint8_t*)data->gaussianConstants + k * gConstSetOffsetSize);
                        scores = context.Output + j*fvCount;

                        gmmKernel->GMM8_MAXMIX_G3(fv, means, vars, consts, maxScore, fvLength, mixCount, scores);
                    }
                    break;
                case 4:

                    for(j = 0; j < context.StateCount; j++)
                    {
                        k = context.ActiveList->Indices[j];
                        means = data->meanValues + k * meanSetOffsetSize;
                        vars = data->inverseCovariancesForMaxMix8 + k * varSetOffsetSize;
                        consts = (uint32_t*)((uint8_t*)data->gaussianConstants + k * gConstSetOffsetSize);
                        scores = (uint32_t*)context.Output + j*fvCount;

                        gmmKernel->GMM8_MAXMIX_G4(fv, means, vars, consts, maxScore, fvLength, mixCount, scores);
                    }
                    break;
                case 5:

                    for(j = 0; j < context.StateCount; j++)
                    {
                        k = context.ActiveList->Indices[j];
                        means = data->meanValues + k * meanSetOffsetSize;
                        vars = data->inverseCovariancesForMaxMix8 + k * varSetOffsetSize;
                        consts = (uint32_t*)((uint8_t*)data->gaussianConstants + k * gConstSetOffsetSize);
                        scores = context.Output + j*fvCount;

                        gmmKernel->GMM8_MAXMIX_G5(fv, means, vars, consts, maxScore, fvLength, mixCount, scores);
                    }
                    break;
                case 6:

                    for(j = 0; j < context.StateCount; j++)
                    {
                        k = context.ActiveList->Indices[j];
                        means = data->meanValues + k * meanSetOffsetSize;
                        vars = data->inverseCovariancesForMaxMix8 + k * varSetOffsetSize;
                        consts = (uint32_t*)((uint8_t*)data->gaussianConstants + k * gConstSetOffsetSize);
                        scores = context.Output + j*fvCount;

                        gmmKernel->GMM8_MAXMIX_G6(fv, means, vars, consts, maxScore, fvLength, mixCount, scores);
                    }
                    break;
                case 7:

                    for(j = 0; j < context.StateCount; j++)
                    {
                        k = context.ActiveList->Indices[j];
                        means = data->meanValues + k * meanSetOffsetSize;
                        vars = data->inverseCovariancesForMaxMix8 + k * varSetOffsetSize;
                        consts = (uint32_t*)((uint8_t*)data->gaussianConstants + k * gConstSetOffsetSize);
                        scores = context.Output + j*fvCount;

                        gmmKernel->GMM8_MAXMIX_G7(fv, means, vars, consts, maxScore, fvLength, mixCount, scores);
                    }
                    break;
                case 8:

                    for(j = 0; j < context.StateCount; j++)
                    {
                        k = context.ActiveList->Indices[j];
                        means = data->meanValues + k * meanSetOffsetSize;
                        vars = data->inverseCovariancesForMaxMix8 + k * varSetOffsetSize;
                        consts = (uint32_t*)((uint8_t*)data->gaussianConstants + k * gConstSetOffsetSize);
                        scores = context.Output + j*fvCount;

                        gmmKernel->GMM8_MAXMIX_G8(fv, means, vars, consts, maxScore, fvLength, mixCount, scores);
                    }
                    break;
                }
            } // else //if(GNA_SW_SSE4_2 == accel || GNA_SW_AVX1 == accel || GNA_SW_AVX2 == accel)
        }//else if (gmm.mode == GNA_MAXMIX8)
        else if (gmm->Config.mode == GNA_MAXMIX16)
        {
            uint16_t *vars; // auxiliary pointer
            scores = context.Output;

            for(j = 0; j < context.StateCount; j++)
            {
                k = context.ActiveList->Indices[j];
                means = data->meanValues + k * meanSetOffsetSize;
                vars = data->inverseCovariancesForMaxMix16 + k * varSetOffsetSize / GMM_COVARIANCE_SIZE_MAX;
                consts = (uint32_t*)((uint8_t*)data->gaussianConstants + k * gConstSetOffsetSize);
                fv = context.Input;

                for(i = 0; i < fvCount; i++)
                {
                    *scores = gmmKernel->GMM16(fv, means, vars, consts, maxScore, fvLength, mixCount);
                    scores++;
                    fv += fvLength;
                }
            }
        }//else if (gmm.mode == MAXMIX16)
    }// has active list
    checkScoresSaturation(context.StateCount, fvCount, context.Output, maxScore, nSaturated);
}
