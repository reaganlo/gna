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
#include "ActiveList.h"
#include "RequestHandler.h"
#include "ThreadPool.h"

using namespace GNA;

AcceleratorSw::AcceleratorSw(acceleration nProcessorType)
    :Accelerator(nProcessorType)
{
    status_t sts = init();
    if (GNA_SUCCESS != sts)
    {
        throw GnaException(sts);
    }
}

status_t AcceleratorSw::init()
{
    //return kd.Init(accMode);
    return GNA_SUCCESS;
}

status_t AcceleratorSw::submit(Request* r)
{
    Sw* sw  = nullptr;

    //r->handle = new Sw();
    //if (nullptr == r->handle)
    //{
    //    return GNA_ERR_RESOURCES;
    //}
    //sw  = r->handle;

    return GNA_SUCCESS;
}

status_t AcceleratorSw::Wait(Request *r, uint32_t timeout, perf_t* perfResults)
{
    status_t    status = GNA_SUCCESS;
    status_t    status2 = GNA_SUCCESS;
    // FIXME: do sth about the handle
    Sw*         sw = nullptr; // r->handle;

    // wait for threaded request r to finish with timeout
    auto future_status = sw->handle.wait_for(std::chrono::milliseconds(timeout));
    if (future_status == std::future_status::deferred || future_status == std::future_status::timeout)
    {
        status = GNA_DEVICEBUSY;
    }
    else if (future_status == std::future_status::ready)
    {
        status = sw->handle.get();
        // finish profiling
        profilerDTscStop(&r->profiler.process);
        // save profiling results to app provided space
#ifdef PROFILE
        if (nullptr != perfResults)
        {
            perfResults->lib.preprocess = r->profiler.preprocess.passed;
            perfResults->lib.process = r->profiler.process.passed;
            perfResults->lib.submit = r->profiler.submit.passed;
            perfResults->lib.scoring = r->profiler.scoring.passed;
            perfResults->lib.total = r->profiler.total.passed;
            perfResults->total.start = r->profiler.submit.start;
            perfResults->total.stop = r->profiler.process.stop;
        }
#endif // PROFILE

        // TODO: should be done by Device upon status return
        // remove request from queue
        //status2 = requestHandler.removeRequest(r);
        //if (GNA_SUCCESS != status2) status = status2; // notify if dequeue error occurred
    }

    return status;
}

status_t AcceleratorSw::checkScoresSaturation(uint32_t nGMMs, uint32_t nVectors, uint32_t *pS, uint32_t maxScore)
{
    uint32_t i = 0;

    for(i = 0; i < nGMMs * nVectors; i++)
    {
        if (maxScore == *pS)
        {
            return GNA_SSATURATE;
        }
        pS++;
    }

    return GNA_SUCCESS;
}

status_t AcceleratorSw::gmmSoftwareKernel(GmmLayer* gmm, req_profiler* profiler)
{
    const uint8_t * const input = static_cast<const uint8_t * const>(gmm->Input.Buffer); // TODO:KJ:use request config outputs (only if defined for current layer)
    uint32_t *output = (uint32_t*)(gmm->Output.Buffer); // TODO:KJ:use request config outputs (only if defined for current layer)
    ActiveList activeList(0, 0, nullptr);// TODO:KJ:use request active list (only if defined for current layer)
    const gna_gmm_data* data = &gmm->Data;
    const uint32_t fvCount = gmm->Input.VectorCount;
    const uint32_t fvLength = gmm->Input.ElementCount;
    const uint32_t mixCount = gmm->Config.mixtureComponentCount;
    const uint32_t meanSetOffsetSize = gmm->Params.MeanSetOffsetSize;
    const uint32_t varSetOffsetSize = gmm->Params.VarSetOffsetSize;
    const uint32_t gConstSetOffsetSize = gmm->Params.GaussConstSetOffsetSize;
    const uint32_t maxScore = gmm->Config.maximumScore;
    
    uint32_t stateCount = 0; // no of GMM states or active indices when applicable
    uint32_t i, j, k;
    status_t s = GNA_SUCCESS;
    // auxiliary pointers
    uint8_t const *fv;
    uint8_t *means;
    uint32_t *consts;
    uint32_t *scores;

    profilerDTscAStart(&profiler->scoring);

    if (false == activeList.enabled)
    {
        const uint32_t meanOffset = meanSetOffsetSize / GMM_MEAN_VALUE_SIZE;;
        const uint32_t varOffset = varSetOffsetSize / (gmm->Config.mode + 1);;
        const uint32_t gConstOffset = gConstSetOffsetSize / GMM_CONSTANTS_SIZE;
        stateCount = gmm->Config.stateCount;

        if (gmm->Config.mode == GNA_MAXMIX8)
        {
            uint8_t *vars; // auxiliary pointer
            //if(GNA_GEN_FAST == kd.accel || GNA_GEN_SAT == kd.accel)
            {

                for(j = 0; j < stateCount; j++)
                {
                    fv = input;
                    means = data->meanValues + j*meanOffset;
                    vars = data->inverseCovariancesForMaxMix8 + j*varOffset;
                    consts = data->gaussianConstants + j*gConstOffset;
                    scores = output + j*fvCount;

                    for(i = 0; i < fvCount; i++)
                    {
                        *scores = kd.gmms->GMM8(fv, means, vars, consts, maxScore, fvLength, mixCount);
                        scores++;
                        fv += fvLength;
                    }
                }
            }
            //else //if(GNA_SW_SSE4_2 == kd.accel || GNA_SW_AVX1 == kd.accel || GNA_SW_AVX2 == kd.accel)
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
                            *((uint64_t*)fv) = *((uint64_t*)((input) + g * fvLength + n));
                            fv += GMM_FV_COUNT_MAX;
                        }
                    }
                    fv = (uint8_t*)(((unsigned long long)pFeatureBuffer+GMM_FV_MEM_ALIGN) & 0xffffffffffffffc0ull);        
                }
                else
                {
                    fv = input;
                }
                switch (fvCount)
                {
                case 1:
                    for(j = 0; j < stateCount; j++)
                    {                    
                        means = data->meanValues + j*meanOffset;
                        vars = data->inverseCovariancesForMaxMix8 + j*varOffset;
                        consts = data->gaussianConstants + j*gConstOffset;
                        scores = output + j*fvCount;

                        kd.gmms->GMM8_MAXMIX_G1(fv, means, vars, consts, maxScore, fvLength, mixCount, scores);
                    }
                    break;
                case 2:
                    for(j = 0; j < stateCount; j++)
                    {
                        means = data->meanValues + j*meanOffset;
                        vars = data->inverseCovariancesForMaxMix8 + j*varOffset;
                        consts = data->gaussianConstants + j*gConstOffset;
                        scores = output + j*fvCount;

                        kd.gmms->GMM8_MAXMIX_G2(fv, means, vars, consts, maxScore, fvLength, mixCount, scores);
                    }            
                    break;
                case 3:

                    for(j = 0; j < stateCount; j++)
                    {
                        means = data->meanValues + j*meanOffset;
                        vars = data->inverseCovariancesForMaxMix8 + j*varOffset;
                        consts = data->gaussianConstants + j*gConstOffset;
                        scores = output + j*fvCount;

                        kd.gmms->GMM8_MAXMIX_G3(fv, means, vars, consts, maxScore, fvLength, mixCount, scores);
                        scores += fvCount;

                        means += meanOffset;
                        vars += varOffset;
                        consts += gConstOffset;
                    }            
                    break;
                case 4:

                    for(j = 0; j < stateCount; j++)
                    {
                        means = data->meanValues + j*meanOffset;
                        vars = data->inverseCovariancesForMaxMix8 + j*varOffset;
                        consts = data->gaussianConstants + j*gConstOffset;
                        scores = output + j*fvCount;

                        kd.gmms->GMM8_MAXMIX_G4(fv, means, vars, consts, maxScore, fvLength, mixCount, scores);
                        scores += fvCount;

                        means += meanOffset;
                        vars += varOffset;
                        consts += gConstOffset;
                    }            
                    break;
                case 5:

                    for(j = 0; j < stateCount; j++)
                    {
                        means = data->meanValues + j*meanOffset;
                        vars = data->inverseCovariancesForMaxMix8 + j*varOffset;
                        consts = data->gaussianConstants + j*gConstOffset;
                        scores = output + j*fvCount;

                        kd.gmms->GMM8_MAXMIX_G5(fv, means, vars, consts, maxScore, fvLength, mixCount, scores);
                    }            
                    break;
                case 6:

                    for(j = 0; j < stateCount; j++)
                    {
                        means = data->meanValues + j*meanOffset;
                        vars = data->inverseCovariancesForMaxMix8 + j*varOffset;
                        consts = data->gaussianConstants + j*gConstOffset;
                        scores = output + j*fvCount;

                        kd.gmms->GMM8_MAXMIX_G6(fv, means, vars, consts, maxScore, fvLength, mixCount, scores);
                    }            
                    break;
                case 7:

                    for(j = 0; j < stateCount; j++)
                    {
                        means = data->meanValues + j*meanOffset;
                        vars = data->inverseCovariancesForMaxMix8 + j*varOffset;
                        consts = data->gaussianConstants + j*gConstOffset;
                        scores = output + j*fvCount;

                        kd.gmms->GMM8_MAXMIX_G7(fv, means, vars, consts, maxScore, fvLength, mixCount, scores);
                    }            
                    break;
                case 8:

                    for(j = 0; j < stateCount; j++)
                    {
                        means = data->meanValues + j*meanOffset;
                        vars = data->inverseCovariancesForMaxMix8 + j*varOffset;
                        consts = data->gaussianConstants + j*gConstOffset;
                        scores = output + j*fvCount;

                        kd.gmms->GMM8_MAXMIX_G8(fv, means, vars, consts, maxScore, fvLength, mixCount, scores);
                    }            
                    break;                
                }                    
            } //else //if(GNA_SW_SSE4_2 == kd.accel || GNA_SW_AVX1 == kd.accel || GNA_SW_AVX2 == kd.accel)
        }//else if (gmm->mode == GNA_MAXMIX8)
        else if (gmm->Config.mode == GNA_MAXMIX16)
        {
            // auxiliary pointers
            uint16_t *vars;

            for(j = 0; j < stateCount; j++)
            {
                fv = input;
                means = data->meanValues + j*meanOffset;
                vars = data->inverseCovariancesForMaxMix16 + j*varOffset;
                consts = data->gaussianConstants + j*gConstOffset;
                scores = output + j*fvCount;

                for(i = 0; i < fvCount; i++)
                {
                    *scores = kd.gmms->GMM16(fv, means, vars, consts, maxScore, fvLength, mixCount);
                    scores++;
                    fv += fvLength;
                }

                means += meanOffset;
                vars += varOffset;
                consts += gConstOffset;
            }
        }//else if (gmm->mode == MAXMIX16)
    }
    else // has active list
    {
        stateCount = activeList.indicesCount;

        if (gmm->Config.mode == GNA_MAXMIX8)
        {
            uint8_t *vars; // auxiliary pointer
            //if(GNA_GEN_FAST == kd.accel || GNA_GEN_SAT == kd.accel)
            {
                for(j = 0; j < stateCount; j++)
                {
                    k = activeList.indices[j];
                    means = data->meanValues + k * meanSetOffsetSize;
                    vars = data->inverseCovariancesForMaxMix8 + k * varSetOffsetSize;
                    consts = (uint32_t*)((uint8_t*)data->gaussianConstants + k * gConstSetOffsetSize);
                    fv = input;
                    scores = output + j*fvCount;

                    for(i = 0; i < fvCount; i++)
                    {
                        *scores = kd.gmms->GMM8(fv, means, vars, consts, maxScore, fvLength, mixCount);
                        scores++;
                        fv += fvLength;
                    }
                }
            }
            //else //if(GNA_SW_SSE4_2 == kd.accel || GNA_SW_AVX1 == kd.accel || GNA_SW_AVX2 == kd.accel)
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
                            *((uint64_t*)fv) = *((uint64_t*)((input) + g * fvLength + n));
                            fv += GMM_FV_COUNT_MAX;
                        }
                    }
                    fv = (uint8_t*)(((unsigned long long)pFeatureBuffer+GMM_FV_MEM_ALIGN) & 0xffffffffffffffc0ull);
                }
                else
                {
                    fv = input;
                }

                switch(fvCount)
                {
                case 1:

                    for(j = 0; j < stateCount; j++)
                    {
                        k = activeList.indices[j];
                        means = data->meanValues + k * meanSetOffsetSize;
                        vars = data->inverseCovariancesForMaxMix8 + k * varSetOffsetSize;
                        consts = (uint32_t*)((uint8_t*)data->gaussianConstants + k * gConstSetOffsetSize);
                        scores = output + j*fvCount;

                        kd.gmms->GMM8_MAXMIX_G1(fv, means, vars, consts, maxScore, fvLength, mixCount, scores);
                    }
                    break;
                case 2:

                    for(j = 0; j < stateCount; j++)
                    {
                        k = activeList.indices[j];
                        means = data->meanValues + k * meanSetOffsetSize;
                        vars = data->inverseCovariancesForMaxMix8 + k * varSetOffsetSize;
                        consts = (uint32_t*)((uint8_t*)data->gaussianConstants + k * gConstSetOffsetSize);
                        scores = output + j*fvCount;

                        kd.gmms->GMM8_MAXMIX_G2(fv, means, vars, consts, maxScore, fvLength, mixCount, scores);
                    }
                    break;
                case 3:

                    for(j = 0; j < stateCount; j++)
                    {
                        k = activeList.indices[j];
                        means = data->meanValues + k * meanSetOffsetSize;
                        vars = data->inverseCovariancesForMaxMix8 + k * varSetOffsetSize;
                        consts = (uint32_t*)((uint8_t*)data->gaussianConstants + k * gConstSetOffsetSize);
                        scores = output + j*fvCount;

                        kd.gmms->GMM8_MAXMIX_G3(fv, means, vars, consts, maxScore, fvLength, mixCount, scores);
                    }
                    break;
                case 4:

                    for(j = 0; j < stateCount; j++)
                    {
                        k = activeList.indices[j];
                        means = data->meanValues + k * meanSetOffsetSize;
                        vars = data->inverseCovariancesForMaxMix8 + k * varSetOffsetSize;
                        consts = (uint32_t*)((uint8_t*)data->gaussianConstants + k * gConstSetOffsetSize);
                        scores = (uint32_t*)output + j*fvCount;

                        kd.gmms->GMM8_MAXMIX_G4(fv, means, vars, consts, maxScore, fvLength, mixCount, scores);
                    }
                    break;
                case 5:

                    for(j = 0; j < stateCount; j++)
                    {
                        k = activeList.indices[j];
                        means = data->meanValues + k * meanSetOffsetSize;
                        vars = data->inverseCovariancesForMaxMix8 + k * varSetOffsetSize;
                        consts = (uint32_t*)((uint8_t*)data->gaussianConstants + k * gConstSetOffsetSize);
                        scores = output + j*fvCount;

                        kd.gmms->GMM8_MAXMIX_G5(fv, means, vars, consts, maxScore, fvLength, mixCount, scores);
                    }
                    break;
                case 6:

                    for(j = 0; j < stateCount; j++)
                    {
                        k = activeList.indices[j];
                        means = data->meanValues + k * meanSetOffsetSize;
                        vars = data->inverseCovariancesForMaxMix8 + k * varSetOffsetSize;
                        consts = (uint32_t*)((uint8_t*)data->gaussianConstants + k * gConstSetOffsetSize);
                        scores = output + j*fvCount;

                        kd.gmms->GMM8_MAXMIX_G6(fv, means, vars, consts, maxScore, fvLength, mixCount, scores);
                    }
                    break;
                case 7:

                    for(j = 0; j < stateCount; j++)
                    {
                        k = activeList.indices[j];
                        means = data->meanValues + k * meanSetOffsetSize;
                        vars = data->inverseCovariancesForMaxMix8 + k * varSetOffsetSize;
                        consts = (uint32_t*)((uint8_t*)data->gaussianConstants + k * gConstSetOffsetSize);
                        scores = output + j*fvCount;

                        kd.gmms->GMM8_MAXMIX_G7(fv, means, vars, consts, maxScore, fvLength, mixCount, scores);
                    }
                    break;
                case 8:

                    for(j = 0; j < stateCount; j++)
                    {
                        k = activeList.indices[j];
                        means = data->meanValues + k * meanSetOffsetSize;
                        vars = data->inverseCovariancesForMaxMix8 + k * varSetOffsetSize;
                        consts = (uint32_t*)((uint8_t*)data->gaussianConstants + k * gConstSetOffsetSize);
                        scores = output + j*fvCount;

                        kd.gmms->GMM8_MAXMIX_G8(fv, means, vars, consts, maxScore, fvLength, mixCount, scores);
                    }
                    break;                    
                }
            } // else //if(GNA_SW_SSE4_2 == kd.accel || GNA_SW_AVX1 == kd.accel || GNA_SW_AVX2 == kd.accel)
        }//else if (gmm->mode == GNA_MAXMIX8)
        else if (gmm->Config.mode == GNA_MAXMIX16)
        {
            uint16_t *vars; // auxiliary pointer
            scores = output;
      
            for(j = 0; j < stateCount; j++)
            {
                k = activeList.indices[j];
                means = data->meanValues + k * meanSetOffsetSize;
                vars = data->inverseCovariancesForMaxMix16 + k * varSetOffsetSize / GMM_COVARIANCE_SIZE_MAX;
                consts = (uint32_t*)((uint8_t*)data->gaussianConstants + k * gConstSetOffsetSize);
                fv = input;

                for(i = 0; i < fvCount; i++)
                {
                    *scores = kd.gmms->GMM16(fv, means, vars, consts, maxScore, fvLength, mixCount);
                    scores++;
                    fv += fvLength;
                }
            }
        }//else if (gmm->mode == MAXMIX16)
    }// has active list
    s = checkScoresSaturation(stateCount, fvCount, output, maxScore);

    profilerDTscAStop(&profiler->scoring);
    profilerDTscAStop(&profiler->total);
    return s;
}

status_t AcceleratorSw::xnnSoftwareKernel(SoftwareModel* model, req_profiler* profiler)
{
    status_t    sts     = GNA_SUCCESS;
    uint32_t    i       = 0;
    nn_layer* lyr     = nullptr;
    uint32_t    nOuts   = 0;        // # of outputs (all or active list indices)
    uint32_t*   al      = 0;        // active list indices data
    uint32_t    sat     = 0;        // scoring saturation counter

    profilerDTscAStart(&profiler->scoring);

    // TODO: add thread buffer as input to calculation function
    aligned_fv_bufs* fvBuffers = nullptr; // threadPool.getThreadBuffer();

    for (; i < model->layerCount; i++)
    {
        lyr = const_cast<nn_layer*>(&model->Layers[i]->sourceLayer);
        Validate::IsNull(lyr);
        if (INTEL_AFFINE == lyr->nLayerKind)
        {
            //// use active list in last layer if available
            //if ((i == model->layerCount - 1) && model->activeList.enabled)
            //{
            //    nOuts   = model->activeList.indicesCount;
            //    al      = (uint32_t*)model->activeList.indices;
            //}
            //else // regular outputs
            //{
                nOuts   = lyr->nOutputRows;
                al      = nullptr;
            //}
            sts = kd.xnns->affine(lyr, al, nOuts, &sat, fvBuffers);
            if (GNA_SUCCESS != sts) return sts;
            if (0 != (&((nn_layer_affine*)lyr->pLayerStruct)->pwl)->nSegments)
            {
                kd.xnns->pwl(lyr, 0, nOuts - 1, 0, lyr->nInputColumns - 1, &sat, fvBuffers->pwl);
            }
        }
        else if (INTEL_AFFINE_MULTIBIAS == lyr->nLayerKind)
        {
            sts = kd.xnns->affineMbias(lyr, al, nOuts, &sat, fvBuffers);
            if (GNA_SUCCESS != sts) return sts;
            if (0 != (&((nn_layer_affine*)lyr->pLayerStruct)->pwl)->nSegments)
            {
                kd.xnns->pwl(lyr, 0, lyr->nOutputRows - 1, 0, lyr->nInputColumns - 1, &sat, fvBuffers->pwl);
            }
        }
        else if (lyr->nLayerKind == INTEL_AFFINE_DIAGONAL)
        {
            sts = kd.xnns->diagonal(lyr, &sat);
            if (GNA_SUCCESS != sts) return sts;
            if (0 != (&((nn_layer_affine*)lyr->pLayerStruct)->pwl)->nSegments)
            {
                kd.xnns->pwl(lyr, 0, lyr->nOutputRows - 1, 0, lyr->nInputColumns - 1, &sat, fvBuffers->pwl);
            }
        }
        else if (lyr->nLayerKind == INTEL_RECURRENT)
        {
            sts = kd.xnns->recurrent(lyr, &sat, fvBuffers->pwl);
        }
        else if (INTEL_INTERLEAVE   == lyr->nLayerKind || 
                 INTEL_DEINTERLEAVE == lyr->nLayerKind)
        {
            sts = kd.xnns->transpose(lyr);
        }
        else if (lyr->nLayerKind == INTEL_COPY)
        {
            sts = kd.xnns->copy(lyr);
        }
        else if (lyr->nLayerKind == INTEL_CONVOLUTIONAL)
        {
            sts = kd.xnns->conv(lyr, &sat, fvBuffers->pwl, fvBuffers->pool);
            if (sts != GNA_SUCCESS)return sts;
        }
        else
        {
            sts = XNN_ERR_LYR_TYPE;
        }

        if (GNA_SUCCESS != sts) return sts;
    }

    profilerDTscAStop(&profiler->scoring);
    profilerDTscAStop(&profiler->total);

    if (sat > 0)
    {
        return GNA_SSATURATE;
    }
    return sts;
}
