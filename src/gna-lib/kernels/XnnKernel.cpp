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

#include "XnnKernelApi.h"

#include <string.h>

#include "convnet.h"
#include "igemv16.h"
#include "igemv8.h"
#include "KernelMacros.h"
#include "pwl.h"

namespace GNA
{

inline int32_t* getOutputBuffer(const intel_pwl_func_t * const pwl, const nn_layer* const layer)
{
    if (IsActivationFunctionEnabled(pwl))
    {
        return static_cast<int32_t*>(layer->pOutputsIntermediate);
    }
    else
    {
         return static_cast<int32_t*>(layer->pOutputs);
    }
}

#define CONVOLUTIONAL_ROW_STRIDE 1

#define GNAApplyPiecewiseLinearTransform    KERNEL(GNAApplyPiecewiseLinearTransform)
#define GNAApplyAffineTransform             KERNEL(GNAApplyAffineTransform)
#define GNAApplyAffineMBiasTransform        KERNEL(GNAApplyAffineMBiasTransform)
#define GNAApplyDiagonalTransform           KERNEL(GNAApplyDiagonalTransform)
#define GNAApplyRecurrentTransform          KERNEL(GNAApplyRecurrentTransform)
#define GNAApplyTranspose                   KERNEL(GNAApplyTranspose)
#define GNAApplyCopy                        KERNEL(GNAApplyCopy)
#define GNAApplyConvolutionalTransform      KERNEL(GNAApplyConvolutionalTransform)

void GNAApplyPiecewiseLinearTransform(
    const nn_layer*     pLayer,
          uint32_t      nRowBegin,
          uint32_t      nRowEnd,
          uint32_t      nColBegin,
          uint32_t      nColEnd,
          uint32_t*     nSaturated,
          void*         pwlBuff)
{
    nn_pwl_seg* segments = NULL;
    pwl_params* params;

    params = (pwl_params*)pwlBuff;

    if (pLayer->nLayerKind != INTEL_CONVOLUTIONAL)
    {
        segments = (&((nn_layer_affine*)pLayer->pLayerStruct)->pwl)->pSegments;
        params->NS = (uint8_t)(&((nn_layer_affine*)pLayer->pLayerStruct)->pwl)->nSegments;
    }
    else
    {
        segments = (&((nn_layer_conv*)pLayer->pLayerStruct)->pwl)->pSegments;
        params->NS = (uint8_t)(&((nn_layer_conv*)pLayer->pLayerStruct)->pwl)->nSegments;
    }

    PwlSetup(params, nRowBegin, nRowEnd, nColBegin, nColEnd,
        pLayer->nOutputColumns, nSaturated, (int32_t*)pLayer->pOutputsIntermediate,
        (int16_t*)pLayer->pOutputs, segments);

    params->pwlAll(params);
}

void GNAApplyAffineTransform(
    const nn_layer*          pLayer,
    const uint32_t*          pActiveIndices,
          uint32_t           nActiveIndices,
          uint32_t*          nSaturated,
          KernelBuffers*     fvBuffers)
{
    nn_layer_affine* aff = (nn_layer_affine*)pLayer->pLayerStruct;
    uint32_t         m = pLayer->nOutputRows;
    uint32_t         n = pLayer->nInputColumns;
    uint32_t         k = pLayer->nInputRows;
    void*            w = aff->affine.pWeights;
    int16_t*         in = (int16_t*)pLayer->pInputs;
    int32_t*         out = getOutputBuffer(&aff->pwl, pLayer);

    if (aff->affine.nBytesPerWeight == 1)
    {
        nn_bias_c *bias = (nn_bias_c*)aff->affine.pBiases;
        if (NULL == pActiveIndices)
        {
            igemm8(m, n, k, in, (int8_t*)w, bias, out, nSaturated, fvBuffers);
        }
        else
        {
            igemm8_subset(m, n, k, in, (int8_t*)w, bias, out,
                pActiveIndices, nActiveIndices, nSaturated, fvBuffers);
        }
    }
    else if (aff->affine.nBytesPerWeight == 2)
    {
        nn_bias_s *bias = (nn_bias_s*)aff->affine.pBiases;
        if (pActiveIndices == NULL)
        {
            igemm16(m, n, k, in, (int16_t*)w, bias, out, nSaturated, fvBuffers);
        }
        else
        {
            igemm16_subset(m, n, k, in, (int16_t*)w, bias, out,
                pActiveIndices, nActiveIndices, nSaturated, fvBuffers);
        }
    }
}

void GNAApplyAffineMBiasTransform(
    const nn_layer*          pLayer,
    const uint32_t*          pActiveIndices,
          uint32_t           nActiveIndices,
          uint32_t*          nSaturated,
          KernelBuffers*     fvBuffers)
{
    nn_layer_affine_multi* aff = static_cast<nn_layer_affine_multi*>(pLayer->pLayerStruct);
    uint32_t        m = pLayer->nOutputRows;
    uint32_t        n = pLayer->nInputColumns;
    uint32_t        k = pLayer->nInputRows;
    void*           w = aff->affine.pWeights;
    int16_t* in = static_cast<int16_t*>(pLayer->pInputs);
    int32_t* out = getOutputBuffer(&aff->pwl, pLayer);

    if (aff->affine.nBytesPerWeight == 1)
    {
        nn_bias_c *cbias = aff->affine.weightScaleFactors;
        nn_bias_s *bias = aff->affine.pBiases + aff->affine.biasVectorIndex;
        int8_t* w_8 = static_cast<int8_t*>(w);
        if (pActiveIndices == NULL)
        {
            igemm8_mb(m, n, k, in, w_8, bias, aff->affine.biasVectorCount, cbias, out, nSaturated, fvBuffers);
        }
        else
        {
            igemm8_subset_mb(m, n, k, in, w_8, bias, aff->affine.biasVectorCount, cbias, out,
                pActiveIndices, nActiveIndices, nSaturated, fvBuffers);
        }
    }
    else if (aff->affine.nBytesPerWeight == 2)
    {
        nn_bias_s *bias = aff->affine.pBiases + aff->affine.biasVectorIndex;
        int16_t* w_16 = static_cast<int16_t*>(w);
        if (pActiveIndices == NULL)
        {
            igemm16_mb(m, n, k, in, w_16, bias, aff->affine.biasVectorCount, out, nSaturated, fvBuffers);
        }
        else
        {
            igemm16_subset_mb(m, n, k, in, w_16, bias, aff->affine.biasVectorCount, out,
                pActiveIndices, nActiveIndices, nSaturated, fvBuffers);
        }
    }
}

void GNAApplyDiagonalTransform(
    const nn_layer*     pLayer,
          uint32_t*     nSaturated)
{
    nn_layer_affine* aff = (nn_layer_affine*)pLayer->pLayerStruct;
    uint32_t         m = pLayer->nOutputRows;
    uint32_t         n = pLayer->nInputColumns;
    void*            w = aff->affine.pWeights;
    int16_t*         in = (int16_t*)pLayer->pInputs;
    int32_t*         out = getOutputBuffer(&aff->pwl, pLayer);

    if (aff->affine.nBytesPerWeight == 1)
    {
        nn_bias_c *bias = (nn_bias_c*)aff->affine.pBiases;
        isbmm8(m, n, (int8_t*)w, in, bias, out, nSaturated);
    }
    else if (aff->affine.nBytesPerWeight == 2)
    {
        nn_bias_s *bias = (int32_t*)aff->affine.pBiases;
        isbmm16(m, n, (int16_t*)w, in, bias, out, nSaturated);
    }
}

void GNAApplyRecurrentTransform(
    const nn_layer*     pLayer,
          uint32_t*     nSaturated,
          void*         pwlBuff)
{
    nn_layer_reccurent*   rnn = (nn_layer_reccurent*)pLayer->pLayerStruct;
    uint32_t        k = pLayer->nInputColumns;
    uint32_t        m = pLayer->nOutputColumns;
    void*           w = rnn->affine.pWeights;
    int16_t*        in;
    int16_t*        fb;
    int32_t*        out;

    // for each input vector
    for (uint32_t i = 0; i < pLayer->nInputRows; i++)
    {
        in = (int16_t*)pLayer->pInputs + i * k;
        fb = (int16_t*)rnn->pFeedbackBuffer + (i*m);
        out = (int32_t*)pLayer->pOutputsIntermediate + i * m;

        if (rnn->affine.nBytesPerWeight == 1)
        {
            nn_bias_c *B = (nn_bias_c*)rnn->affine.pBiases;
            igemv8(m, k, in, fb, (int8_t*)w, B, out, nSaturated);
        }
        else if (rnn->affine.nBytesPerWeight == 2)
        {
            nn_bias_s *B = (nn_bias_s*)rnn->affine.pBiases;
            igemv16(m, k, in, fb, (int16_t*)w, B, out, nSaturated);
        }
        GNAApplyPiecewiseLinearTransform(pLayer, i, i, 0, m, nSaturated, pwlBuff);
    }
}

void GNAApplyTranspose(
    const nn_layer* pLayer)
{
    uint32_t m = pLayer->nInputRows;
    uint32_t n = pLayer->nInputColumns;
    int16_t *in = (int16_t*)pLayer->pInputs;
    int16_t *out = (int16_t*)pLayer->pOutputs;

    transpose16(m, n, in, out);
}

void GNAApplyCopy(
    const nn_layer* pLayer)
{
    uint32_t    r;                  // row iterator
    int16_t*    in = (int16_t*)pLayer->pInputs;
    int16_t*    out = (int16_t*)pLayer->pOutputs;
    uint32_t    nRows = ((nn_layer_copy*)pLayer->pLayerStruct)->nCopyRows;
    uint32_t    nBytesCp = ((nn_layer_copy*)pLayer->pLayerStruct)->nCopyCols
        * sizeof(int16_t);

    for (r = 0; r < nRows; r++)
    {
        memcpy_s(
            out + (pLayer->nOutputColumns * r),
            nBytesCp,
            in + (pLayer->nInputColumns * r),
            nBytesCp);
    }
}

void GNAApplyConvolutionalTransform(
    const nn_layer*     pLayer,
          uint32_t*     nSaturated,
          void*         pwlBuff,
          int64_t*      pool)
{
    auto conv = (nn_layer_conv*)pLayer->pLayerStruct;
    auto pwl = &conv->pwl;

    if (conv->poolType == INTEL_NO_POOLING)
    {
        CNNFilter16(
            pLayer->nInputColumns,
            conv->nFeatureMaps,
            conv->nFeatureMapColumns,
            conv->nFilters,
            conv->nFilterCoefficients,
            (int16_t*)pLayer->pInputs,
            (int16_t*)conv->pFilters,
            (nn_bias_s*)conv->pBiases,
            getOutputBuffer(pwl, pLayer),
            nSaturated);
        if (IsActivationFunctionEnabled(pwl))
        {
            GNAApplyPiecewiseLinearTransform(
                pLayer,
                0,
                pLayer->nOutputRows - 1,
                0,
                pLayer->nOutputColumns - 1,
                nSaturated,
                pwlBuff);
        }
    }
    else
    {
        CNNFilterPool16(pLayer->nInputColumns,
            conv->nFeatureMaps,
            conv->nFeatureMapColumns,
            conv->nFilters,
            conv->nFilterCoefficients,
            conv->nPoolSize,
            conv->nPoolStride,
            conv->pwl.nSegments,
            conv->pwl.pSegments,
            (nn_bias_s*)conv->pBiases,
            (int16_t*)conv->pFilters,
            (int16_t*)pLayer->pInputs,
            (int16_t*)pLayer->pOutputs,
            nSaturated,
            conv->poolType,
            pwlBuff,
            pool);
    }
}

XnnKernel KERNEL(xnnKernel) =
{
        GNAApplyAffineTransform,
        GNAApplyAffineMBiasTransform,
        GNAApplyDiagonalTransform,
        GNAApplyPiecewiseLinearTransform,
        GNAApplyRecurrentTransform,
        GNAApplyTranspose,
        GNAApplyCopy,
        GNAApplyConvolutionalTransform
};

}