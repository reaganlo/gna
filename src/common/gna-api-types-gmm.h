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

/******************************************************************************
 *
 * GMM Scoring and Neural Network Accelerator Module
 * API Gaussian Mixture Model types definition
 *
 *****************************************************************************/

#ifndef __GNA_API_TYPES_GMM_H
#define __GNA_API_TYPES_GMM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** GMM Calculation modes */
typedef enum _gmm_mode
{
    GNA_MAXMIX8,                    // MaxMix mode with 1B Inverse Covariances, use with inverseCovariancesForMaxMix8.
    GNA_MAXMIX16,                   // MaxMix mode with 2B Inverse Covariances, use with inverseCovariancesForMaxMix16.
    GNA_LINF,                       // L-infinite distance computation.
    GNA_L1,                         // L-1 distance computation.
    GNA_L2,                         // L-2 distance computation.
    GNA_GMM_MODES_COUNT             // Number of modes.

} gna_gmm_mode;

/** GMM Data layouts */
typedef enum _gmm_layout
{
    GMM_LAYOUT_FLAT,                // Each data component is grouped by type. gna_gmm_data buffers can be separate.
    GMM_LAYOUT_INTERLEAVED,         // Each data component is grouped by state. gna_gmm_data buffers use single memory buffer.

} gna_gmm_layout;

/** GMM detailed configuration */
typedef struct _gmm_config
{
    gna_gmm_mode    mode;           // Calculation mode.
    gna_gmm_layout  layout;         // Data layout.
    uint32_t        mixtureComponentCount;// Number of mixture components.
    uint32_t        stateCount;     // Number of states.
    uint32_t        maximumScore;   // Maximum Score value above which scores are saturated. TODO:KJ: Open to clarify if can be set per GMM layer or per request

} gna_gmm_config;

/** GMM Data buffers */
typedef struct _gmm_data
{
    uint8_t* meanValues;            // Mean values buffer.
    union {
    uint8_t* inverseCovariancesForMaxMix8;  // Inverse Covariances buffer, use with GNA_MAXMIX8 gna_gmm_mode.
    uint16_t* inverseCovariancesForMaxMix16;// Inverse Covariances buffer, use with GNA_MAXMIX16 gna_gmm_mode.
    };
    uint32_t* gaussianConstants;    // Gaussian constants buffer.

} gna_gmm_data;

/** GMM Layer detailed configuration */
typedef struct _gmm_layer
{
    gna_gmm_config config;          // GMM configuration.
    gna_gmm_data data;              // GMM data buffers.

} gna_gmm_layer;

/** Maximum number of mixture components per GMM State */
const uint32_t GMM_MIXTURE_COMP_COUNT_MAX = 4096;

/** Maximum number of GMM states, active list elements and  */
const uint32_t GMM_STATES_COUNT_MAX = 262144;

/** Size of memory alignment for mean, variance vectors and Gaussian Constants */
const uint32_t GMM_MEM_ALIGNMENT = 8;

/** Mean vector width in bytes */
const uint32_t GMM_MEAN_VALUE_SIZE = 1;

/** Minimum variance vector width in bytes */
const uint32_t GMM_COVARIANCE_SIZE_MIN = 1;

/** Maximum variance vector width in bytes */
const uint32_t GMM_COVARIANCE_SIZE_MAX = 2;

/** Gaussian Constants width in bytes */
const uint32_t GMM_CONSTANTS_SIZE = 4;

/** Score width in bytes */
const uint32_t GMM_SCORE_SIZE = 4;

/** Size of memory alignment for feature vectors */
const uint32_t GMM_FV_MEM_ALIGN = 64;

/** Feature vector width in bytes */
const uint32_t GMM_FV_ELEMENT_SIZE = 1;

/** Maximum number of feature vectors */
const uint32_t GMM_FV_COUNT_MAX = 8;

/** Minimum length of a vector */
const uint32_t GMM_FV_ELEMENT_COUNT_MIN = 24;

/** Maximum length of a vector */
const uint32_t GMM_FV_ELEMENT_COUNT_MAX = 96;

/** The allowed alignment of vector lengths */
const uint32_t GMM_FV_ELEMENT_COUNT_MULTIPLE_OF = 8;

#ifdef __cplusplus
}
#endif

#endif  // ifndef __GNA_API_TYPES_GMM_H
