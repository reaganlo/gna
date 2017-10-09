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
 * GNA 2.0 API
 *
 * Gaussian Mixture Models and Neural Network Accelerator Module
 * Extra API functions definitions
 *
 *****************************************************************************/

#pragma once

#include "gna-api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum _gna_device_kind
{
    GNA_SUE,        // GNA v1.0
    GNA_SUE_2,      // GNA v2.0
    GNA_CNL,        // GNA v1.0
    GNA_GLK,        // GNA v1.0
    GNA_ICL,        // GNA v1.0
    GNA_TGL,        // GNA v2.0
} gna_device_kind;

/**
 * Dumps the hardware-consumable model to the file
 * Model should be created through standard API GnaModelCreate function
 * Model will be validated against device kind provided as function argument
 * File path can be as well relative or absolute path to output file
 *
 * @param modelId       Model to be dumped to file
 * @param deviceKind    Device on which model will be used
 * @param filepath      Absolute or relative path to output file
 */
GNAAPI intel_gna_status_t GnaModelDump(
    gna_model_id        modelId,
    gna_device_kind     deviceKind,
    const char*         filepath);

#ifdef __cplusplus
}
#endif
