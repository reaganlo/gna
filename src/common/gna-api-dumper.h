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
* Header describing parameters of dumped model.
* Structured is partially filled by GNADumpXnn with parameters necessary for SueScrek,
* other fields are populated by user as necessary, other fields are populated by user.
*/
typedef struct _intel_gna_model_header
{
    uint32_t layer_descriptor_base; // Offset in bytes of first layer descriptor in network.
    uint32_t model_size;            // Total size of model in bytes determined by GNADumpXnn including hw descriptors, model data and input/output buffers.
    uint32_t gna_mode;              // Mode of GNA operation, 1 = XNN mode (default), 0 = GMM mode.
    uint32_t layer_count;           // Number of layers in model.

    uint32_t bytes_per_input;       // Network Input resolution in bytes.
    uint32_t bytes_per_output;      // Network Output resolution in bytes.
    uint32_t input_nodes_count;     // Number of network input nodes.
    uint32_t output_nodes_count;    // Number of network output nodes.

    uint32_t input_descriptor_offset;// Offset in bytes of input pointer descriptor field that need to be set for processing.
    uint32_t output_descriptor_offset;// Offset in bytes of output pointer descriptor field that need to be set for processing.

    uint32_t rw_region_size;        // Size in bytes of read-write region of statically linked GNA model.
    float    input_scaling_factor;   // Scaling factor used for quantization of input values.
    float    output_scaling_factor;  // Scaling factor used for quantization of output values.

    uint8_t  reserved[12];          // Padding to 64B.
} intel_gna_model_header;

static_assert(64 == sizeof(intel_gna_model_header), "Invalid size of intel_gna_model_header");

typedef void* (*intel_gna_alloc_cb)(size_t size);

/**
* Dumps the hardware-consumable model to the file
* Model should be created through standard API GnaModelCreate function
* Model will be validated against device kind provided as function argument
* File path can be as well relative or absolute path to output file
*
* @param modelId       Model to be dumped to file
* @param deviceKind    Device on which model will be used
* @param modelHeader   (out) Header describing parameters of model being dumped.
* @param status        (out) Status of conversion and dumping.
* @param customAlloc   Pointer to a function with custom memory allocation. Total model size needs to be passed as parameter.
*/
GNAAPI void* GnaModelDump(
    gna_model_id modelId,
    gna_device_kind deviceKind,
    intel_gna_model_header* modelHeader,
    intel_gna_status_t* status,
    intel_gna_alloc_cb customAlloc);

#ifdef __cplusplus
}
#endif