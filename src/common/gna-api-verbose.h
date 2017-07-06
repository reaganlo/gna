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
 * Gaussian Mixture Models and Neural Network Accelerator Module
 * API Definition
 *
 *****************************************************************************/

#ifndef __GNA_DEBUG_API_H
#define __GNA_DEBUG_API_H

#include <stdint.h>

#include "gna-api.h"
#include "gna-api-status.h"
#include "gna-api-types-gmm.h"
#include "gna-api-types-xnn.h"

#ifdef __cplusplus
extern "C" {
#endif

/******************  GNA Debug API ******************/
/* This API is for internal GNA hardware testing only*/

typedef enum _dbg_action_type
{
    GnaDumpMmio,
    GnaDumpPageDirectory,
    GnaZeroMemory,
    GnaDumpMemory,
    GnaDumpXnnDescriptor,
    GnaSetXnnDescriptor,
    GnaReadRegister,
    GnaWriteRegister,
    GnaLogMessage,
    GnaSleep,

    NUM_ACTION_TYPES
} dbg_action_type;

typedef enum _register_operation
{
    Equal,
    And,
    Or
} register_op;

typedef enum _gna_register
{
    GNA_STS          = 0x80,
    GNA_CTRL         = 0x84,
    GNA_MCTL         = 0x88,
    GNA_PTC          = 0x8C,
    GNA_SC           = 0x90,
    GNA_ISI          = 0x94,
    GNA_ISV_LOW      = 0x98,
    GNA_ISV_HIGH     = 0x9C,
    GNA_BP_LOW       = 0xA0,
    GNA_BP_HIGH      = 0xA4,
    GNA_D0i3C        = 0xA8,
    GNA_DESBASE      = 0xB0,
    GNA_IBUFFS       = 0xB4,
    GNA_SAI1_LOW     = 0x100,
    GNA_SAI1_HIGH    = 0x104,
    GNA_SAI2_LOW     = 0x108,
    GNA_SAI2_HIGH    = 0x10C,
    GNA_SAIV         = 0x110

} gna_reg;

typedef enum _gna_set_size
{
    GNA_SET_BYTE =   0,
    GNA_SET_WORD =   1,
    GNA_SET_DWORD =  2,
    GNA_SET_QWORD =  3,
    GNA_SET_XNNLYR = 4,
} gna_set_size;

typedef struct _dbg_action
{
    dbg_action_type action_type;
    gna_reg gna_register;
    union 
    {
        gna_timeout timeout;
        const char *log_message;
        const char *filename;
        uint64_t xnn_value;
        uint32_t reg_value;
        void *outputs;
    };
    union
    {
        register_op reg_operation;
        struct
        {
            uint32_t xnn_offset : 29;
            gna_set_size xnn_value_size : 3;
        };
        uint32_t outputs_size;
    };
    uint32_t layer_number;
} dbg_action;

/**
 * Adds a custom debug scenario to the model
 * Actions will be performed sequentially in order
 *
 * @param modelId
 * @param nActions
 * @param pActions
 */
GNAAPI intel_gna_status_t GnaModelSetPrescoreScenario(
    gna_model_id modelId,
    uint32_t nActions,
    dbg_action *pActions);

GNAAPI intel_gna_status_t GnaModelSetAfterscoreScenario(
    gna_model_id modelId,
    uint32_t nActions,
    dbg_action *pActions);

#ifdef __cplusplus
}
#endif

#endif // __GNA_DEBUG_API_H
