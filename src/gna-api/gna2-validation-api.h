/*
    Copyright 2018 Intel Corporation.
    This software and the related documents are Intel copyrighted materials,
    and your use of them is governed by the express license under which they
    were provided to you (Intel OBL Software License Agreement (OEM/IHV/ISV
    Distribution & Single User) (v. 11.2.2017) ). Unless the License provides
    otherwise, you may not use, modify, copy, publish, distribute, disclose or
    transmit this software or the related documents without Intel's prior
    written permission.

    This software and the related documents are provided as is, with no
    express or implied warranties, other than those that are expressly
    stated in the License.
*/

/**************************************************************************//**
 @file gna2-validation-api.h
 @brief Gaussian and Neural Accelerator (GNA) 2.0 Validation API.
 @nosubgrouping

 ******************************************************************************

 @addtogroup GNA2_VALIDATION_API Gaussian and Neural Accelerator (GNA) 2.0 Validation API

 API for validating GNA library and devices.

 @{
 *****************************************************************************/

#ifndef __GNA2_VALIDATION_API_H
#define __GNA2_VALIDATION_API_H

#include "gna2-api.h"

#include <stdint.h>


/******************  GNA Debug API ******************/
/* This API is for internal GNA hardware testing only*/

typedef enum _dbg_action_type
{
    GnaDumpMmio,
    GnaReservedAction,
    GnaZeroMemory,
    GnaDumpMemory,
    GnaDumpXnnDescriptor,
    GnaDumpGmmDescriptor,
    GnaSetXnnDescriptor,
    GnaSetGmmDescriptor,
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

#if defined(_WIN32)
#pragma warning(disable : 201)
#endif
typedef struct _dbg_action
{
    dbg_action_type action_type;
    gna_reg gna_register;
    union
    {
        uint32_t timeout;
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
GNA_API enum GnaStatus GnaModelSetPrescoreScenario(
    uint32_t modelId,
    uint32_t nActions,
    dbg_action *pActions);

GNA_API enum GnaStatus GnaModelSetAfterscoreScenario(
    uint32_t modelId,
    uint32_t nActions,
    dbg_action *pActions);

#endif // __GNA2_VALIDATION_API_H

/**
 @}
 */
