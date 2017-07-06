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

#pragma once

#include "GnaDrvApi.h"
#include "GnaTypes.h"

namespace GNA
{

/**
 * Common aliases for IOCTL data types for Windows and Linux
 */
#ifdef USING_GCC
typedef gmm_ioctl_score_gmms_t hw_calc_out_t;
#else
typedef GNA_MM_IN       hw_mmap_in_t;
typedef GNA_PGDIR_OUT   hw_pgdir_out_t;
typedef GNA_CALC_IN     hw_calc_in_t;
typedef GNA_READREG_IN  hw_read_in_t;
typedef GNA_READREG_OUT hw_read_out_t;
typedef GNA_WRITEREG_IN hw_write_in_t;
#endif // USING_GCC

# pragma pack (1) // set structure packaging to 1 to ensure alignment and size

typedef uint8_t     __1B_RES;       // 1B of reserved memory

/******************************************************************************
*
* GNA HW data configuration definitions
*
******************************************************************************/

/**
 * GMM Configuration
 *
 * Offset:  0x0100 (interchangeably with xNN Configuration)
 * Size:    0x80
 * See:     HAS Section 5.4.2.7
 * Note:    Specifies Acoustic Model and scoring parameters
 */
typedef struct _GMM_CONFIG
{
    FVADDR      fvaddr;             // 0000 - 0003
    FVOFFSET    fvoffset;           // 0004 - 0007
    FVWIDTH     fvwidth;            // 0008 - 000B
    GMM_MODE_CTRL mode;             // 000C - 000F GMM mode control
    NUMFV       numfv;              // 0010 - 0013
    VLENGTH     vlength;            // 0014 - 0017
    MVADDR      mvaddr;             // 0018 - 001B
    __1B_RES    __res_001c[4];      // 001C - 001F (4B reserved)
    MVWIDTH     mvwidth;            // 0020 - 0023
    __1B_RES    __res_0024[4];      // 0024 - 0027 (4B reserved)
    MVSOFFSET   mvsoffset;          // 0028 - 002B
    __1B_RES    __res_002c[4];      // 002C - 002F (4B reserved)
    VVADDR      vvaddr;             // 0030 - 0033
    __1B_RES    __res_0034[4];      // 0034 - 0037 (4B reserved)
    VVWIDTH     vvwidth;            // 0038 - 003B
    __1B_RES    __res_003c[4];      // 003C - 003F (4B reserved)
    VVSOFFSET   vvsoffset;          // 0040 - 0043
    GCADDR      gcaddr;             // 0044 - 0047
    __1B_RES    __res_0048[4];      // 0048 - 004B (4B reserved)
    GCWIDTH     gcwidth;            // 004C - 004F
    GCSOFFSET   gcsoffset;          // 0050 - 0053
    MAXLSSCORE  maxlsscore;         // 0054 - 0057
    MAXLSWIDTH  maxlswidth;         // 0058 - 005B
    NUMMCPG     nummcpg;            // 005C - 005F
    GMMTELST    gmmtelst;           // 0060 - 0063
    NUMGMMS     numgmms;            // 0064 - 0067
    ASLADDR     asladdr;            // 0068 - 006B
    __1B_RES    __res_006c[4];      // 006C - 006F (4B reserved)
    ASTLISTLEN  astlistlen;         // 0070 - 0073
    GMMSCRWIDTH gmmscrwdth;         // 0074 - 0077
    GMMSCRADD   gmmscradd;          // 0078 - 007B
    GMMSCRLEN   gmmscrlen;          // 007C - 007F

} GMM_CONFIG;                       // GMM Configuration

static_assert(128 == sizeof(GMM_CONFIG), "Invalid size of GMM_CONFIG");

/**
 * xNN Configuration
 *
 * Offset:  0x0100 (interchangeably with GMM Configuration)
 * Size:    0x06
 * See:     HAS Section 5.4.3.8
 * Note:    Specifies Neural Network and scoring parameters
 */
typedef struct _XNN_CONFIG
{
    LABASE      labase;             // 0100 - 0103 - Layer array base
    LACNT       lacnt;              // 0104 - 0105 - Layer array count

} XNN_CONFIG;                       // xNN Configuration

static_assert(6 == sizeof(XNN_CONFIG), "Invalid size of XNN_CONFIG");

/**
* xNN Data Structures - xNN Operation Type
*
* See:     HAS Section 5.4.3.1
* Note:    Enumerates the supported operations by the xNN operation
*/
typedef enum _NN_OP_TYPE : uint8_t
{
    NN_AFFINE = 0x00,
    NN_AFF_AL = 0x01,
    NN_DIAG = 0x02,
    NN_RNN = 0x04,
    NN_CNN = 0x08,
    NN_AFF_MB = 0x09,
    NN_PMA = 0x0A,
    NN_DEINT = 0x10,
    NN_INTER = 0x11,
    NN_COPY = 0x12,
    NN_GMM = 0x20,
    NN_GMM_ACTIVE_LIST = 0x21,
    NN_RESERVED = 0xff

} NN_OP_TYPE;

static_assert(1 == sizeof(NN_OP_TYPE), "Invalid size of NNOOPERATIONTYPE");

/**
 * xNN - NN Flags 
 *
 * Offset:  0x01
 * Size:    0x01 B
 * See:     HAS Section 5.4.3.1
 * Note:    List of flags that impact flavors of the xNN operation
 */
typedef union _NN_FLAGS
{
    struct
    {
    uint8_t     weight_size : 2;    // 00:01 Weight element size:
                                    //      0b0 - 16-bit element, Dens Const format
                                    //      0b1 - 8-bit element, Rich Const format
    uint8_t     act_fn_en   : 1;    // 02:02 Activation function is disabled (0b0) or enabled (0b1)
    uint8_t     pool_param  : 2;    // 03:04 No Pool (0b00), MaxPool (0b01), AvaragePool (0b10), Reserved (0b11). Applicable in CNN layers only.
    uint8_t     __res_05    : 3;    // 05:07 Reserved
    };
    uint8_t     _char;              // value of whole register

} NN_FLAGS;                         // Flavor of the xNN operation

static_assert(1 == sizeof(NN_FLAGS), "Invalid size of NN_FLAGS");

/**
 * xNN - universal Layer Descriptor for all operation types
 *
 * Size:    0x80 B
 * See:     HAS Section 5.4.3.1
 */
typedef union _XNN_LYR
{
    struct
    {
    NN_OP_TYPE  op;                 // 0x00 : 0x00 Type of xNN operation to be scored (NNOOPERATIONTYPE enum)
    NN_FLAGS    flags;              // 0x01 : 0x01 Flavors of the xNN operation
    uint16_t    n_in_elems;         // 0x02 : 0x03 Total number of input elements
    union{                          // 
    uint16_t    n_out_elems;        // 0x04 : 0x05 Number of output elements [1 - (2^16-1)]
    uint16_t    cnn_n_out_p_flt;    // 0x04 : 0x05 CNN Number of output elements per Filter in full iterations
    };                              //
    union{                          // 
    uint8_t     n_groups;           // 0x06 : 0x06 Number of input groups used
    uint8_t     cnn_n_flt_last;     // 0x06 : 0x06 CNN Number of filters in buffer in last iteration [4,8,12,16)]
    };                              //
    union{                          // 
    uint8_t     n_iters;            // 0x07 : 0x07 Blocking size used to fit size of input buffer
    uint8_t     cnn_pool_stride;    // 0x07 : 0x07 CNN Pool Stride [1-6]
    };                              //
    union{                          // 
    uint16_t    n_elems_last;       // 0x08 : 0x09 Number of input elements in last iteration per group
    uint16_t    cnn_n_flt_stride;   // 0x08 : 0x09 CNN Input-filter stride - Number of input elements for convolution operation [1-768]
    };                              //
    union{                          // 
    uint8_t     rnn_n_fb_iters;     // 0x0a : 0x0a Number of iterations in feedback stage
    uint8_t     cnn_pool_size;      // 0x0a : 0x0a CNN Size of Pool [1-6]
    };                              //
    __1B_RES    __res_0b;           // 0x0b : 0x0b Reserved
    union{                          // 
    uint16_t    rnn_n_elems_first;  // 0x0c : 0x0d Number of elements in first feedback iteration 
    uint16_t    cnn_n_flts;         // 0x0c : 0x0d CNN Number of convolution filters [4 - (2^16 - 4)], %4
    };                              //
    union{                          //
    uint16_t    rnn_n_elems_last;   // 0x0e : 0x0f Number of elements in last feedback iteration
    uint16_t    cnn_n_flt_iters;    // 0x0e : 0x0f CNN Number of iterations for all convolution filters
    };                              //
    uint8_t     pwl_n_segs;         // 0x10 : 0x10 Number of activation function segments
    __1B_RES    __res_11;           // 0x11 : 0x11 Reserved
    union{                          // 
    uint16_t    act_list_n_elems;   // 0x12 : 0x13 Number of output elements in output active list enabled mode
    uint16_t    cpy_n_elems;        // 0x12 : 0x13 Number of elements copied in copy OP operation [8 - (2^16 - 8)], %8
    uint16_t    cnn_flt_size;       // 0x12 : 0x13 CNN convolution filter size (elements per filter) [48 - 768], %8
    uint16_t    bias_grp_cnt;       // 0x12 : 0x13 Grouping of the bias array [1-8]
    };                              //
    union{                          //
    uint16_t    cnn_n_flts_iter;    // 0x14 : 0x15 CNN Number of filters in input buffer in full iterations [4,8,12,16]
    uint16_t    bias_grp_value;     // 0x14 : 0x15 Current column selected [0-7]
    };                              //
    uint16_t    cnn_n_flt_outs;     // 0x16 : 0x17 CNN Number of output elements per Filter after conv., before pooling 
    uint16_t    cnn_flt_bf_sz_iter; // 0x18 : 0x19 CNN filter buffer size per (non-last) iteration (B) [1-InBufSize/2]
    uint16_t    cnn_flt_bf_sz_last; // 0x1A : 0x1B CNN filter buffer size in last iteration (B) [1-InBufSize/2]
    __1B_RES    __res_1c[4];        // 0x1C : 0x1F Reserved
    union{                          //
    uint32_t    in_buffer;          // 0x20 : 0x23 Pointer to input array [2B elements]
    uint32_t    gmm_descriptor;     // 0x20 : 0x23 Pointer GMM layer descriptor
    };                              //
    uint32_t    out_act_fn_buffer;  // 0x24 : 0x27 Pointer to 2B output array after pwl act. fn. [2B elements]
    uint32_t    out_sum_buffer;     // 0x28 : 0x2B Pointer to 4B intermediate output sum array. [4B elements]
    uint32_t    rnn_out_fb_buffer;  // 0x2C : 0x2f Pointer to output FB array
    union{                          // 
    uint32_t    aff_weight_buffer;  // 0x30 : 0x33 Pointer to weights array [1B or 2B elements]
    uint32_t    cnn_flt_buffer;     // 0x30 : 0x33 CNN Pointer to Filter array [2B elements]
    };                              //
    uint32_t    aff_const_buffer;   // 0x34 : 0x37 Pointer to const and weight scale array. [4B elements or 1B scale +3B res.]
    union{                          // 
    uint32_t    act_list_buffer;    // 0x38 : 0x3b Active outputs list pointer [4B elements]
    uint32_t    bias_grp_ptr;       // 0x38 : 0x3b Bias grouping array pointer [4B elements]
    };                              //
    uint32_t    pwl_seg_def_buffer; // 0x3c : 0x3f Pointer to array that holds the activation function section definition [8B elements]
    __1B_RES    __res_40[64];       // 0x40 : 0x7f Reserved
    };
    uint8_t     _char[128];         // value of whole register

} XNN_LYR;                          // DNN Layer Descriptor

static_assert(128 == sizeof(XNN_LYR), "Invalid size of XNN_LYR");
static_assert(XNN_LYR_DSC_SIZE == sizeof(XNN_LYR), "Size of XNN_LYR inconsistent with driver requirement");

# pragma pack ()

}
