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

#include <cstring>
#include <cstdlib>
#include <iostream>

#include "gna-api.h"
#include "gna-api-verbose.h"

#include "SetupDnnModel_1.h"

namespace
{
const int layersNum = 1;
const int groupingNum = 4;
const int inVecSz = 16;
const int outVecSz = 8;

const int8_t weights_1B[outVecSz * inVecSz] =
{
    -6, -2, -1, -1, -2,  9,  6,  5,  2,  4, -1,  5, -2, -4,  0,  9,
    -8,  8, -4,  6,  5,  3, -7, -9,  7,  0, -4, -1,  1,  7,  6, -6,
    2, -8,  6,  5, -1, -2,  7,  5, -1,  4,  8,  7, -9, -1,  7,  1,
    0, -2,  1,  0,  6, -6,  7,  4, -6,  0,  3, -2,  1,  8, -6, -2,
    -6, -3,  4, -2, -8, -6,  6,  5,  6, -9, -5, -2, -5, -8, -6, -2,
    -7,  0,  6, -3, -1, -6,  4,  1, -4, -5, -3,  7,  9, -9,  9,  9,
    0, -2,  6, -3,  5, -2, -1, -3, -5,  7,  6,  6, -8,  0, -4,  9,
    2,  7, -8, -7,  8, -6, -6,  1,  7, -4, -4,  9, -6, -6,  5, -7
};

const int16_t weights_2B[outVecSz * inVecSz] =
{
    -6, -2, -1, -1, -2,  9,  6,  5,  2,  4, -1,  5, -2, -4,  0,  9,
    -8,  8, -4,  6,  5,  3, -7, -9,  7,  0, -4, -1,  1,  7,  6, -6,
    2, -8,  6,  5, -1, -2,  7,  5, -1,  4,  8,  7, -9, -1,  7,  1,
    0, -2,  1,  0,  6, -6,  7,  4, -6,  0,  3, -2,  1,  8, -6, -2,
    -6, -3,  4, -2, -8, -6,  6,  5,  6, -9, -5, -2, -5, -8, -6, -2,
    -7,  0,  6, -3, -1, -6,  4,  1, -4, -5, -3,  7,  9, -9,  9,  9,
    0, -2,  6, -3,  5, -2, -1, -3, -5,  7,  6,  6, -8,  0, -4,  9,
    2,  7, -8, -7,  8, -6, -6,  1,  7, -4, -4,  9, -6, -6,  5, -7
};

const int16_t inputs[inVecSz * groupingNum] = {
    -5,  9, -7,  4,
    5, -4, -7,  4,
    0,  7,  1, -7,
    1,  6,  7,  9,
    2, -4,  9,  8,
    -5, -1,  2,  9,
    -8, -8,  8,  1,
    -7,  2, -1, -1,
    -9, -5, -8,  5,
    0, -1,  3,  9,
    0,  8,  1, -2,
    -9,  8,  0, -7,
    -9, -8, -1, -4,
    -3, -7, -2,  3,
    -8,  0,  1,  3,
    -4, -6, -8, -2
};

const intel_bias_t regularBiases[outVecSz*groupingNum] = {
    5, 4, -2, 5,
    -7, -5, 4, -1
};

const  intel_compound_bias_t compoundBiases[outVecSz*groupingNum] =
{
    { 5,1,{0} }, {4,1,{0}}, {-2,1,{0}}, {5,1,{0}},
    {-7,1,{0}}, {-5,1,{0}}, {4,1,{0}}, {-1,1,{0}},
};

const int32_t ref_output[outVecSz * groupingNum] =
{
    -177, -85, 29, 28,
    96, -173, 25, 252,
    -160, 274, 157, -29,
    48, -60, 158, -29,
    26, -2, -44, -251,
    -173, -70, -1, -323,
    99, 144, 38, -63,
    20, 56, -103, 10
};

const uint32_t alIndices[outVecSz / 2]
{
    0, 2, 4, 7
};
}

typedef uint8_t     __1B_RES;       // 1B of reserved memory

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



SetupDnnModel_1::SetupDnnModel_1(DeviceController & deviceCtrl, bool wght2B, bool activeListEn, bool pwlEn)
    : deviceController{deviceCtrl},
    weightsAre2Bytes{wght2B},
    activeListEnabled{activeListEn},
    pwlEnabled{pwlEn}
{
    nnet.nGroup = groupingNum;
    nnet.nLayers = layersNum;
    nnet.pLayers = (intel_nnet_layer_t*)calloc(nnet.nLayers, sizeof(intel_nnet_layer_t));

    sampleAffineLayer();

    deviceController.ModelCreate(&nnet, &modelId);

    configId = deviceController.ConfigAdd(modelId);

    deviceController.BufferAdd(configId, GNA_IN, 0, inputBuffer);
    deviceController.BufferAdd(configId, GNA_OUT, 0, outputBuffer);

    if (activeListEnabled)
    {
        deviceController.ActiveListAdd(configId, 0, indicesCount, indices);
    }

#if HW_VERBOSE == 1
    constexpr uint32_t nActions = 17;

    dbg_action afterActions[nActions]; 
    afterActions[0].action_type = GnaSleep;
    afterActions[0].timeout = 100;

    afterActions[1].action_type = GnaLogMessage;
    afterActions[1].log_message = "Log message test\n";

    afterActions[2].action_type = GnaZeroMemory;
    afterActions[2].outputs = outputBuffer;
    afterActions[2].outputs_size = outVecSz * groupingNum * sizeof(int32_t);

    afterActions[3].action_type = GnaReadRegister;
    afterActions[3].gna_register = GNA_STS;

    afterActions[4].action_type = GnaDumpMmio;

    afterActions[5].action_type = GnaDumpPageDirectory;

    afterActions[6].action_type = GnaDumpMemory;
    afterActions[6].filename = "dump.bin";

    afterActions[7].action_type = GnaDumpXnnDescriptor;
    afterActions[7].layer_number = 0;

    afterActions[8].action_type = GnaSetXnnDescriptor;
    afterActions[8].layer_number = 0;
    afterActions[8].xnn_offset = offsetof(XNN_LYR, op);
    afterActions[8].xnn_value = 0x13; // invalid op
    afterActions[8].xnn_value_size = GNA_SET_BYTE;

    afterActions[9] = afterActions[7];

    // start accelerator
    afterActions[10].action_type = GnaWriteRegister;
    afterActions[10].gna_register = GNA_CTRL;
    afterActions[10].reg_operation = Or;
    afterActions[10].reg_value = 1;

    afterActions[11].action_type = GnaReadRegister;
    afterActions[11].gna_register = GNA_STS;

    afterActions[12].action_type = GnaSleep;
    afterActions[12].timeout = 200;

    afterActions[13].action_type = GnaReadRegister;
    afterActions[13].gna_register = GNA_STS;

    // abort
    afterActions[14].action_type = GnaWriteRegister;
    afterActions[14].gna_register = GNA_CTRL;
    afterActions[14].reg_operation = Or;
    afterActions[14].reg_value = 4;

    afterActions[15] = afterActions[12];

    afterActions[16].action_type = GnaReadRegister;
    afterActions[16].gna_register = GNA_STS;

    deviceController.AfterscoreDebug(modelId, nActions, afterActions);
#endif

    //deviceController.DumpModel(modelId, "dump.bin");
}

SetupDnnModel_1::~SetupDnnModel_1()
{
    deviceController.Free();

    free(nnet.pLayers);
}

void SetupDnnModel_1::checkReferenceOutput(int modelIndex, int configIndex) const
{
    for (int i = 0; i < sizeof(ref_output) / sizeof(int32_t); ++i)
    {
        int32_t outElemVal = static_cast<const int32_t*>(outputBuffer)[i];
        if (ref_output[i] != outElemVal)
        {
            // TODO: how it should notified? return or throw
            throw std::runtime_error("Wrong output");
        }
    }
}

void SetupDnnModel_1::sampleAffineLayer()
{
    int buf_size_weights = weightsAre2Bytes ? ALIGN64(sizeof(weights_2B)) : ALIGN64(sizeof(weights_1B));
    int buf_size_inputs = ALIGN64(sizeof(inputs));
    int buf_size_biases = weightsAre2Bytes ? ALIGN64(sizeof(regularBiases)) : ALIGN64(sizeof(compoundBiases));
    int buf_size_outputs = ALIGN64(outVecSz * groupingNum * sizeof(int32_t));
    int buf_size_tmp_outputs = ALIGN64(outVecSz * groupingNum * sizeof(int32_t));
    int buf_size_pwl = ALIGN64(nSegments * sizeof(intel_pwl_segment_t));

    uint32_t bytes_requested = buf_size_weights + buf_size_inputs + buf_size_biases + buf_size_outputs + buf_size_tmp_outputs;
    if (activeListEnabled)
    {
        indicesCount = outVecSz / 2; 
        bytes_requested += indicesCount * sizeof(uint32_t);
    }
    if (pwlEnabled)
    {
        bytes_requested += buf_size_pwl;
    }
    uint32_t bytes_granted;

    uint8_t* pinned_mem_ptr = deviceController.Alloc(bytes_requested, nnet.nLayers, 0, &bytes_granted);

    void* pinned_weights = pinned_mem_ptr;
    if (weightsAre2Bytes)
    {
        memcpy(pinned_weights, weights_2B, sizeof(weights_2B));
    }
    else
    {
        memcpy(pinned_weights, weights_1B, sizeof(weights_1B));
    }
    pinned_mem_ptr += buf_size_weights;

    inputBuffer = pinned_mem_ptr;
    memcpy(inputBuffer, inputs, sizeof(inputs));
    pinned_mem_ptr += buf_size_inputs;

    int32_t *pinned_biases = (int32_t*)pinned_mem_ptr;
    if (weightsAre2Bytes)
    {
        memcpy(pinned_biases, regularBiases, sizeof(regularBiases));
    }
    else
    {
        memcpy(pinned_biases, compoundBiases, sizeof(compoundBiases));
    }
    pinned_mem_ptr += buf_size_biases;

    outputBuffer = pinned_mem_ptr;
    pinned_mem_ptr += buf_size_outputs;

    if (activeListEnabled)
    {
        size_t indicesSize = indicesCount * sizeof(uint32_t);
        indices = (uint32_t*)pinned_mem_ptr;
        memcpy(indices, alIndices, indicesSize);
        pinned_mem_ptr += indicesSize;
    }

    void *tmp_outputs = nullptr;
    if (pwlEnabled)
    {
        tmp_outputs = pinned_mem_ptr;
        pinned_mem_ptr += buf_size_tmp_outputs;

        intel_pwl_segment_t *pinned_pwl = reinterpret_cast<intel_pwl_segment_t*>(pinned_mem_ptr);
        pinned_mem_ptr += buf_size_pwl;

        pwl.nSegments = nSegments;
        pwl.pSegments = pinned_pwl;
        samplePwl(pwl.pSegments, pwl.nSegments);
    }
    else
    {
        pwl.nSegments = 0;
        pwl.pSegments = nullptr;
    }

    affine_func.nBytesPerWeight = weightsAre2Bytes ? 2 : 1;
    affine_func.nBytesPerBias = weightsAre2Bytes ? sizeof(intel_bias_t) : sizeof(intel_compound_bias_t);
    affine_func.pWeights = pinned_weights;
    affine_func.pBiases = pinned_biases;

    affine_layer.affine = affine_func;
    affine_layer.pwl = pwl;

    nnet.pLayers[0].nInputColumns = nnet.nGroup;
    nnet.pLayers[0].nInputRows = inVecSz;
    nnet.pLayers[0].nOutputColumns = nnet.nGroup;
    nnet.pLayers[0].nOutputRows = outVecSz;
    nnet.pLayers[0].nBytesPerInput = sizeof(int16_t);
    nnet.pLayers[0].nBytesPerIntermediateOutput = 4;
    nnet.pLayers[0].nLayerKind = INTEL_AFFINE;
    nnet.pLayers[0].type = INTEL_INPUT_OUTPUT;
    nnet.pLayers[0].pLayerStruct = &affine_layer;
    nnet.pLayers[0].pInputs = nullptr;
    nnet.pLayers[0].pOutputs = nullptr;

    if (pwlEnabled)
    {
        nnet.pLayers[0].pOutputsIntermediate = tmp_outputs;
        nnet.pLayers[0].nBytesPerOutput = sizeof(int16_t);
    }
    else
    {
        nnet.pLayers[0].pOutputsIntermediate = nullptr;
        nnet.pLayers[0].nBytesPerOutput = sizeof(int32_t);
    }
}

void SetupDnnModel_1::samplePwl(intel_pwl_segment_t *segments, uint32_t nSegments)
{
    auto xBase = INT32_MIN;
    auto xBaseInc = UINT32_MAX / nSegments;
    auto yBase = INT32_MAX;
    auto yBaseInc = UINT16_MAX / nSegments;
    for (auto i = uint32_t{0}; i < nSegments; i++, xBase += xBaseInc, yBase += yBaseInc)
    {
        segments[i].xBase = xBase;
        segments[i].yBase = yBase;
        segments[i].slope = 1;
    }
}
