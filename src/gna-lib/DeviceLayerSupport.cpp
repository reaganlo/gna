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

#include "DeviceLayerSupport.h"

using namespace GNA;

const ApiSupport API_1_0 =
{
    {GNA_API_1_0, true},
    {GNA_API_2_0, true},
    {GNA_API_3_0, true},
};

const ApiSupport API_2_0 =
{
    {GNA_API_2_0, true},
    {GNA_API_3_0, true},
};

const ApiSupport API_3_0 =
{
    {GNA_API_3_0, true},
};

const HwSupport HW_GMM =
{
    {GMM_DEVICE, true},
    {GNA_0_9, true},
    {GNA_1_0, true},
    {GNA_2_0, true},
    {GNA_3_0, true},
};

const HwSupport HW_0_9 =
{
    {GNA_0_9, true},
    {GNA_1_0, true},
    {GNA_2_0, true},
    {GNA_3_0, true},
};

const HwSupport HW_1_0_AND_2_0 =
{
    {GNA_1_0, true},
    {GNA_2_0, true},
};

const HwSupport HW_2_0 =
{
    {GNA_2_0, true},
    {GNA_3_0, true},
};

const HwSupport HW_3_0 =
{
    {GNA_3_0, true},
};

static const Support FROM_GMM = { API_1_0, HW_GMM};
static const Support FROM_0_9 = { API_1_0, HW_0_9};
static const Support FROM_0_9_AUX = FROM_0_9; // Helper for changes of AUX layers
static const Support FROM_1_0_TILL_2_0 = { API_1_0, HW_1_0_AND_2_0 };
static const Support FROM_2_0 = { API_2_0, HW_2_0 };
static const Support FROM_3_0 = { API_3_0, HW_3_0 };

static const std::map<const gna_layer_operation, const Support> FROM_1_0_GMM =
{
    {INTEL_GMM,                 FROM_GMM},
};

static const std::map<const gna_layer_operation, const Support> FROM_3_0_AFF_RNN_CNN =
{
    {INTEL_AFFINE,              FROM_3_0},
    {INTEL_AFFINE_DIAGONAL,     FROM_3_0},
    {INTEL_AFFINE_MULTIBIAS,    FROM_3_0},
    {INTEL_RECURRENT,           FROM_3_0},  // TODO:3:CAPS: LOW priority in const weight mode
    {INTEL_CONVOLUTIONAL_2D,    FROM_3_0},
};

static const std::map<const gna_layer_operation, const Support> FROM_0_9_COPY_TRANSPOSE =
{
    {INTEL_COPY,                FROM_0_9_AUX},
    {INTEL_DEINTERLEAVE,        FROM_0_9_AUX},
    {INTEL_INTERLEAVE,          FROM_0_9_AUX},
};

static const std::map<const gna_layer_operation, const Support> FROM_3_0_COPY_TRANSPOSE =
{
    {INTEL_COPY,                FROM_3_0},
    {INTEL_DEINTERLEAVE,        FROM_3_0},
    {INTEL_INTERLEAVE,          FROM_3_0},
};

static const std::map<const gna_layer_operation, const Support> FROM_3_0_AFF_RNN_CNN_AUX =
{
    {INTEL_AFFINE,              FROM_3_0},
    {INTEL_AFFINE_DIAGONAL,     FROM_3_0},
    {INTEL_AFFINE_MULTIBIAS,    FROM_3_0},
    {INTEL_RECURRENT,           FROM_3_0},    // TODO:3:CAPS: LOW priority in const weight mode
    {INTEL_CONVOLUTIONAL_2D,    FROM_3_0},
    {INTEL_COPY,                FROM_0_9_AUX},//FROM_3_0
    {INTEL_DEINTERLEAVE,        FROM_0_9_AUX},//FROM_3_0
    {INTEL_INTERLEAVE,          FROM_0_9_AUX},//FROM_3_0
};

static const std::map<const gna_layer_operation, const Support> FROM_3_0_AFF_CNN =
{
    {INTEL_AFFINE,              FROM_3_0},
    {INTEL_AFFINE_DIAGONAL,     FROM_3_0},
    {INTEL_AFFINE_MULTIBIAS,    FROM_3_0},
    {INTEL_CONVOLUTIONAL_2D,    FROM_3_0},
};

static const std::map<const gna_layer_operation, const Support> FROM_3_0_AFF_CNN_GMM =
{
    {INTEL_AFFINE,              FROM_3_0},
    {INTEL_AFFINE_DIAGONAL,     FROM_3_0},
    {INTEL_AFFINE_MULTIBIAS,    FROM_3_0},
    {INTEL_CONVOLUTIONAL_2D,    FROM_3_0},
};

static const std::map<const gna_layer_operation, const Support> FROM_3_0_AFF_RNN_CNN_MB_FALSE =
{
    {INTEL_AFFINE,              FROM_3_0},
    {INTEL_AFFINE_DIAGONAL,     FROM_3_0},
    {INTEL_RECURRENT,           FROM_3_0},    // TODO:3:CAPS: LOW priority in const weight mode
    {INTEL_CONVOLUTIONAL_2D,    FROM_3_0},
};

static const std::map<const gna_layer_operation, const Support> FROM_3_0_AFF_CNN_MB_FALSE =
{
    {INTEL_AFFINE,              FROM_3_0},
    {INTEL_AFFINE_DIAGONAL,     FROM_3_0},
    {INTEL_CONVOLUTIONAL_2D,    FROM_3_0},
};

static const std::map<const gna_layer_operation, const Support> FROM_3_0_CNN =
{
    {INTEL_CONVOLUTIONAL_2D,    FROM_3_0},
};

static const std::map<const gna_layer_operation, const Support> FROM_3_0_CNN_MB =
{
    {INTEL_AFFINE_MULTIBIAS,    FROM_3_0},
    {INTEL_CONVOLUTIONAL_2D,    FROM_3_0},
};

static const std::map<const gna_layer_operation, const Support> FROM_2_0_MB_3_0_CNN =
{
    {INTEL_AFFINE_MULTIBIAS,    FROM_2_0},
    {INTEL_CONVOLUTIONAL_2D,    FROM_3_0},
};

static const std::map<const gna_layer_operation, const Support> FROM_3_0_CNN_AUX =
{
    {INTEL_CONVOLUTIONAL_2D,    FROM_3_0},
    {INTEL_COPY,                FROM_0_9_AUX},
    {INTEL_DEINTERLEAVE,        FROM_0_9_AUX},
    {INTEL_INTERLEAVE,          FROM_0_9_AUX},
};

static const std::map<const gna_layer_operation, const Support> FROM_3_0_CNN_MB_AUX =
{
    {INTEL_AFFINE_MULTIBIAS,    FROM_3_0},
    {INTEL_CONVOLUTIONAL_2D,    FROM_3_0},
    {INTEL_COPY,                FROM_0_9_AUX},
    {INTEL_DEINTERLEAVE,        FROM_0_9_AUX},
    {INTEL_INTERLEAVE,          FROM_0_9_AUX},
};

static const std::map<const gna_layer_operation, const Support> FROM_0_9_AUX_2_0_MB_3_0_CNN =
{
    {INTEL_AFFINE_MULTIBIAS,    FROM_2_0},
    {INTEL_CONVOLUTIONAL_2D,    FROM_3_0},
    {INTEL_COPY,                FROM_0_9_AUX},
    {INTEL_DEINTERLEAVE,        FROM_0_9_AUX},
    {INTEL_INTERLEAVE,          FROM_0_9_AUX},
};

static const std::map<const gna_layer_operation, const Support> FROM_3_0_INPUT_DISABLED =
{
    //{INTEL_AFFINE,              FROM_3_0}, // TODO:3:CAPS: Low priority
    {INTEL_AFFINE_DIAGONAL,     FROM_3_0},
    {INTEL_AFFINE_MULTIBIAS,    FROM_3_0},
    //{INTEL_RECURRENT,           FROM_3_0}, // TODO:3:CAPS: Low priority
    {INTEL_CONVOLUTIONAL_2D,    FROM_3_0},
    {INTEL_COPY,                FROM_3_0},
    {INTEL_DEINTERLEAVE,        FROM_3_0},
    {INTEL_INTERLEAVE,          FROM_3_0},
};

static const std::map<const gna_layer_operation, const Support> FROM_3_0_INPUT_DISABLED_BIAS_DISABLED =
{
    //{INTEL_AFFINE,              FROM_3_0}, // TODO:3:CAPS: Low priority
    //{INTEL_AFFINE_DIAGONAL,     FROM_3_0},  // TODO:3:CAPS: Low priority
    //{INTEL_AFFINE_MULTIBIAS,    FROM_3_0},  // TODO:3:CAPS: Low priority
    //{INTEL_RECURRENT,           FROM_3_0}, // TODO:3:CAPS: Low priority
    //{INTEL_CONVOLUTIONAL_2D,    FROM_3_0},  // TODO:3:CAPS: Low priority
    //{INTEL_COPY,                FROM_3_0},  // TODO:3:CAPS: Low priority
    //{INTEL_DEINTERLEAVE,        FROM_3_0},  // TODO:3:CAPS: Low priority
    //{INTEL_INTERLEAVE,          FROM_3_0},  // TODO:3:CAPS: Low priority
};

bool Support::IsSupported() const
{
    return !API_1_0.empty() && !HW_0_9.empty();
}

const std::map<const DataConfig, std::map<const gna_layer_operation, const Support>> DataConfig::Capabilities =
{
    // input, weight/filter/mean, bias/covariance, output
    {{GNA_UINT8, GNA_UINT8, GNA_UINT32, GNA_UINT32},
        FROM_1_0_GMM
    },
    {{GNA_UINT8, GNA_UINT16, GNA_UINT32, GNA_UINT32},
        FROM_1_0_GMM
    },
    {{GNA_INT8, GNA_DATA_DISABLED, GNA_DATA_DISABLED, GNA_INT8},
        FROM_3_0_COPY_TRANSPOSE
    },
    {{GNA_INT8, GNA_INT8, GNA_INT8, GNA_INT8},
        FROM_3_0_AFF_RNN_CNN
    },
    {{GNA_INT8, GNA_INT8, GNA_INT8, GNA_INT16},
       FROM_3_0_AFF_CNN
    },
    {{GNA_INT8, GNA_INT8, GNA_INT8, GNA_INT32},
        FROM_3_0_AFF_CNN
    },
    {{GNA_INT8, GNA_INT8, GNA_INT8, GNA_DATA_ACTIVATION_DISABLED},
        FROM_3_0_AFF_CNN
    },
    {{GNA_INT8, GNA_INT8, GNA_INT16, GNA_INT8},
        FROM_3_0_AFF_RNN_CNN
    },
    {{GNA_INT8, GNA_INT8, GNA_INT16, GNA_INT16},
        FROM_3_0_AFF_CNN
    },
    {{GNA_INT8, GNA_INT8, GNA_INT16, GNA_INT32},
       FROM_3_0_AFF_CNN
    },
    {{GNA_INT8, GNA_INT8, GNA_INT16, GNA_DATA_ACTIVATION_DISABLED},
        FROM_3_0_AFF_CNN
    },
    {{GNA_INT8, GNA_INT8, GNA_INT32, GNA_INT8},
        FROM_3_0_AFF_RNN_CNN
    },
    {{GNA_INT8, GNA_INT8, GNA_INT32, GNA_INT16},
        FROM_3_0_AFF_CNN
    },
    {{GNA_INT8, GNA_INT8, GNA_INT32, GNA_INT32},
        FROM_3_0_AFF_CNN_GMM
    },
    {{GNA_INT8, GNA_INT8, GNA_INT32, GNA_DATA_ACTIVATION_DISABLED},
        {
            {INTEL_AFFINE,              FROM_3_0},
            {INTEL_AFFINE_DIAGONAL,     FROM_3_0},
            {INTEL_AFFINE_MULTIBIAS,    FROM_3_0},
            {INTEL_CONVOLUTIONAL_2D,    FROM_3_0},
        }
    },
    {{GNA_INT8, GNA_INT8, GNA_DATA_DISABLED, GNA_INT8},
        FROM_3_0_AFF_RNN_CNN_MB_FALSE
    },
    {{GNA_INT8, GNA_INT8, GNA_DATA_DISABLED, GNA_INT16},
        FROM_3_0_AFF_CNN_MB_FALSE
    },
    {{GNA_INT8, GNA_INT8, GNA_DATA_DISABLED, GNA_INT32},
       FROM_3_0_AFF_CNN_MB_FALSE
    },
    {{GNA_INT8, GNA_INT8, GNA_DATA_DISABLED, GNA_DATA_ACTIVATION_DISABLED},
        FROM_3_0_AFF_CNN_MB_FALSE
    },
    {{GNA_INT8, GNA_INT16, GNA_INT8, GNA_INT8},
        FROM_3_0_AFF_RNN_CNN
    },
    {{GNA_INT8, GNA_INT16, GNA_INT8, GNA_INT16},
       FROM_3_0_AFF_CNN
    },
    {{GNA_INT8, GNA_INT16, GNA_INT8, GNA_INT32},
        FROM_3_0_AFF_CNN
    },
    {{GNA_INT8, GNA_INT16, GNA_INT8, GNA_DATA_ACTIVATION_DISABLED},
        FROM_3_0_AFF_CNN
    },
    {{GNA_INT8, GNA_INT16, GNA_INT16, GNA_INT8},
        FROM_3_0_AFF_RNN_CNN
    },
    {{GNA_INT8, GNA_INT16, GNA_INT16, GNA_INT16},
        FROM_3_0_AFF_CNN
    },
    {{GNA_INT8, GNA_INT16, GNA_INT16, GNA_INT32},
       FROM_3_0_AFF_CNN
    },
    {{GNA_INT8, GNA_INT16, GNA_INT16, GNA_DATA_ACTIVATION_DISABLED},
        FROM_3_0_AFF_CNN
    },
    {{GNA_INT8, GNA_INT16, GNA_INT32, GNA_INT8},
        FROM_3_0_AFF_RNN_CNN
    },
    {{GNA_INT8, GNA_INT16, GNA_INT32, GNA_INT16},
        FROM_3_0_AFF_CNN
    },
    {{GNA_INT8, GNA_INT16, GNA_INT32, GNA_INT32},
        FROM_3_0_AFF_CNN_GMM
    },
    {{GNA_INT8, GNA_INT16, GNA_INT32, GNA_DATA_ACTIVATION_DISABLED},
        {
            {INTEL_AFFINE,              FROM_3_0},
            {INTEL_AFFINE_DIAGONAL,     FROM_3_0},
            {INTEL_AFFINE_MULTIBIAS,    FROM_3_0},
            {INTEL_CONVOLUTIONAL_2D,    FROM_3_0},
        }
    },
    {{GNA_INT8, GNA_INT16, GNA_DATA_DISABLED, GNA_INT8},
        FROM_3_0_AFF_RNN_CNN_MB_FALSE
    },
    {{GNA_INT8, GNA_INT16, GNA_DATA_DISABLED, GNA_INT16},
        FROM_3_0_AFF_CNN_MB_FALSE
    },
    {{GNA_INT8, GNA_INT16, GNA_DATA_DISABLED, GNA_INT32},
       FROM_3_0_AFF_CNN_MB_FALSE
    },
    {{GNA_INT8, GNA_INT16, GNA_DATA_DISABLED, GNA_DATA_ACTIVATION_DISABLED},
        FROM_3_0_AFF_CNN_MB_FALSE
    },

    // 2B Input
    {{GNA_INT16, GNA_DATA_DISABLED, GNA_DATA_DISABLED, GNA_INT16},
        FROM_0_9_COPY_TRANSPOSE
    },
    {{GNA_INT16, GNA_INT8, GNA_INT8, GNA_INT8},
        FROM_3_0_CNN_MB
    },
    {{GNA_INT16, GNA_INT8, GNA_INT8, GNA_INT16},
       FROM_3_0_CNN_MB_AUX
    },
    {{GNA_INT16, GNA_INT8, GNA_INT8, GNA_INT32},
        FROM_3_0_CNN_MB
    },
    {{GNA_INT16, GNA_INT8, GNA_INT8, GNA_DATA_ACTIVATION_DISABLED},
        FROM_3_0_CNN_MB
    },
    {{GNA_INT16, GNA_INT8, GNA_INT16, GNA_INT8},
        FROM_3_0_CNN_MB
    },
    {{GNA_INT16, GNA_INT8, GNA_INT16, GNA_INT16},
        FROM_3_0_CNN_MB_AUX
    },
    {{GNA_INT16, GNA_INT8, GNA_INT16, GNA_INT32},
       FROM_3_0_CNN_MB
    },
    {{GNA_INT16, GNA_INT8, GNA_INT16, GNA_DATA_ACTIVATION_DISABLED},
        FROM_3_0_CNN_MB
    },
    {{GNA_INT16, GNA_INT8, GNA_INT32, GNA_INT8},
        FROM_3_0_CNN_MB
    },
    {{GNA_INT16, GNA_INT8, GNA_INT32, GNA_INT16},
        FROM_0_9_AUX_2_0_MB_3_0_CNN
    },
    {{GNA_INT16, GNA_INT8, GNA_INT32, GNA_INT32},
        FROM_3_0_CNN_MB
    },
    {{GNA_INT16, GNA_INT8, GNA_INT32, GNA_DATA_ACTIVATION_DISABLED},
        FROM_2_0_MB_3_0_CNN
    },
    {{GNA_INT16, GNA_INT8, GNA_DATA_DISABLED, GNA_INT8},
        FROM_3_0_AFF_CNN_MB_FALSE
    },
    {{GNA_INT16, GNA_INT8, GNA_DATA_DISABLED, GNA_INT16},
        {
            {INTEL_AFFINE,              FROM_3_0},
            {INTEL_AFFINE_DIAGONAL,     FROM_3_0},
            {INTEL_RECURRENT,           FROM_3_0},    // TODO:3:CAPS: LOW priority in const weight mode
            {INTEL_CONVOLUTIONAL_2D,    FROM_3_0},
            {INTEL_COPY,                FROM_0_9_AUX},
            {INTEL_DEINTERLEAVE,        FROM_0_9_AUX},
            {INTEL_INTERLEAVE,          FROM_0_9_AUX},
        }
    },
    {{GNA_INT16, GNA_INT8, GNA_DATA_DISABLED, GNA_INT32},
       FROM_3_0_AFF_CNN_MB_FALSE
    },
    {{GNA_INT16, GNA_INT8, GNA_DATA_DISABLED, GNA_DATA_ACTIVATION_DISABLED},
        FROM_3_0_AFF_CNN_MB_FALSE
    },
    {{GNA_INT16, GNA_INT8, GNA_DATA_RICH_FORMAT, GNA_INT8},
        {
            {INTEL_AFFINE,              FROM_3_0},
            {INTEL_AFFINE_DIAGONAL,     FROM_3_0},
        }
    },
    {{GNA_INT16, GNA_INT8, GNA_DATA_RICH_FORMAT, GNA_INT16},
        {
            {INTEL_AFFINE,              FROM_0_9},
            {INTEL_AFFINE_DIAGONAL,     FROM_0_9},
            {INTEL_AFFINE_MULTIBIAS,    FROM_2_0},
            {INTEL_RECURRENT,           FROM_0_9},
            {INTEL_COPY,                FROM_0_9_AUX},
            {INTEL_DEINTERLEAVE,        FROM_0_9_AUX},
            {INTEL_INTERLEAVE,          FROM_0_9_AUX},
        }
    },
    {{GNA_INT16, GNA_INT8, GNA_DATA_RICH_FORMAT, GNA_INT32},
        {
            {INTEL_AFFINE,              FROM_3_0},
            {INTEL_AFFINE_DIAGONAL,     FROM_3_0},
        }
    },
    {{GNA_INT16, GNA_INT8, GNA_DATA_RICH_FORMAT, GNA_DATA_ACTIVATION_DISABLED},
         {
            {INTEL_AFFINE,              FROM_0_9},
            {INTEL_AFFINE_DIAGONAL,     FROM_0_9},
            {INTEL_AFFINE_MULTIBIAS,    FROM_2_0},
        }
    },
    {{GNA_INT16, GNA_INT16, GNA_INT8, GNA_INT8},
        FROM_3_0_AFF_CNN
    },
    {{GNA_INT16, GNA_INT16, GNA_INT8, GNA_INT16},
       FROM_3_0_AFF_RNN_CNN_AUX
    },
    {{GNA_INT16, GNA_INT16, GNA_INT8, GNA_INT32},
        FROM_3_0_AFF_CNN
    },
    {{GNA_INT16, GNA_INT16, GNA_INT8, GNA_DATA_ACTIVATION_DISABLED},
        FROM_3_0_AFF_CNN
    },
    {{GNA_INT16, GNA_INT16, GNA_INT16, GNA_INT8},
        FROM_3_0_AFF_CNN
    },
    {{GNA_INT16, GNA_INT16, GNA_INT16, GNA_INT16},
        FROM_3_0_AFF_RNN_CNN_AUX
    },
    {{GNA_INT16, GNA_INT16, GNA_INT16, GNA_INT32},
       FROM_3_0_AFF_CNN
    },
    {{GNA_INT16, GNA_INT16, GNA_INT16, GNA_DATA_ACTIVATION_DISABLED},
        FROM_3_0_AFF_CNN
    },
    {{GNA_INT16, GNA_INT16, GNA_INT32, GNA_INT8},
        FROM_3_0_AFF_CNN
    },
    {{GNA_INT16, GNA_INT16, GNA_INT32, GNA_INT16},
        {
            {INTEL_AFFINE,              FROM_0_9},
            {INTEL_AFFINE_DIAGONAL,     FROM_0_9},
            {INTEL_AFFINE_MULTIBIAS,    FROM_2_0},
            {INTEL_RECURRENT,           FROM_0_9},
            {INTEL_CONVOLUTIONAL,       FROM_1_0_TILL_2_0},
            {INTEL_CONVOLUTIONAL_2D,    FROM_3_0},
            {INTEL_COPY,                FROM_0_9_AUX},
            {INTEL_DEINTERLEAVE,        FROM_0_9_AUX},
            {INTEL_INTERLEAVE,          FROM_0_9_AUX},
        }
    },
    {{GNA_INT16, GNA_INT16, GNA_INT32, GNA_INT32},
        FROM_3_0_AFF_CNN
    },
    {{GNA_INT16, GNA_INT16, GNA_INT32, GNA_DATA_ACTIVATION_DISABLED},
        {
            {INTEL_AFFINE,              FROM_0_9},
            {INTEL_AFFINE_DIAGONAL,     FROM_0_9},
            {INTEL_AFFINE_MULTIBIAS,    FROM_2_0},
            {INTEL_CONVOLUTIONAL,       FROM_1_0_TILL_2_0},
            {INTEL_CONVOLUTIONAL_2D,    FROM_3_0}
        }
    },
    {{GNA_INT16, GNA_INT16, GNA_DATA_DISABLED, GNA_INT8},
         FROM_3_0_AFF_CNN_MB_FALSE
    },
    {{GNA_INT16, GNA_INT16, GNA_DATA_DISABLED, GNA_INT16},
        {
            {INTEL_AFFINE,              FROM_3_0},
            {INTEL_AFFINE_DIAGONAL,     FROM_3_0},
            {INTEL_RECURRENT,           FROM_3_0},    // TODO:3:CAPS: LOW priority in const weight mode
            {INTEL_CONVOLUTIONAL_2D,    FROM_3_0},
            {INTEL_COPY,                FROM_0_9_AUX},
            {INTEL_DEINTERLEAVE,        FROM_0_9_AUX},
            {INTEL_INTERLEAVE,          FROM_0_9_AUX},
        }
    },
    {{GNA_INT16, GNA_INT16, GNA_DATA_DISABLED, GNA_INT32},
       FROM_3_0_AFF_CNN_MB_FALSE
    },
    {{GNA_INT16, GNA_INT16, GNA_DATA_DISABLED, GNA_DATA_ACTIVATION_DISABLED},
        FROM_3_0_AFF_CNN_MB_FALSE
    },
 };
