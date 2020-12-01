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

static const HwSupport HW_GMM =
{
    {Gna2DeviceGenerationGmm, true},
    {Gna2DeviceGeneration0_9, true},
    {Gna2DeviceGeneration1_0, true},
    {Gna2DeviceGeneration2_0, true},
    {Gna2DeviceGeneration3_0, true},
    {Gna2DeviceGeneration3_5, true},
};

static const HwSupport HW_0_9 =
{
    {Gna2DeviceGeneration0_9, true},
    {Gna2DeviceGeneration1_0, true},
    {Gna2DeviceGeneration2_0, true},
    {Gna2DeviceGeneration3_0, true},
    {Gna2DeviceGeneration3_5, true},
};

static const HwSupport HW_1_0_AND_2_0 =
{
    {Gna2DeviceGeneration1_0, true},
    {Gna2DeviceGeneration2_0, true},
};

static const HwSupport HW_2_0 =
{
    {Gna2DeviceGeneration2_0, true},
    {Gna2DeviceGeneration3_0, true},
    {Gna2DeviceGeneration3_5, true},
};

static const HwSupport HW_3_0 =
{
    {Gna2DeviceGeneration3_0, true},
    {Gna2DeviceGeneration3_5, true},
};

static const Support FROM_GMM = { std::move(HW_GMM) };
static const Support FROM_0_9 = { std::move(HW_0_9) };
static const Support FROM_0_9_AUX = FROM_0_9; // Helper for changes of AUX layers
static const Support FROM_1_0_TILL_2_0 = { std::move(HW_1_0_AND_2_0) };
static const Support FROM_2_0 = { std::move(HW_2_0) };
static const Support FROM_3_0 = { std::move(HW_3_0) };

static const std::map<const nn_operation, const Support> FROM_1_0_GMM =
{
    {INTEL_GMM,                 FROM_GMM},
};

static const std::map<const nn_operation, const Support> FROM_3_0_AFF_RNN_CNN =
{
    {INTEL_AFFINE,              FROM_3_0},
    {INTEL_AFFINE_DIAGONAL,     FROM_3_0},
    {INTEL_AFFINE_MULTIBIAS,    FROM_3_0},
    {INTEL_RECURRENT,           FROM_3_0},  // TODO:3:CAPS: LOW priority in const weight mode
    {INTEL_CONVOLUTIONAL_2D,    FROM_3_0},
};

static const std::map<const nn_operation, const Support> FROM_0_9_COPY_TRANSPOSE =
{
    {INTEL_COPY,                FROM_0_9_AUX},
    {INTEL_DEINTERLEAVE,        FROM_0_9_AUX},
    {INTEL_INTERLEAVE,          FROM_0_9_AUX},
};

static const std::map<const nn_operation, const Support> FROM_3_0_COPY_TRANSPOSE =
{
    {INTEL_COPY,                FROM_3_0},
    {INTEL_DEINTERLEAVE,        FROM_3_0},
    {INTEL_INTERLEAVE,          FROM_3_0},
};

static const std::map<const nn_operation, const Support> FROM_3_0_AFF_RNN_CNN_AUX =
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

static const std::map<const nn_operation, const Support> FROM_3_0_AFF_CNN =
{
    {INTEL_AFFINE,              FROM_3_0},
    {INTEL_AFFINE_DIAGONAL,     FROM_3_0},
    {INTEL_AFFINE_MULTIBIAS,    FROM_3_0},
    {INTEL_CONVOLUTIONAL_2D,    FROM_3_0},
};

static const std::map<const nn_operation, const Support> FROM_3_0_AFF_RNN_CNN_MB_FALSE =
{
    {INTEL_AFFINE,              FROM_3_0},
    {INTEL_AFFINE_DIAGONAL,     FROM_3_0},
    {INTEL_RECURRENT,           FROM_3_0},    // TODO:3:CAPS: LOW priority in const weight mode
    {INTEL_CONVOLUTIONAL_2D,    FROM_3_0},
};

static const std::map<const nn_operation, const Support> FROM_3_0_AFF_CNN_MB_FALSE =
{
    {INTEL_AFFINE,              FROM_3_0},
    {INTEL_AFFINE_DIAGONAL,     FROM_3_0},
    {INTEL_CONVOLUTIONAL_2D,    FROM_3_0},
};

static const std::map<const nn_operation, const Support> FROM_3_0_CNN =
{
    {INTEL_CONVOLUTIONAL_2D,    FROM_3_0},
};

static const std::map<const nn_operation, const Support> FROM_3_0_CNN_MB =
{
    {INTEL_AFFINE_MULTIBIAS,    FROM_3_0},
    {INTEL_CONVOLUTIONAL_2D,    FROM_3_0},
};

static const std::map<const nn_operation, const Support> FROM_2_0_MB_3_0_CNN =
{
    {INTEL_AFFINE_MULTIBIAS,    FROM_2_0},
    {INTEL_CONVOLUTIONAL_2D,    FROM_3_0},
};

const std::map<const DataConfig, std::map<const nn_operation, const Support>>& DataConfig::Capabilities()
{
    static const std::map<const DataConfig, std::map<const nn_operation, const Support>> caps =
    {
        // input, weight/filter/mean, bias/covariance, output
        {{Gna2DataTypeUint8, Gna2DataTypeUint8, Gna2DataTypeUint32, Gna2DataTypeUint32},
            FROM_1_0_GMM
        },
        {{Gna2DataTypeUint8, Gna2DataTypeUint16, Gna2DataTypeUint32, Gna2DataTypeUint32},
            FROM_1_0_GMM
        },
        {{Gna2DataTypeInt8, DataMode{}, DataMode{}, Gna2DataTypeInt8},
            FROM_3_0_COPY_TRANSPOSE
        },
        {{Gna2DataTypeInt8, Gna2DataTypeInt8, Gna2DataTypeInt8, Gna2DataTypeInt8},
            FROM_3_0_AFF_RNN_CNN
        },
        {{Gna2DataTypeInt8, Gna2DataTypeInt8, Gna2DataTypeInt8, Gna2DataTypeInt16},
           FROM_3_0_AFF_CNN
        },
        {{Gna2DataTypeInt8, Gna2DataTypeInt8, Gna2DataTypeInt8, Gna2DataTypeInt32},
            FROM_3_0_AFF_CNN
        },
        {{Gna2DataTypeInt8, Gna2DataTypeInt8, Gna2DataTypeInt8, Gna2DataTypeInt32, true},
            FROM_3_0_AFF_CNN
        },
        {{Gna2DataTypeInt8, Gna2DataTypeInt8, Gna2DataTypeInt16, Gna2DataTypeInt8},
            FROM_3_0_AFF_RNN_CNN
        },
        {{Gna2DataTypeInt8, Gna2DataTypeInt8, Gna2DataTypeInt16, Gna2DataTypeInt16},
            FROM_3_0_AFF_CNN
        },
        {{Gna2DataTypeInt8, Gna2DataTypeInt8, Gna2DataTypeInt16, Gna2DataTypeInt32},
           FROM_3_0_AFF_CNN
        },
        {{Gna2DataTypeInt8, Gna2DataTypeInt8, Gna2DataTypeInt16, Gna2DataTypeInt32, true},
            FROM_3_0_AFF_CNN
        },
        {{Gna2DataTypeInt8, Gna2DataTypeInt8, Gna2DataTypeInt32, Gna2DataTypeInt8},
            FROM_3_0_AFF_RNN_CNN
        },
        {{Gna2DataTypeInt8, Gna2DataTypeInt8, Gna2DataTypeInt32, Gna2DataTypeInt16},
            FROM_3_0_AFF_CNN
        },
        {{Gna2DataTypeInt8, Gna2DataTypeInt8, Gna2DataTypeInt32, Gna2DataTypeInt32},
            FROM_3_0_AFF_CNN
        },
        {{Gna2DataTypeInt8, Gna2DataTypeInt8, Gna2DataTypeInt32, Gna2DataTypeInt32, true},
            FROM_3_0_AFF_CNN
        },
        {{Gna2DataTypeInt8, Gna2DataTypeInt8, DataMode{}, Gna2DataTypeInt8},
            FROM_3_0_AFF_RNN_CNN_MB_FALSE
        },
        {{Gna2DataTypeInt8, Gna2DataTypeInt8, DataMode{}, Gna2DataTypeInt16},
            FROM_3_0_AFF_CNN_MB_FALSE
        },
        {{Gna2DataTypeInt8, Gna2DataTypeInt8, DataMode{}, Gna2DataTypeInt32},
           FROM_3_0_AFF_CNN_MB_FALSE
        },
        {{Gna2DataTypeInt8, Gna2DataTypeInt8, DataMode{}, Gna2DataTypeInt32, true},
            FROM_3_0_AFF_CNN_MB_FALSE
        },
        {{Gna2DataTypeInt8, Gna2DataTypeInt16, Gna2DataTypeInt8, Gna2DataTypeInt8},
            FROM_3_0_AFF_RNN_CNN
        },
        {{Gna2DataTypeInt8, Gna2DataTypeInt16, Gna2DataTypeInt8, Gna2DataTypeInt16},
           FROM_3_0_AFF_CNN
        },
        {{Gna2DataTypeInt8, Gna2DataTypeInt16, Gna2DataTypeInt8, Gna2DataTypeInt32},
            FROM_3_0_AFF_CNN
        },
        {{Gna2DataTypeInt8, Gna2DataTypeInt16, Gna2DataTypeInt8, Gna2DataTypeInt32, true},
            FROM_3_0_AFF_CNN
        },
        {{Gna2DataTypeInt8, Gna2DataTypeInt16, Gna2DataTypeInt16, Gna2DataTypeInt8},
            FROM_3_0_AFF_RNN_CNN
        },
        {{Gna2DataTypeInt8, Gna2DataTypeInt16, Gna2DataTypeInt16, Gna2DataTypeInt16},
            FROM_3_0_AFF_CNN
        },
        {{Gna2DataTypeInt8, Gna2DataTypeInt16, Gna2DataTypeInt16, Gna2DataTypeInt32},
           FROM_3_0_AFF_CNN
        },
        {{Gna2DataTypeInt8, Gna2DataTypeInt16, Gna2DataTypeInt16, Gna2DataTypeInt32, true},
            FROM_3_0_AFF_CNN
        },
        {{Gna2DataTypeInt8, Gna2DataTypeInt16, Gna2DataTypeInt32, Gna2DataTypeInt8},
            FROM_3_0_AFF_RNN_CNN
        },
        {{Gna2DataTypeInt8, Gna2DataTypeInt16, Gna2DataTypeInt32, Gna2DataTypeInt16},
            FROM_3_0_AFF_CNN
        },
        {{Gna2DataTypeInt8, Gna2DataTypeInt16, Gna2DataTypeInt32, Gna2DataTypeInt32},
            FROM_3_0_AFF_CNN
        },
        {{Gna2DataTypeInt8, Gna2DataTypeInt16, Gna2DataTypeInt32, Gna2DataTypeInt32, true},
            FROM_3_0_AFF_CNN
        },
        {{Gna2DataTypeInt8, Gna2DataTypeInt16, DataMode{}, Gna2DataTypeInt8},
            FROM_3_0_AFF_RNN_CNN_MB_FALSE
        },
        {{Gna2DataTypeInt8, Gna2DataTypeInt16, DataMode{}, Gna2DataTypeInt16},
            FROM_3_0_AFF_CNN_MB_FALSE
        },
        {{Gna2DataTypeInt8, Gna2DataTypeInt16, DataMode{}, Gna2DataTypeInt32},
           FROM_3_0_AFF_CNN_MB_FALSE
        },
        {{Gna2DataTypeInt8, Gna2DataTypeInt16, DataMode{}, Gna2DataTypeInt32, true},
            FROM_3_0_AFF_CNN_MB_FALSE
        },

        // 2B Input
        {{Gna2DataTypeInt16, DataMode{}, DataMode{}, Gna2DataTypeInt16},
            FROM_0_9_COPY_TRANSPOSE
        },
        {{Gna2DataTypeInt16, Gna2DataTypeInt8, Gna2DataTypeInt8, Gna2DataTypeInt8},
            FROM_3_0_CNN_MB
        },
        {{Gna2DataTypeInt16, Gna2DataTypeInt8, Gna2DataTypeInt8, Gna2DataTypeInt16},
           FROM_3_0_CNN_MB
        },
        {{Gna2DataTypeInt16, Gna2DataTypeInt8, Gna2DataTypeInt8, Gna2DataTypeInt32},
            FROM_3_0_CNN_MB
        },
        {{Gna2DataTypeInt16, Gna2DataTypeInt8, Gna2DataTypeInt8, Gna2DataTypeInt32, true},
            FROM_3_0_CNN_MB
        },
        {{Gna2DataTypeInt16, Gna2DataTypeInt8, Gna2DataTypeInt16, Gna2DataTypeInt8},
            FROM_3_0_CNN_MB
        },
        {{Gna2DataTypeInt16, Gna2DataTypeInt8, Gna2DataTypeInt16, Gna2DataTypeInt16},
            FROM_3_0_CNN_MB
        },
        {{Gna2DataTypeInt16, Gna2DataTypeInt8, Gna2DataTypeInt16, Gna2DataTypeInt32},
           FROM_3_0_CNN_MB
        },
        {{Gna2DataTypeInt16, Gna2DataTypeInt8, Gna2DataTypeInt16, Gna2DataTypeInt32, true},
            FROM_3_0_CNN_MB
        },
        {{Gna2DataTypeInt16, Gna2DataTypeInt8, Gna2DataTypeInt32, Gna2DataTypeInt8},
            FROM_3_0_CNN_MB
        },
        {{Gna2DataTypeInt16, Gna2DataTypeInt8, Gna2DataTypeInt32, Gna2DataTypeInt16},
            FROM_2_0_MB_3_0_CNN
        },
        {{Gna2DataTypeInt16, Gna2DataTypeInt8, Gna2DataTypeInt32, Gna2DataTypeInt32},
            FROM_3_0_CNN_MB
        },
        {{Gna2DataTypeInt16, Gna2DataTypeInt8, Gna2DataTypeInt32, Gna2DataTypeInt32, true},
            FROM_2_0_MB_3_0_CNN
        },
        {{Gna2DataTypeInt16, Gna2DataTypeInt8, DataMode{}, Gna2DataTypeInt8},
            FROM_3_0_CNN
        },
        {{Gna2DataTypeInt16, Gna2DataTypeInt8, DataMode{}, Gna2DataTypeInt16},
            FROM_3_0_CNN
        },
        {{Gna2DataTypeInt16, Gna2DataTypeInt8, DataMode{}, Gna2DataTypeInt32},
           FROM_3_0_CNN
        },
        {{Gna2DataTypeInt16, Gna2DataTypeInt8, DataMode{}, Gna2DataTypeInt32, true},
            FROM_3_0_CNN
        },
        {{Gna2DataTypeInt16, Gna2DataTypeInt8, Gna2DataTypeCompoundBias, Gna2DataTypeInt8},
            {
                {INTEL_AFFINE,              FROM_3_0},
                {INTEL_AFFINE_DIAGONAL,     FROM_3_0},
            }
        },
        {{Gna2DataTypeInt16, Gna2DataTypeInt8, Gna2DataTypeCompoundBias, Gna2DataTypeInt16},
            {
                {INTEL_AFFINE,              FROM_0_9},
                {INTEL_AFFINE_DIAGONAL,     FROM_0_9},
                {INTEL_AFFINE_MULTIBIAS,    FROM_2_0},
                {INTEL_RECURRENT,           FROM_0_9},
            }
        },
        {{Gna2DataTypeInt16, Gna2DataTypeInt8, Gna2DataTypeCompoundBias, Gna2DataTypeInt32},
            {
                {INTEL_AFFINE,              FROM_3_0},
                {INTEL_AFFINE_DIAGONAL,     FROM_3_0},
            }
        },
        {{Gna2DataTypeInt16, Gna2DataTypeInt8, Gna2DataTypeCompoundBias, Gna2DataTypeInt32, true},
             {
                {INTEL_AFFINE,              FROM_0_9},
                {INTEL_AFFINE_DIAGONAL,     FROM_0_9},
                {INTEL_AFFINE_MULTIBIAS,    FROM_2_0},
            }
        },
        {{Gna2DataTypeInt16, Gna2DataTypeInt16, Gna2DataTypeInt8, Gna2DataTypeInt8},
            FROM_3_0_AFF_CNN
        },
        {{Gna2DataTypeInt16, Gna2DataTypeInt16, Gna2DataTypeInt8, Gna2DataTypeInt16},
           FROM_3_0_AFF_RNN_CNN_AUX
        },
        {{Gna2DataTypeInt16, Gna2DataTypeInt16, Gna2DataTypeInt8, Gna2DataTypeInt32},
            FROM_3_0_AFF_CNN
        },
        {{Gna2DataTypeInt16, Gna2DataTypeInt16, Gna2DataTypeInt8, Gna2DataTypeInt32, true},
            FROM_3_0_AFF_CNN
        },
        {{Gna2DataTypeInt16, Gna2DataTypeInt16, Gna2DataTypeInt16, Gna2DataTypeInt8},
            FROM_3_0_AFF_CNN
        },
        {{Gna2DataTypeInt16, Gna2DataTypeInt16, Gna2DataTypeInt16, Gna2DataTypeInt16},
            FROM_3_0_AFF_RNN_CNN_AUX
        },
        {{Gna2DataTypeInt16, Gna2DataTypeInt16, Gna2DataTypeInt16, Gna2DataTypeInt32},
           FROM_3_0_AFF_CNN
        },
        {{Gna2DataTypeInt16, Gna2DataTypeInt16, Gna2DataTypeInt16, Gna2DataTypeInt32, true},
            FROM_3_0_AFF_CNN
        },
        {{Gna2DataTypeInt16, Gna2DataTypeInt16, Gna2DataTypeInt32, Gna2DataTypeInt8},
            FROM_3_0_AFF_CNN
        },
        {{Gna2DataTypeInt16, Gna2DataTypeInt16, Gna2DataTypeInt32, Gna2DataTypeInt16},
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
        {{Gna2DataTypeInt16, Gna2DataTypeInt16, Gna2DataTypeInt32, Gna2DataTypeInt32},
            FROM_3_0_AFF_CNN
        },
        {{Gna2DataTypeInt16, Gna2DataTypeInt16, Gna2DataTypeInt32, Gna2DataTypeInt32, true},
            {
                {INTEL_AFFINE,              FROM_0_9},
                {INTEL_AFFINE_DIAGONAL,     FROM_0_9},
                {INTEL_AFFINE_MULTIBIAS,    FROM_2_0},
                {INTEL_CONVOLUTIONAL,       FROM_1_0_TILL_2_0},
                {INTEL_CONVOLUTIONAL_2D,    FROM_3_0}
            }
        },
        {{Gna2DataTypeInt16, Gna2DataTypeInt16, DataMode{}, Gna2DataTypeInt8},
             FROM_3_0_AFF_CNN_MB_FALSE
        },
        {{Gna2DataTypeInt16, Gna2DataTypeInt16, DataMode{}, Gna2DataTypeInt16},
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
        {{Gna2DataTypeInt16, Gna2DataTypeInt16, DataMode{}, Gna2DataTypeInt32},
           FROM_3_0_AFF_CNN_MB_FALSE
        },
        {{Gna2DataTypeInt16, Gna2DataTypeInt16, DataMode{}, Gna2DataTypeInt32, true},
            FROM_3_0_AFF_CNN_MB_FALSE
        },
    };
    return caps;
}
