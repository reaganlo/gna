/*
 INTEL CONFIDENTIAL
 Copyright 2019 Intel Corporation.

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

enum TlvStatus
{
    TLV_SUCCESS = 0,
    TLV_ERROR_DATA_NULL = -1,
    TLV_ERROR_ZERO_LENGTH = -2,
    TLV_ERROR_ARGS_OUT_OF_RANGE = -3,
    TLV_ERROR_NODES_THE_SAME = -4,
    TLV_ERROR_MEMORY_ALLOC = -5,
    TLV_ERROR_INVALID_SIZE = -6,
    TLV_ERROR_EXCEEDED_MAX = -7,
    TLV_ERROR_MEMORY_OVERRUN = -8,
};
