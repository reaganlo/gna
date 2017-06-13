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

#include "SetupRecurrentModel.h"

namespace
{
const int layersNum = 1;
const int groupingNum = 4;
const int inVecSz = 8;
const int outVecSz = 32;

const int8_t weights_1B[(outVecSz + inVecSz) * outVecSz] =
{
    -6, -2, -1, -1, -2,  9,  6,  5,  2,  4, -1,  5, -2, -4,  0,  9, -8,  8, -4,  6,  5,  3, -7, -9,  7,  0, -4, -1,  1,  7,  6, -6, 2, -8,  6,  5, 1, 2, 3, 4,
    -1, -2,  7,  5, -1,  4,  8,  7, -9, -1,  7,  1,  0, -2,  1,  0,  6, -6,  7,  4, -6,  0,  3, -2,  1,  8, -6, -2, -6, -3,  4, -2, 5,  6, -9, -5, 1, 2, 3, 4,
    -2, -5, -8, -6, -2, -7,  0,  6, -3, -1, -6,  4,  1, -4, -5, -3,  7,  9, -9,  9,  9,  0, -2,  6, -3,  5, -2, -1, -3, -5,  7,  6,  6, -8, 0, -4,1, 2, 3, 4,
     9,  2,  7, -8, -7,  8, -6, -6,  1,  7, -4, -4,  9, -6, -6,  5, -7, -6, -2, -1, -1, -2,  9,  6,  5,  2,  4, -1,  5, -2, -4,  0,  9, -8, 8, -4,1, 2, 3, 4,
    -6, -2, -1, -1, -2,  9,  6,  5,  2,  4, -1,  5, -2, -4,  0,  9, -8,  8, -4,  6,  5,  3, -7, -9,  7,  0, -4, -1,  1,  7,  6, -6, 2, -8,  6,  5, 1, 2, 3, 4,
    -1, -2,  7,  5, -1,  4,  8,  7, -9, -1,  7,  1,  0, -2,  1,  0,  6, -6,  7,  4, -6,  0,  3, -2,  1,  8, -6, -2, -6, -3,  4, -2, 5,  6, -9, -5, 1, 2, 3, 4,
    -2, -5, -8, -6, -2, -7,  0,  6, -3, -1, -6,  4,  1, -4, -5, -3,  7,  9, -9,  9,  9,  0, -2,  6, -3,  5, -2, -1, -3, -5,  7,  6,  6, -8, 0, -4,1, 2, 3, 4,
     9,  2,  7, -8, -7,  8, -6, -6,  1,  7, -4, -4,  9, -6, -6,  5, -7, -6, -2, -1, -1, -2,  9,  6,  5,  2,  4, -1,  5, -2, -4,  0,  9, -8,  8, -4,1, 2, 3, 4,
    -6, -2, -1, -1, -2,  9,  6,  5,  2,  4, -1,  5, -2, -4,  0,  9, -8,  8, -4,  6,  5,  3, -7, -9,  7,  0, -4, -1,  1,  7,  6, -6, 2, -8,  6,  5, 1, 2, 3, 4,
    -1, -2,  7,  5, -1,  4,  8,  7, -9, -1,  7,  1,  0, -2,  1,  0,  6, -6,  7,  4, -6,  0,  3, -2,  1,  8, -6, -2, -6, -3,  4, -2, 5,  6, -9, -5, 1, 2, 3, 4,
    -2, -5, -8, -6, -2, -7,  0,  6, -3, -1, -6,  4,  1, -4, -5, -3,  7,  9, -9,  9,  9,  0, -2,  6, -3,  5, -2, -1, -3, -5,  7,  6,  6, -8, 0, -4,1, 2, 3, 4,
     9,  2,  7, -8, -7,  8, -6, -6,  1,  7, -4, -4,  9, -6, -6,  5, -7, -6, -2, -1, -1, -2,  9,  6,  5,  2,  4, -1,  5, -2, -4,  0,  9, -8,  8, -4,1, 2, 3, 4,
    -6, -2, -1, -1, -2,  9,  6,  5,  2,  4, -1,  5, -2, -4,  0,  9, -8,  8, -4,  6,  5,  3, -7, -9,  7,  0, -4, -1,  1,  7,  6, -6, 2, -8,  6,  5, 1, 2, 3, 4,
    -1, -2,  7,  5, -1,  4,  8,  7, -9, -1,  7,  1,  0, -2,  1,  0,  6, -6,  7,  4, -6,  0,  3, -2,  1,  8, -6, -2, -6, -3,  4, -2, 5,  6, -9, -5, 1, 2, 3, 4,
    -2, -5, -8, -6, -2, -7,  0,  6, -3, -1, -6,  4,  1, -4, -5, -3,  7,  9, -9,  9,  9,  0, -2,  6, -3,  5, -2, -1, -3, -5,  7,  6,  6, -8, 0, -4,1, 2, 3, 4,
     9,  2,  7, -8, -7,  8, -6, -6,  1,  7, -4, -4,  9, -6, -6,  5, -7, -6, -2, -1, -1, -2,  9,  6,  5,  2,  4, -1,  5, -2, -4,  0,  9, -8,  8, -4,1, 2, 3, 4,
    -6, -2, -1, -1, -2,  9,  6,  5,  2,  4, -1,  5, -2, -4,  0,  9, -8,  8, -4,  6,  5,  3, -7, -9,  7,  0, -4, -1,  1,  7,  6, -6, 2, -8,  6,  5, 1, 2, 3, 4,
    -1, -2,  7,  5, -1,  4,  8,  7, -9, -1,  7,  1,  0, -2,  1,  0,  6, -6,  7,  4, -6,  0,  3, -2,  1,  8, -6, -2, -6, -3,  4, -2, 5,  6, -9, -5, 1, 2, 3, 4,
    -2, -5, -8, -6, -2, -7,  0,  6, -3, -1, -6,  4,  1, -4, -5, -3,  7,  9, -9,  9,  9,  0, -2,  6, -3,  5, -2, -1, -3, -5,  7,  6,  6, -8, 0, -4,1, 2, 3, 4,
     9,  2,  7, -8, -7,  8, -6, -6,  1,  7, -4, -4,  9, -6, -6,  5, -7, -6, -2, -1, -1, -2,  9,  6,  5,  2,  4, -1,  5, -2, -4,  0,  9, -8, 8, -4,1, 2, 3, 4,
    -6, -2, -1, -1, -2,  9,  6,  5,  2,  4, -1,  5, -2, -4,  0,  9, -8,  8, -4,  6,  5,  3, -7, -9,  7,  0, -4, -1,  1,  7,  6, -6, 2, -8,  6,  5, 1, 2, 3, 4,
    -1, -2,  7,  5, -1,  4,  8,  7, -9, -1,  7,  1,  0, -2,  1,  0,  6, -6,  7,  4, -6,  0,  3, -2,  1,  8, -6, -2, -6, -3,  4, -2, 5,  6, -9, -5, 1, 2, 3, 4,
    -2, -5, -8, -6, -2, -7,  0,  6, -3, -1, -6,  4,  1, -4, -5, -3,  7,  9, -9,  9,  9,  0, -2,  6, -3,  5, -2, -1, -3, -5,  7,  6,  6, -8, 0, -4,1, 2, 3, 4,
     9,  2,  7, -8, -7,  8, -6, -6,  1,  7, -4, -4,  9, -6, -6,  5, -7, -6, -2, -1, -1, -2,  9,  6,  5,  2,  4, -1,  5, -2, -4,  0,  9, -8,  8, -4,1, 2, 3, 4,
    -6, -2, -1, -1, -2,  9,  6,  5,  2,  4, -1,  5, -2, -4,  0,  9, -8,  8, -4,  6,  5,  3, -7, -9,  7,  0, -4, -1,  1,  7,  6, -6, 2, -8,  6,  5, 1, 2, 3, 4,
    -1, -2,  7,  5, -1,  4,  8,  7, -9, -1,  7,  1,  0, -2,  1,  0,  6, -6,  7,  4, -6,  0,  3, -2,  1,  8, -6, -2, -6, -3,  4, -2, 5,  6, -9, -5, 1, 2, 3, 4,
    -2, -5, -8, -6, -2, -7,  0,  6, -3, -1, -6,  4,  1, -4, -5, -3,  7,  9, -9,  9,  9,  0, -2,  6, -3,  5, -2, -1, -3, -5,  7,  6,  6, -8, 0, -4,1, 2, 3, 4,
     9,  2,  7, -8, -7,  8, -6, -6,  1,  7, -4, -4,  9, -6, -6,  5, -7, -6, -2, -1, -1, -2,  9,  6,  5,  2,  4, -1,  5, -2, -4,  0,  9, -8,  8, -4,1, 2, 3, 4,
    -6, -2, -1, -1, -2,  9,  6,  5,  2,  4, -1,  5, -2, -4,  0,  9, -8,  8, -4,  6,  5,  3, -7, -9,  7,  0, -4, -1,  1,  7,  6, -6, 2, -8,  6,  5, 1, 2, 3, 4,
    -1, -2,  7,  5, -1,  4,  8,  7, -9, -1,  7,  1,  0, -2,  1,  0,  6, -6,  7,  4, -6,  0,  3, -2,  1,  8, -6, -2, -6, -3,  4, -2, 5,  6, -9, -5, 1, 2, 3, 4,
    -2, -5, -8, -6, -2, -7,  0,  6, -3, -1, -6,  4,  1, -4, -5, -3,  7,  9, -9,  9,  9,  0, -2,  6, -3,  5, -2, -1, -3, -5,  7,  6,  6, -8, 0, -4,1, 2, 3, 4,
     9,  2,  7, -8, -7,  8, -6, -6,  1,  7, -4, -4,  9, -6, -6,  5, -7, -6, -2, -1, -1, -2,  9,  6,  5,  2,  4, -1,  5, -2, -4,  0,  9, -8,  8, -4, 1, 2, 3, 4
};

const int16_t weights_2B[(outVecSz + inVecSz) * outVecSz] =
{
    -6, -2, -1, -1, -2,  9,  6,  5,  2,  4, -1,  5, -2, -4,  0,  9, -8,  8, -4,  6,  5,  3, -7, -9,  7,  0, -4, -1,  1,  7,  6, -6, 2, -8,  6,  5, 1, 2, 3, 4,
    -1, -2,  7,  5, -1,  4,  8,  7, -9, -1,  7,  1,  0, -2,  1,  0,  6, -6,  7,  4, -6,  0,  3, -2,  1,  8, -6, -2, -6, -3,  4, -2, 5,  6, -9, -5, 1, 2, 3, 4,
    -2, -5, -8, -6, -2, -7,  0,  6, -3, -1, -6,  4,  1, -4, -5, -3,  7,  9, -9,  9,  9,  0, -2,  6, -3,  5, -2, -1, -3, -5,  7,  6,  6, -8, 0, -4,1, 2, 3, 4,
     9,  2,  7, -8, -7,  8, -6, -6,  1,  7, -4, -4,  9, -6, -6,  5, -7, -6, -2, -1, -1, -2,  9,  6,  5,  2,  4, -1,  5, -2, -4,  0,  9, -8, 8, -4, 1, 2, 3, 4,
    -6, -2, -1, -1, -2,  9,  6,  5,  2,  4, -1,  5, -2, -4,  0,  9, -8,  8, -4,  6,  5,  3, -7, -9,  7,  0, -4, -1,  1,  7,  6, -6, 2, -8,  6,  5, 1, 2, 3, 4,
    -1, -2,  7,  5, -1,  4,  8,  7, -9, -1,  7,  1,  0, -2,  1,  0,  6, -6,  7,  4, -6,  0,  3, -2,  1,  8, -6, -2, -6, -3,  4, -2, 5,  6, -9, -5, 1, 2, 3, 4,
    -2, -5, -8, -6, -2, -7,  0,  6, -3, -1, -6,  4,  1, -4, -5, -3,  7,  9, -9,  9,  9,  0, -2,  6, -3,  5, -2, -1, -3, -5,  7,  6,  6, -8, 0, -4,1, 2, 3, 4,
     9,  2,  7, -8, -7,  8, -6, -6,  1,  7, -4, -4,  9, -6, -6,  5, -7, -6, -2, -1, -1, -2,  9,  6,  5,  2,  4, -1,  5, -2, -4,  0,  9, -8,  8, -4,1, 2, 3, 4,
    -6, -2, -1, -1, -2,  9,  6,  5,  2,  4, -1,  5, -2, -4,  0,  9, -8,  8, -4,  6,  5,  3, -7, -9,  7,  0, -4, -1,  1,  7,  6, -6, 2, -8,  6,  5, 1, 2, 3, 4,
    -1, -2,  7,  5, -1,  4,  8,  7, -9, -1,  7,  1,  0, -2,  1,  0,  6, -6,  7,  4, -6,  0,  3, -2,  1,  8, -6, -2, -6, -3,  4, -2, 5,  6, -9, -5, 1, 2, 3, 4,
    -2, -5, -8, -6, -2, -7,  0,  6, -3, -1, -6,  4,  1, -4, -5, -3,  7,  9, -9,  9,  9,  0, -2,  6, -3,  5, -2, -1, -3, -5,  7,  6,  6, -8, 0, -4,1, 2, 3, 4,
     9,  2,  7, -8, -7,  8, -6, -6,  1,  7, -4, -4,  9, -6, -6,  5, -7, -6, -2, -1, -1, -2,  9,  6,  5,  2,  4, -1,  5, -2, -4,  0,  9, -8,  8, -4,1, 2, 3, 4,
    -6, -2, -1, -1, -2,  9,  6,  5,  2,  4, -1,  5, -2, -4,  0,  9, -8,  8, -4,  6,  5,  3, -7, -9,  7,  0, -4, -1,  1,  7,  6, -6, 2, -8,  6,  5, 1, 2, 3, 4,
    -1, -2,  7,  5, -1,  4,  8,  7, -9, -1,  7,  1,  0, -2,  1,  0,  6, -6,  7,  4, -6,  0,  3, -2,  1,  8, -6, -2, -6, -3,  4, -2, 5,  6, -9, -5, 1, 2, 3, 4,
    -2, -5, -8, -6, -2, -7,  0,  6, -3, -1, -6,  4,  1, -4, -5, -3,  7,  9, -9,  9,  9,  0, -2,  6, -3,  5, -2, -1, -3, -5,  7,  6,  6, -8, 0, -4,1, 2, 3, 4,
     9,  2,  7, -8, -7,  8, -6, -6,  1,  7, -4, -4,  9, -6, -6,  5, -7, -6, -2, -1, -1, -2,  9,  6,  5,  2,  4, -1,  5, -2, -4,  0,  9, -8,  8, -4,1, 2, 3, 4,
    -6, -2, -1, -1, -2,  9,  6,  5,  2,  4, -1,  5, -2, -4,  0,  9, -8,  8, -4,  6,  5,  3, -7, -9,  7,  0, -4, -1,  1,  7,  6, -6, 2, -8,  6,  5, 1, 2, 3, 4,
    -1, -2,  7,  5, -1,  4,  8,  7, -9, -1,  7,  1,  0, -2,  1,  0,  6, -6,  7,  4, -6,  0,  3, -2,  1,  8, -6, -2, -6, -3,  4, -2, 5,  6, -9, -5, 1, 2, 3, 4,
    -2, -5, -8, -6, -2, -7,  0,  6, -3, -1, -6,  4,  1, -4, -5, -3,  7,  9, -9,  9,  9,  0, -2,  6, -3,  5, -2, -1, -3, -5,  7,  6,  6, -8, 0, -4,1, 2, 3, 4,
     9,  2,  7, -8, -7,  8, -6, -6,  1,  7, -4, -4,  9, -6, -6,  5, -7, -6, -2, -1, -1, -2,  9,  6,  5,  2,  4, -1,  5, -2, -4,  0,  9, -8, 8, -4,1, 2, 3, 4,
    -6, -2, -1, -1, -2,  9,  6,  5,  2,  4, -1,  5, -2, -4,  0,  9, -8,  8, -4,  6,  5,  3, -7, -9,  7,  0, -4, -1,  1,  7,  6, -6, 2, -8,  6,  5, 1, 2, 3, 4,
    -1, -2,  7,  5, -1,  4,  8,  7, -9, -1,  7,  1,  0, -2,  1,  0,  6, -6,  7,  4, -6,  0,  3, -2,  1,  8, -6, -2, -6, -3,  4, -2, 5,  6, -9, -5, 1, 2, 3, 4,
    -2, -5, -8, -6, -2, -7,  0,  6, -3, -1, -6,  4,  1, -4, -5, -3,  7,  9, -9,  9,  9,  0, -2,  6, -3,  5, -2, -1, -3, -5,  7,  6,  6, -8, 0, -4,1, 2, 3, 4,
     9,  2,  7, -8, -7,  8, -6, -6,  1,  7, -4, -4,  9, -6, -6,  5, -7, -6, -2, -1, -1, -2,  9,  6,  5,  2,  4, -1,  5, -2, -4,  0,  9, -8,  8, -4,1, 2, 3, 4,
    -6, -2, -1, -1, -2,  9,  6,  5,  2,  4, -1,  5, -2, -4,  0,  9, -8,  8, -4,  6,  5,  3, -7, -9,  7,  0, -4, -1,  1,  7,  6, -6, 2, -8,  6,  5, 1, 2, 3, 4,
    -1, -2,  7,  5, -1,  4,  8,  7, -9, -1,  7,  1,  0, -2,  1,  0,  6, -6,  7,  4, -6,  0,  3, -2,  1,  8, -6, -2, -6, -3,  4, -2, 5,  6, -9, -5, 1, 2, 3, 4,
    -2, -5, -8, -6, -2, -7,  0,  6, -3, -1, -6,  4,  1, -4, -5, -3,  7,  9, -9,  9,  9,  0, -2,  6, -3,  5, -2, -1, -3, -5,  7,  6,  6, -8, 0, -4,1, 2, 3, 4,
     9,  2,  7, -8, -7,  8, -6, -6,  1,  7, -4, -4,  9, -6, -6,  5, -7, -6, -2, -1, -1, -2,  9,  6,  5,  2,  4, -1,  5, -2, -4,  0,  9, -8,  8, -4,1, 2, 3, 4,
    -6, -2, -1, -1, -2,  9,  6,  5,  2,  4, -1,  5, -2, -4,  0,  9, -8,  8, -4,  6,  5,  3, -7, -9,  7,  0, -4, -1,  1,  7,  6, -6, 2, -8,  6,  5, 1, 2, 3, 4,
    -1, -2,  7,  5, -1,  4,  8,  7, -9, -1,  7,  1,  0, -2,  1,  0,  6, -6,  7,  4, -6,  0,  3, -2,  1,  8, -6, -2, -6, -3,  4, -2, 5,  6, -9, -5, 1, 2, 3, 4,
    -2, -5, -8, -6, -2, -7,  0,  6, -3, -1, -6,  4,  1, -4, -5, -3,  7,  9, -9,  9,  9,  0, -2,  6, -3,  5, -2, -1, -3, -5,  7,  6,  6, -8, 0, -4,1, 2, 3, 4,
     9,  2,  7, -8, -7,  8, -6, -6,  1,  7, -4, -4,  9, -6, -6,  5, -7, -6, -2, -1, -1, -2,  9,  6,  5,  2,  4, -1,  5, -2, -4,  0,  9, -8,  8, -4,1, 2, 3, 4,
};

const int16_t inputs[inVecSz * groupingNum] = {
    -5,  9, -7,  4, 5, -4, -7,  4
};

const intel_bias_t regularBiases[outVecSz*groupingNum] = {
    5, 4, -2, 5, 5, 4, -2, 5, 5, 4, -2, 5, 5, 4, -2, 5, 5, 4, -2, 5, 5, 4, -2, 5, 5, 4, -2, 5, 5, 4, -2, 5
};

const  intel_compound_bias_t compoundBiases[outVecSz*groupingNum] =
{
    { 5,1,{0} }, {4,1,{0}}, {-2,1,{0}}, {5,1,{0}}, { 5,1,{0} }, {4,1,{0}}, {-2,1,{0}}, {5,1,{0}},
    { 5,1,{0} }, {4,1,{0}}, {-2,1,{0}}, {5,1,{0}}, { 5,1,{0} }, {4,1,{0}}, {-2,1,{0}}, {5,1,{0}},
    { 5,1,{0} }, {4,1,{0}}, {-2,1,{0}}, {5,1,{0}}, { 5,1,{0} }, {4,1,{0}}, {-2,1,{0}}, {5,1,{0}},
    { 5,1,{0} }, {4,1,{0}}, {-2,1,{0}}, {5,1,{0}}, { 5,1,{0} }, {4,1,{0}}, {-2,1,{0}}, {5,1,{0}}
};

const int16_t ref_output[outVecSz * groupingNum] =
{
 125,   -10,   580,  -164,   125,   -10,   580,  -164,   125,   -10,   580,  -164,   125,   -10,   580,  -164,
 125,   -10,   580,  -164,   125,   -10,   580,  -164,   125,   -10,   580,  -164,   125,   -10,   580,  -164,
 395,   395,   361,   395,   395,   395,   361,   395,   395,   395,   361,   395,   395,   395,   361,   395,
 395,   395,   361,   395,   395,   395,   361,   395,   395,   395,   361,   395,   395,   395,   361,   395,
 395,   395,   361,   395,   395,   395,   361,   395,   395,   395,   361,   395,   395,   395,   361,   395,
 395,   395,   361,   395,   395,   395,   361,   395,   395,   395,   361,   395,   395,   395,   361,   395,
-200,  1816,  -200,  1816,  -200,  1816,  -200,  1816,  -200,  1816,  -200,  1816,  -200,  1816,  -200,  1816,
-200,  1816,  -200,  1816,  -200,  1816,  -200,  1816,  -200,  1816,  -200,  1816,  -200,  1816,  -200,  1816
};

}

SetupRecurrentModel::SetupRecurrentModel(DeviceController & deviceCtrl, bool wght2B)
    : deviceController{deviceCtrl},
    weightsAre2Bytes{wght2B}
{
    nnet.nGroup = groupingNum;
    nnet.nLayers = layersNum;
    nnet.pLayers = (intel_nnet_layer_t*)calloc(nnet.nLayers, sizeof(intel_nnet_layer_t));

    sampleRnnLayer(nnet);

    deviceController.ModelCreate(&nnet, &modelId);

    configId = deviceController.ConfigAdd(modelId);

    deviceController.BufferAdd(configId, GNA_IN, 0, inputBuffer);
    deviceController.BufferAdd(configId, GNA_OUT, 0, outputBuffer);
}

SetupRecurrentModel::~SetupRecurrentModel()
{
    deviceController.ModelRelease(modelId);
    deviceController.Free();

    free(nnet.pLayers);
}

void SetupRecurrentModel::checkReferenceOutput() const
{
    for (int i = 0; i < sizeof(ref_output) / sizeof(int16_t); ++i)
    {
        int16_t outElemVal = static_cast<const int16_t*>(outputBuffer)[i];
        if (ref_output[i] != outElemVal)
        {
            // TODO: how it should notified? return or throw
            throw std::exception("Wrong output");
        }
    }
}

void SetupRecurrentModel::samplePwl(intel_pwl_segment_t *segments, uint32_t nSegments)
{
    auto xBase = -200;
    auto xBaseInc = 2*abs(xBase) / nSegments;
    auto yBase = -200;
    auto yBaseInc = 1;
    for (auto i = 0ui32; i < nSegments; i++, xBase += xBaseInc, yBase += yBaseInc, yBaseInc++) 
    {
        segments[i].xBase = xBase;
        segments[i].yBase = yBase;
        segments[i].slope = 1;
    }
}

void SetupRecurrentModel::sampleRnnLayer(intel_nnet_type_t& nnet)
{
    int buf_size_weights = weightsAre2Bytes ? ALIGN64(sizeof(weights_2B)) : ALIGN64(sizeof(weights_1B));
    int buf_size_inputs = ALIGN64(sizeof(inputs));
    int buf_size_biases = weightsAre2Bytes ? ALIGN64(sizeof(regularBiases)) : ALIGN64(sizeof(compoundBiases));
    int buf_size_scratchpad = ALIGN64(outVecSz * groupingNum * sizeof(int32_t));
    int buf_size_outputs = ALIGN64(outVecSz * groupingNum * sizeof(int16_t));
    int buf_size_tmp_outputs = ALIGN64(outVecSz * groupingNum * sizeof(int32_t));
    int buf_size_pwl = ALIGN64(nSegments * sizeof(intel_pwl_segment_t));

    uint32_t bytes_requested = buf_size_weights + buf_size_inputs + buf_size_biases + buf_size_scratchpad + buf_size_outputs + buf_size_tmp_outputs + buf_size_pwl;
    uint32_t bytes_granted;

    uint8_t* pinned_mem_ptr = deviceController.Alloc(bytes_requested, &bytes_granted);

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

    scratchpad = pinned_mem_ptr;
    memset(scratchpad, 0, buf_size_scratchpad);
    pinned_mem_ptr += buf_size_scratchpad;

    outputBuffer = pinned_mem_ptr;
    pinned_mem_ptr += buf_size_outputs;

    affine_func.nBytesPerWeight = weightsAre2Bytes ? 2 : 1;
    affine_func.nBytesPerBias = weightsAre2Bytes ? sizeof(intel_bias_t) : sizeof(intel_compound_bias_t);
    affine_func.pWeights = pinned_weights;
    affine_func.pBiases = pinned_biases;

    pwl.nSegments = nSegments;
    pwl.pSegments = reinterpret_cast<intel_pwl_segment_t*>(pinned_mem_ptr);
    samplePwl(pwl.pSegments, pwl.nSegments);
    pinned_mem_ptr += buf_size_pwl;

    recurrent_layer.affine = affine_func;
    recurrent_layer.pwl = pwl;
    recurrent_layer.feedbackFrameDelay = 3;

    nnet.pLayers[0].nInputColumns = inVecSz;
    nnet.pLayers[0].nInputRows = nnet.nGroup;
    nnet.pLayers[0].nOutputColumns = outVecSz;
    nnet.pLayers[0].nOutputRows = nnet.nGroup;
    nnet.pLayers[0].nBytesPerInput = sizeof(int16_t);
    nnet.pLayers[0].nBytesPerOutput = sizeof(int16_t);
    nnet.pLayers[0].nBytesPerIntermediateOutput = sizeof(int32_t);
    nnet.pLayers[0].nLayerKind = INTEL_RECURRENT;
    nnet.pLayers[0].type = INTEL_INPUT_OUTPUT;
    nnet.pLayers[0].pLayerStruct = &recurrent_layer;
    nnet.pLayers[0].pInputs = nullptr;
    nnet.pLayers[0].pOutputsIntermediate = scratchpad;
    nnet.pLayers[0].pOutputs = nullptr;
}
