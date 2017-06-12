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

#include "ActiveList.h"
#include "Validator.h"

using namespace GNA;

std::unique_ptr<ActiveList> ActiveList::Create(const ActiveList& activeList, nn_layer_kind layerKind)
{
    Expect::ValidBuffer(activeList.Indices);
    if (INTEL_AFFINE == layerKind)
    {
        Expect::InRange(activeList.IndicesCount, 1, XNN_N_IN_ELEMS_MAX, GNA_INVALIDINDICES);
    }
    else // INTEL_GMM
    {
        Expect::InRange(activeList.IndicesCount, 1, GMM_STATES_COUNT_MAX, GNA_INVALIDINDICES);
    }
    
    return std::make_unique<ActiveList>(activeList);
}

ActiveList::ActiveList(const uint32_t indicesCountIn, const uint32_t* indicesIn) :
    IndicesCount{indicesCountIn},
    Indices{indicesIn} { }
