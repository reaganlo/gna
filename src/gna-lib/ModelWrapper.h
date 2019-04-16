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

#ifndef __GNA2_MODEL_WRAPPER_H
#define __GNA2_MODEL_WRAPPER_H

#include "gna2-model-impl.h"

#include "Expect.h"
#include "GnaException.h"

#include <cstdint>
#include <cstring> 
#include <map>
#include <vector>

namespace GNA
{

class ModelWrapper
{
public:
    static void OperationInit(ApiOperation * const operation,
        const OperationType type, const Gna2UserAllocator userAllocator);

protected:
    static uint32_t GetNumberOfOperands(OperationType operationType);
    static uint32_t GetNumberOfParameters(OperationType operationType);

private:
    template<typename Type>
    static Type ** AllocateAndFillZeros(const Gna2UserAllocator userAllocator, uint32_t elementCount)
    {
        Expect::NotNull((void *)(userAllocator));
        const auto size =  static_cast<uint32_t>(sizeof(Type *)) * elementCount;
        const auto memory = userAllocator(size);
        Expect::NotNull(memory, CAST1_STATUS Gna2StatusResourceAllocationError);
        memset(memory, 0, size);
        return static_cast<Type **>(memory);
    }
};

}

#endif //ifndef __GNA2_MODEL_WRAPPER_H
