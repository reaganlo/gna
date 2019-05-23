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

#include "DataMode.h"

#include <cstdint>

namespace GNA {
class PoolingFunction2D;
struct ConvolutionFunction2D;

struct GNA3_UMemAlloc {
    GNA3_UMemAlloc()
        :KMemAlloc{0}, CMemAlloc{0}, PMemAlloc{0}, UMemAlloc{0}
    {}
    uint32_t KMemAlloc; // KMEM Allocation in Bytes (Gross)
    uint32_t CMemAlloc; // CMEM Allocation in Bytes (Gross)
    uint32_t PMemAlloc; // PMEM Allocation in Bytes (Gross)
    uint32_t UMemAlloc; // UMEM Allocation in Bytes (Gross) ; Total of K+C+P
};

struct convolutional_fused_configuration {
    convolutional_fused_configuration()
        : Valid{ false }, KWG{ 0 }, KWGIter{ 0 }, uT{ 0 },
        KMemBase{ 0 }, CMemBase{ 0 }, PMemBase{ 0 }, AListMem{ false }
    {}
    bool             Valid;     // Indiacates Valid AdaptHW Configuration
    uint16_t         KWG;       // GNA-3.0 HAS : Kernel-Working-Group (Number of Kernels in IFV Iteration)
    uint8_t          KWGIter;   // GNA-3.0 HAS : Kernel-Working-Group Iterations
    uint8_t          uT;        // GNA-3.0 HAS : Micro-Threads (4-bits)
    uint8_t          KMemBase;  // GNA-3.0 HAS : GNA Descriptor
    uint8_t          CMemBase;  // GNA-3.0 HAS : GNA Descriptor
    uint8_t          PMemBase;  // GNA-3.0 HAS : GNA Descriptor
    bool             AListMem;  // TODO
    GNA3_UMemAlloc   UMemAlloc;         // MetaData
};

bool GNA3_PopulateLD(ConvolutionFunction2D const * cnnIn, PoolingFunction2D const * poolingIn, const DataMode& outputMode, convolutional_fused_configuration* const convConfiguration);

}
