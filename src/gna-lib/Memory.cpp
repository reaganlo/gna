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

#include "Memory.h"

#include "Validator.h"

using namespace GNA;

// just makes object from arguments
Memory::Memory(void * buffer, const size_t size) :
    Buffer(buffer),
    Size(size)
{};

// allocates and zeros memory
Memory::Memory(const size_t size)
{
    Expect::True(size > 0, GNA_INVALIDMEMSIZE);
    Buffer = _gna_malloc(size);
    Expect::ValidBuffer(Buffer);
    Size = size;
    memset(Buffer, 0, Size);
};

Memory::~Memory()
{
    Free();
}

void Memory::Free()
{
    if (Buffer)
    {
        _gna_free(Buffer);
        Buffer = nullptr;
        Size = 0;
    }
}
