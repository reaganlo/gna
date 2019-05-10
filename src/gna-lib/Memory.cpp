/*
INTEL CONFIDENTIAL
Copyright 2018 Intel Corporation.

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

#include <cstring>

#include "common.h"

#include "AccelerationDetector.h"
#include "CompiledModel.h"
#include "Expect.h"

using namespace GNA;

// just makes object from arguments
Memory::Memory(void *bufferIn, uint32_t userSize, uint32_t alignment) :
    Address{bufferIn},
    size{ALIGN(userSize, alignment)}
{
    deallocate = false;
};

// allocates and zeros memory
Memory::Memory(const size_t userSize, uint32_t alignment) :
    size{ALIGN(userSize, alignment)}
{
    Expect::GtZero(size, Gna2StatusMemorySizeInvalid);
    buffer = _gna_malloc(size);
    Expect::ValidBuffer(buffer);
    memset(buffer, 0, size); // this is costly and probably not needed
};

Memory::~Memory()
{
    if (buffer && deallocate)
    {
        if (mapped)
            unmap();

        _gna_free(buffer);
        buffer = nullptr;
        size = 0;
    }
}

void Memory::Map(DriverInterface& ddi)
{
    driverInterface = &ddi;
    if (mapped)
    {
        throw GnaException(Gna2StatusUnknownError);
    }

    id = driverInterface->MemoryMap(buffer, size);

    mapped = true;
}

void Memory::unmap()
{
    driverInterface->MemoryUnmap(id);
    mapped = false;
}

uint64_t Memory::GetId() const
{
    if (!mapped)
    {
        throw GnaException(Gna2StatusUnknownError);
    }

    return id;
}

