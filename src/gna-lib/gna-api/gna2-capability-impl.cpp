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

#include "gna2-capability-impl.h"

#include "ApiWrapper.h"
#include "Expect.h"

using namespace GNA;

GNA2_API enum Gna2Status Gna2GetLibraryVersion(char * versionBuffer, uint32_t versionBufferSize)
{
    static const char versionString[] = GNA_LIBRARY_VERSION_STRING;

    const std::function<Gna2Status()> command = [&]()
    {
        GNA::Expect::NotNull(versionBuffer);
        GNA::Expect::True(static_cast<uint64_t>(32) <= versionBufferSize, Gna2StatusMemorySizeInvalid);
        GNA::Expect::True(sizeof(versionString) <= static_cast<uint64_t>(versionBufferSize), Gna2StatusMemorySizeInvalid);
        const auto reqSize = snprintf(versionBuffer, versionBufferSize, "%s", versionString);
        GNA::Expect::True(reqSize >= 0 && static_cast<unsigned>(reqSize) + 1 <= versionBufferSize,
            Gna2StatusMemorySizeInvalid);
        return Gna2StatusSuccess;
    };
    return GNA::ApiWrapper::ExecuteSafely(command);
}
