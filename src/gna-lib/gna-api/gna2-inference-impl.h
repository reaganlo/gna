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

#ifndef __GNA2_INFERENCE_IMPL_H
#define __GNA2_INFERENCE_IMPL_H

#include "gna2-common-impl.h"
#include "gna-api.h"
#include "../gna-api/gna2-inference-api.h"

#include <string>
#include <vector>
#include <cstdint>


namespace GNA
{
    /**
 * List of all supported acceleration modes
 */
class AccelerationMode
{
public:
    AccelerationMode(Gna2AccelerationMode basicMode, bool hardwareConsistencyEnabled = false);

    AccelerationMode(gna_acceleration legacyMode);

    bool IsHardwareEnforced() const;

    bool IsSoftwareEnforced() const;

    // operator needed by std::map
    bool operator<(const AccelerationMode& right) const;

    AccelerationMode GetEffectiveSoftwareAccelerationMode(const std::vector<Gna2AccelerationMode>& supportedCpuAccelerations) const;

    void SetMode(Gna2AccelerationMode newMode);

    Gna2AccelerationMode GetMode() const;

    void SetHwConsistency(bool consistencyEnabled);
    bool GetHwConsistency() const;

    const char* GetName() const;

private:
    Gna2AccelerationMode mode;

    static const char* UNKNOWN_ACCELERATION_MODE_NAME;

    bool hardwareConsistency = false;

    void enforceValidity();
};

}

#endif // __GNA2_INFERENCE_IMPL_H
