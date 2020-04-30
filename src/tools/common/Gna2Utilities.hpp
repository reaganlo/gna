/*
 INTEL CONFIDENTIAL
 Copyright 2020 Intel Corporation.

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

#include <algorithm>
#include <string>

class Gna2Version
{
public:
    Gna2Version(const std::string& version)
    {
        std::stringstream input(version);
        std::string element;
        while (std::getline(input, element, '.'))
        {
            parts.push_back(element);
        }

        trimR(parts, "");
    }

    std::string GetUnPadded(int index) const
    {
        if(index >= parts.size())
        {
            return "";
        }
        return trimL(parts.at(index), '0');
    }

    int Size() const
    {
        return parts.size();
    }

    std::string GetMajor() const
    {
        return GetUnPadded(0);
    }

    std::string GetReleaseFamily() const
    {
        return GetUnPadded(1);
    }

    std::string GetRelease() const
    {
        return GetUnPadded(2);
    }

    std::string GetBuild() const
    {
        return GetUnPadded(3);
    }
private:
    std::vector<std::string> parts;
    template <class V>
    static V trimL(V all, typename V::value_type element)
    {
        all.erase(all.begin(), std::find_if(all.begin(), all.end(), [&](typename V::value_type val)
        {
            return element != val;
        }));
        return all;
    }

    template <class V>
    void trimR(V& all, typename V::value_type element)
    {
        all.erase(std::find_if(all.rbegin(), all.rend(), [&](typename V::value_type val)
        {
            return element != val;
        }).base(), all.end());
    }
};
