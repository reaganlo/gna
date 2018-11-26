# INTEL CONFIDENTIAL
# Copyright 2018 Intel Corporation.

# The source code contained or described herein and all documents related
# to the source code ("Material") are owned by Intel Corporation or its suppliers
# or licensors. Title to the Material remains with Intel Corporation or its suppliers
# and licensors. The Material may contain trade secrets and proprietary
# and confidential information of Intel Corporation and its suppliers and licensors,
# and is protected by worldwide copyright and trade secret laws and treaty provisions.
# No part of the Material may be used, copied, reproduced, modified, published,
# uploaded, posted, transmitted, distributed, or disclosed in any way without Intel's
# prior express written permission.

# No license under any patent, copyright, trade secret or other intellectual
# property right is granted to or conferred upon you by disclosure or delivery
# of the Materials, either expressly, by implication, inducement, estoppel
# or otherwise. Any license under such intellectual property rights must
# be express and approved by Intel in writing.

# Unless otherwise agreed by Intel in writing, you may not remove or alter this notice
# or any other notice embedded in Materials by Intel or Intel's suppliers or licensors
# in any way.

get_property(IsMultiConfig GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT IsMultiConfig)
  if(NOT CMAKE_BUILD_TYPE
      OR (NOT ${CMAKE_BUILD_TYPE} STREQUAL "${OS_PREFIX}_DEBUG"
        AND NOT ${CMAKE_BUILD_TYPE} STREQUAL "${OS_PREFIX}_RELEASE"))
      set(CMAKE_BUILD_TYPE "${OS_PREFIX}_RELEASE" CACHE STRING "default build type" FORCE)
  endif()
else()
  # configuration types for multi-config generators
  set(CMAKE_CONFIGURATION_TYPES "${OS_PREFIX}_DEBUG;${OS_PREFIX}_RELEASE;KLOCWORK")
endif()

# postfix for debug libraries
if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux" OR ${CMAKE_SYSTEM_NAME} STREQUAL "Android")
  set(CMAKE_${OS_PREFIX}_DEBUG_POSTFIX d)
endif()
