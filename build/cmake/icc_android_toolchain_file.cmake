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

# Android toolchain file for Intel's icc/icpc usage
# When invoking cmake configure, make sure to pass ICC_DIR and CMAKE_ANDROID_STANDALONE_TOOLCHAIN properly set:
#
# cmake -DCMAKE_TOOLCHAIN_FILE=/path/to/this/file \
#       -DICC_DIR=/path/to/intels/icc/icpc/directory \
#       -DCMAKE_ANDROID_STANDALONE_TOOLCHAIN=/path/to/android/ndk/based/standalone/toolchain
#
# Prior to cmake (conf&build) ANDROID_API env variable should be set and exported
# e.g.: export ANDROID_API=21
# The 21st API level is chosen due to fact that NDK v18 is shipped with libc.a of this level
# To control search and install paths wrt host CMAKE_STAGING_PREFIX can be used if necessary

message("Intel's ICC/ICPC was chosen for Android target build")
set(CMAKE_SYSTEM_NAME "Android")
# setting CMAKE_SYSTEM_VERSION=1 supresses cmake from determining compiler
set(CMAKE_SYSTEM_VERSION 1)
set(CMAKE_SYSTEM_PROCESSOR x86_64)

set(CMAKE_SYSROOT ${CMAKE_ANDROID_STANDALONE_TOOLCHAIN}/sysroot)
include_directories("${CMAKE_ANDROID_STANDALONE_TOOLCHAIN}/include/c++/4.9.x")
set(CMAKE_CXX_FLAGS "-L${CMAKE_ANDROID_STANDALONE_TOOLCHAIN}/x86_64-linux-android/lib64 -lc++_shared")

set(CMAKE_C_COMPILER ${ICC_DIR}/icc)
set(CMAKE_CXX_COMPILER ${ICC_DIR}/icpc)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
