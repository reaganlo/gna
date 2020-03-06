#
# @copyright (C) 2018-2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# see the License for the specific language governing permissions
# and limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME gna-samples)
set(CXX_STANDARD_REQUIRED ON)

project(${PROJECT_NAME})

add_library(gna
    SHARED IMPORTED GLOBAL
)

set(BINARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/bin)
file(MAKE_DIRECTORY ${BINARY_DIR})

set(GNA_LIB_PATH ${CMAKE_SOURCE_DIR}/../gna-lib)
set(GNA_LIB_PATH_LNX_RELEASE ${GNA_LIB_PATH}/LNX-RELEASE/x64)
set(GNA_LIB_PATH_LNX_DEBUG ${GNA_LIB_PATH}/LNX-DEBUG/x64)
set(GNA_LIB_PATH_WIN_RELEASE ${GNA_LIB_PATH}/WIN-RELEASE/x64)
set(GNA_LIB_PATH_WIN_DEBUG ${GNA_LIB_PATH}/WIN-DEBUG/x64)

if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux" OR ${CMAKE_SYSTEM_NAME} STREQUAL "Android")
    set_property(TARGET gna PROPERTY IMPORTED_LOCATION ${GNA_LIB_PATH_LNX_RELEASE}/libgna.so)
    set_property(TARGET gna PROPERTY IMPORTED_LOCATION_DEBUG ${GNA_LIB_PATH_LNX_DEBUG}/libgnad.so)
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    set_property(TARGET gna PROPERTY IMPORTED_LOCATION ${GNA_LIB_PATH_WIN_RELEASE}/gna.dll)
    set_property(TARGET gna PROPERTY IMPORTED_LOCATION_DEBUG ${GNA_LIB_PATH_WIN_DEBUG}/gna.dll)
    set_property(TARGET gna PROPERTY IMPORTED_IMPLIB ${GNA_LIB_PATH_WIN_RELEASE}/gna.lib)
    set_property(TARGET gna PROPERTY IMPORTED_IMPLIB_DEBUG ${GNA_LIB_PATH_WIN_DEBUG}/gna.lib)
endif()

add_subdirectory(src/sample01)
