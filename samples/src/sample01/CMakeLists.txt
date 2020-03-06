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

add_executable(sample01
    sample01.cpp
)
target_link_libraries(sample01
    PRIVATE
    gna
)

target_include_directories(sample01
    PUBLIC
    .
    ${GNA_LIB_PATH}/include/
)

set_target_properties(sample01
  PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${BINARY_DIR}/sample01
  ARCHIVE_OUTPUT_DIRECTORY ${BINARY_DIR}/sample01
  RUNTIME_OUTPUT_DIRECTORY ${BINARY_DIR}/sample01
)

add_custom_command(TARGET
    sample01 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:gna>
    $<TARGET_FILE_DIR:sample01>
)