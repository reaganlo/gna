@echo off

:: Copyright 2020 Intel Corporation.

:: This software and the related documents are Intel copyrighted materials,
:: and your use of them is governed by the express license under which they
:: were provided to you (Intel OBL Software License Agreement (OEM/IHV/ISV
:: Distribution & Single User) (v. 11.2.2017) ). Unless the License provides
:: otherwise, you may not use, modify, copy, publish, distribute, disclose or
:: transmit this software or the related documents without Intel's prior
:: written permission.
:: This software and the related documents are provided as is, with no
:: express or implied warranties, other than those that are expressly
:: stated in the License.


:: LIB Windows gna-lib.sln MSVC ENG
echo Setting CMAKE params

echo GNA_CMAKE_TOOLSET_NAME='%GNA_CMAKE_TOOLSET_NAME%'
IF '%GNA_CMAKE_TOOLSET_NAME%'=='' (
    echo Error: GNA_CMAKE_TOOLSET_NAME is missing
    exit /b 666
)

echo GNA_CMAKE_REPO_PATH='%GNA_CMAKE_REPO_PATH%'
IF '%GNA_CMAKE_REPO_PATH%'=='' (
    echo Error: GNA_CMAKE_REPO_PATH is missing
    exit /b 666
)

echo GNA_CMAKE_TOOLSET='%GNA_CMAKE_TOOLSET%'
IF '%GNA_CMAKE_TOOLSET%'=='' (
    echo Error: GNA_CMAKE_TOOLSET is missing
    exit /b 666
)

echo GNA_CMAKE_TOOLSET_CXX='%GNA_CMAKE_TOOLSET_CXX%'
IF '%GNA_CMAKE_TOOLSET_CXX%'=='' (
    echo Error: GNA_CMAKE_TOOLSET_CXX is missing
    exit /b 666
)

echo GNA_CMAKE_CONFIG='%GNA_CMAKE_CONFIG%'
IF '%GNA_CMAKE_CONFIG%'=='' (
    echo Error: GNA_CMAKE_CONFIG is missing
    exit /b 666
)

@echo on
mkdir %GNA_CMAKE_TOOLSET_NAME%
cd %GNA_CMAKE_TOOLSET_NAME% || exit /b 666
echo Running CMAKE
cmake -G %GNA_CMAKE_TOOLSET% -T %GNA_CMAKE_TOOLSET_CXX% %GNA_CMAKE_REPO_PATH% || exit /b 666

echo Building 
cmake --build . --config %GNA_CMAKE_CONFIG% || exit /b 666

del /Q /F CMakeCache.txt
cd ..
