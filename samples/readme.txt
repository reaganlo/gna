Copyright 2018 Intel Corporation.

This software and the related documents are Intel copyrighted materials,
and your use of them is governed by the express license under which they
were provided to you (Intel OBL Software License Agreement (OEM/IHV/ISV
Distribution & Single User) (v. 11.2.2017) ). Unless the License provides
otherwise, you may not use, modify, copy, publish, distribute, disclose or
transmit this software or the related documents without Intel's prior
written permission.
This software and the related documents are provided as is, with no
express or implied warranties, other than those that are expressly
stated in the License.

----------------------------------------------------------------------------
                                Microsoft* Windows*
----------------------------------------------------------------------------
In order to build the samples:
    1. Put gna.lib and gna.dll into gna-lib\x64 directory
    2. Put gna-api.h, gna-api-status.h, gna-api-types-gmm.h, gna-api-types-xnn.h, gna-api-dumper.h, gna-api-instrumentation.h into gna-lib\include directory
    3. Install Redistributable Libraries for Intel(R) C++ Compiler 17.0 if necessary
       https://software.intel.com/en-us/articles/redistributables-for-intel-parallel-studio-xe-2017-composer-edition-for-windows
    4. Install Redistributable Libraries for Microsoft Visual Studio 2015 if necessary
       https://www.microsoft.com/en-us/download/details.aspx?id=48145
    5. Open gna-samples.sln solution in Microsoft Visual Studio 2015.
    6. Build project in Microsoft Visual Studio 2015.
    7. Executable should be under bin\<sample name>\<config>\x64\ directory.

    Alternatively, generate projects with CMake.
    1. Generate projects
       Example command: cmake -G "Visual Studio 14 2015 Win64" -T "Intel C++ Compiler 17.0" .
       CMake will generate projects for specified architecture (x64 in this case). CMake will generate default configurations.
    2. Build projects
       Example command: cmake --build . --config Debug
       For additional information about how projects are generated and built consult CMakeLists.txt file.

----------------------------------------------------------------------------
                                    LINUX
----------------------------------------------------------------------------
This Software is built on Ubuntu 16.04 LTS (gcc 5.4.0)

    In order to build the samples:
    1. Put libgnad.so.2 (GNA_LIB_DEBUG cmake option) or libgna.so.2 to gna-lib/x64 directory.
    2. Put gna-api.h, gna-api-status.h, gna-api-types-gmm.h, gna-api-types-xnn.h into gna-lib/include directory
    3. Install CMake 3.9 or newer.
    4. Install Redistributable Libraries for Intel(R) C++ Compiler 17.0 if necessary
       https://software.intel.com/en-us/articles/redistributables-for-intel-parallel-studio-xe-2017-composer-edition-for-linux
    5. Generate projects with CMake. Example command:
       cmake -DCMAKE_BUILD_TYPE=Debug -DGNA_LOCAL_PACKAGE:BOOL=ON .
       Note: Setting GNA_LOCAL_PACKAGE variable is currently required.
    6. Build project by either running generator (e.g. GNU Make) directly or with CMake command:
       cmake --build .
    7. Executable should be under bin/<sample name>/<configuration>/x64/ directory.

For additional information about how project is built consult CMakeLists.txt file.

*Other names and brands may be claimed as the property of others.