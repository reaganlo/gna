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
Installation instructions and bring up guideline to verify if the
GNA device is available and functional on Intel platform with Intel GNA driver installed
on Windows* 10 64bit Operating System could be found in official-doc\Intel GNA Driver Bring-up Guide.pdf

    In order to use the GNA:
    1. Put gna.lib and gna.dll into your-GNA-lib-directory\[x86|x64] directory
    2. Put gna-api.h, gna-api-status.h, gna-api-types-gmm.h, gna-api-types-xnn.h, gna-api-dumper.h, gna-api-instrumentation.h into your-GNA-lib-directory\include directory
    3. Install Redistributable Libraries for Intel(R) C++ Compiler 17.0 if necessary
       https://software.intel.com/en-us/articles/redistributables-for-intel-parallel-studio-xe-2017-composer-edition-for-windows
    4. Install Redistributable Libraries for Microsoft Visual Studio 2015 if necessary
       https://www.microsoft.com/en-us/download/details.aspx?id=48145

----------------------------------------------------------------------------
                                    LINUX
----------------------------------------------------------------------------
This Software is built on Ubuntu 16.04 LTS (gcc 5.4.0)

    In order to use the GNA:
    1a. Use installation script to install gna library on your Linux system, or
    1b. Put libgna.so.2 to your-GNA-lib-directory\x64 directory, put gna-api.h, gna-api-status.h, gna-api-types-gmm.h, gna-api-types-xnn.h into your-GNA-lib-directory\include directory
    2. Install Redistributable Libraries for Intel(R) C++ Compiler 17.0 if necessary
       https://software.intel.com/en-us/articles/redistributables-for-intel-parallel-studio-xe-2017-composer-edition-for-linux


*Other names and brands may be claimed as the property of others.