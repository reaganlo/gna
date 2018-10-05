Copyright 2018, Intel Corporation.

This GNA Scoring Accelerator Module ("Software") is
furnished under license and may only be used or copied in accordance with the
terms of that license. No license, express or implied, by estoppel or otherwise,
to any intellectual property rights is granted by this document. The Software
is subject to change without notice, and should not be construed as a commitment
by Intel Corporation to market, license, sell or support any product or technology.
Unless otherwise provided for in the license under which this Software is provided,
the Software is provided AS IS, with no warranties of any kind, express or implied.
Except as expressly permitted by the Software license, neither Intel Corporation
nor its suppliers assumes any responsibility or liability for any errors or
inaccuracies that may appear herein. Except as expressly permitted by the Software
license, no part of the Software may be reproduced, stored in a retrieval system,
transmitted in any form, or distributed by any means without the express written
consent of Intel Corporation.

In order to build the samples:
1. Put libgnad.so.2 (GNA_LIB_DEBUG cmake option) or libgna.so.2 to gna-lib/x64 directory.
2. Put gna-api.h, gna-api-status.h, gna-api-types-gmm.h, gna-api-types-xnn.h into gna-lib/include directory
3. Install Redistributable Libraries for Intel(R) C++ Compiler 17.0 if necessary
   https://software.intel.com/en-us/articles/redistributables-for-intel-parallel-studio-xe-2017-composer-edition-for-linux
4. Generate projects with CMake. Example command:
   cmake -DCMAKE_BUILD_TYPE=Debug -DGNA_LOCAL_PACKAGE:BOOL=ON .
   Note: Setting GNA_LOCAL_PACKAGE variable is currently required.
5. Build project by either running generator (e.g. GNU Make) directly or with CMake command:
   cmake --build .
6. Executable should be under bin/<sample name>/<configuration>/x64/ directory.

For additional information about how project is built consult CMakeLists.txt file.
