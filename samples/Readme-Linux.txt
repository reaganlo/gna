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

This Software is built with Microsoft* Visual Studio*  2015 Update 3
compilers. This implies requirement of having Visual C++ runtime libraries 
available in Windows* in order to run this software.  

In order to run the sample:
1a. Use installation script to install gna library on your Linux system, or
1b. Put libgna.so.2 to GNA\x64 directory, put gna-api.h, gna-api-status.h, gna-api-types-gmm.h, gna-api-types-xnn.h into GNA\include directory
2. Install Redistributable Libraries for Intel(R) C++ Compiler 17.0 if necessary
   https://software.intel.com/en-us/articles/redistributables-for-intel-parallel-studio-xe-2017-composer-edition-for-linux
3. Generate projects with CMake, example command:
   cmake -DCMAKE_ARCHITECTURE=x64 -DCMAKE_BUILD_TYPE=LNX-RELEASE .
   Note: CMAKE_ARCHITECTURE and CMAKE_BUILD_TYPE variables are optional. Defaults are x64 and LNX-RELEASE.
   Note(2): Set GNA_LOCAL_PACKAGE variable to ON if you followed 1b step, and did not use installation script.
4. Explore the code using text editor or IDE of your choice.
5. Build project either using generator of your choice (e.g. make) or with CMake command, for example:
   cmake --build .
6. Executable should be under bin/<sample name>/<config>/<architecture>/ directory.

For additional information about how project is built consult CMakeLists.txt file.




*Other names and brands may be claimed as the property of others. 