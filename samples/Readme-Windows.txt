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
1. Put gna.lib and gna.dll into gna-lib\[x86|x64] directory
2. Put gna-api.h, gna-api-status.h, gna-api-types-gmm.h, gna-api-types-xnn.h, gna-api-dumper.h, gna-api-instrumentation.h into gna-lib\include directory
3. Install Redistributable Libraries for Intel(R) C++ Compiler 17.0 if necessary
   https://software.intel.com/en-us/articles/redistributables-for-intel-parallel-studio-xe-2017-composer-edition-for-windows
4. Install Redistributable Libraries for Microsoft Visual Studio 2015 if necessary 
   https://www.microsoft.com/en-us/download/details.aspx?id=48145
5. Open gna-samples.sln solution in Microsoft Visual Studio 2015.
6. Build project in Microsoft Visual Studio 2015.
7. Executable should be under bin\<sample name>\<config>\<architecture>\ directory.

// TODO: publish cmake script files
(optional) Generate projects with CMake.
1. Generate projects
   Example command: cmake -G "Visual Studio 14 2015 Win64" -T "Intel C++ Compiler 17.0" .
   CMake will generate projects for specified architecture (x64 in this case). CMake will generate WIN-DEBUG and WIN-RELEASE configurations.
2. Build projects
   Example command: cmake --build . --config WIN-RELEASE
   For additional information about how projects are generated and built consult CMakeLists.txt file.




*Other names and brands may be claimed as the property of others. 