@copyright (C) 2018-2020 Intel Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
see the License for the specific language governing permissions
and limitations under the License.

SPDX-License-Identifier: Apache-2.0

Build instruction for samples.

CMake (ver. at least 3.10 is required).
For additional information about how project is built consult CMakeLists.txt file.

1. Generate projects with CMake:
	Example command:
----------------------------------------------------------------------------
		Microsoft* Windows* specific:
		cmake -G "Visual Studio 15 2017 Win64" .
----------------------------------------------------------------------------
		LINUX specific:
		cmake .
----------------------------------------------------------------------------
2. Build project:
   cmake --build .
3. Executable should be under src/sample01/Debug directory.

*Other names and brands may be claimed as the property of others.
