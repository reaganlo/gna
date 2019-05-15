/*
 INTEL CONFIDENTIAL
 Copyright 2019 Intel Corporation.

 The source code contained or described herein and all documents related
 to the source code ("Material") are owned by Intel Corporation or its suppliers
 or licensors. Title to the Material remains with Intel Corporation or its suppliers
 and licensors. The Material may contain trade secrets and proprietary
 and confidential information of Intel Corporation and its suppliers and licensors,
 and is protected by worldwide copyright and trade secret laws and treaty provisions.
 No part of the Material may be used, copied, reproduced, modified, published,
 uploaded, posted, transmitted, distributed, or disclosed in any way without Intel's
 prior express written permission.

 No license under any patent, copyright, trade secret or other intellectual
 property right is granted to or conferred upon you by disclosure or delivery
 of the Materials, either expressly, by implication, inducement, estoppel
 or otherwise. Any license under such intellectual property rights must
 be express and approved by Intel in writing.

 Unless otherwise agreed by Intel in writing, you may not remove or alter this notice
 or any other notice embedded in Materials by Intel or Intel's suppliers or licensors
 in any way.
*/

#include "test-gna-device-api.h"

#include "Macros.h"

#include "gna-api.h"
#include "../../gna-api/gna2-device-api.h"

#include <array>
#include <chrono>
#include <gtest/gtest.h>
#include <vector>

TEST_F(TestGnaDeviceApi, getDeviceCount)
{
	uint32_t deviceCount;
	auto status = Gna2DeviceGetCount(&deviceCount);

	ASSERT_EQ(status, Gna2StatusSuccess);
	ASSERT_EQ(deviceCount, static_cast<uint32_t>(1));
}

TEST_F(TestGnaDeviceApi, getDeviceCountNull)
{
	auto status = Gna2DeviceGetCount(nullptr);
	ASSERT_EQ(status, Gna2StatusNullArgumentNotAllowed);
}

TEST_F(TestGnaDeviceApi, openAndCloseDevice)
{
	uint32_t deviceIndex = 0;

	auto status = Gna2DeviceOpen(deviceIndex);
	ASSERT_EQ(status, Gna2StatusSuccess);

	status = Gna2DeviceClose(deviceIndex);
	ASSERT_EQ(status, Gna2StatusSuccess);
}

TEST_F(TestGnaDeviceApi, openDeviceWrongIndex)
{
	auto status = Gna2DeviceOpen(INT32_MAX);
	ASSERT_EQ(status, Gna2StatusIdentifierInvalid);
}

TEST_F(TestGnaDeviceApi, closeDeviceWrongIndex)
{
	uint32_t deviceIndex = 0;

	auto status = Gna2DeviceOpen(deviceIndex);
	ASSERT_EQ(status, Gna2StatusSuccess);

	status = Gna2DeviceClose(INT32_MAX);
	ASSERT_EQ(status, Gna2StatusIdentifierInvalid);

	status = Gna2DeviceClose(deviceIndex);
	ASSERT_EQ(status, Gna2StatusSuccess);
}

TEST_F(TestGnaDeviceApi, openDeviceTwice)
{
	uint32_t deviceIndex = 0;

	auto status = Gna2DeviceOpen(deviceIndex);
	ASSERT_EQ(status, Gna2StatusSuccess);

	status = Gna2DeviceOpen(deviceIndex);
	ASSERT_EQ(status, Gna2StatusIdentifierInvalid);

	status = Gna2DeviceClose(deviceIndex);
	ASSERT_EQ(status, Gna2StatusSuccess);
}

TEST_F(TestGnaDeviceApi, closeDeviceTwice)
{
	uint32_t deviceIndex = 0;

	auto status = Gna2DeviceOpen(deviceIndex);
	ASSERT_EQ(status, Gna2StatusSuccess);

	status = Gna2DeviceClose(deviceIndex);
	ASSERT_EQ(status, Gna2StatusSuccess);

	status = Gna2DeviceClose(deviceIndex);
	ASSERT_EQ(status, Gna2StatusIdentifierInvalid);
}

TEST_F(TestGnaDeviceApi, deviceVersion)
{
	enum Gna2DeviceVersion deviceVersion;
	uint32_t deviceIndex = 0;

	auto status = Gna2DeviceOpen(deviceIndex);
	ASSERT_EQ(status, Gna2StatusSuccess);

	status = Gna2DeviceGetVersion(deviceIndex, &deviceVersion);
	ASSERT_EQ(status, Gna2StatusSuccess);

	status = Gna2DeviceClose(deviceIndex);
	ASSERT_EQ(status, Gna2StatusSuccess);
}

TEST_F(TestGnaDeviceApi, deviceVersionNoDevice)
{
	enum Gna2DeviceVersion deviceVersion;
	uint32_t deviceIndex = 0;

	auto status = Gna2DeviceGetVersion(deviceIndex, &deviceVersion);
	ASSERT_EQ(status, Gna2StatusSuccess);
}

TEST_F(TestGnaDeviceApi, deviceVersionNull)
{
	uint32_t deviceIndex = 0;

	auto status = Gna2DeviceOpen(deviceIndex);
	ASSERT_EQ(status, Gna2StatusSuccess);

	status = Gna2DeviceGetVersion(deviceIndex, NULL);
	ASSERT_EQ(status, Gna2StatusNullArgumentNotAllowed);

	status = Gna2DeviceClose(deviceIndex);
	ASSERT_EQ(status, Gna2StatusSuccess);
}

TEST_F(TestGnaDeviceApi, setNumberOfThreads)
{
	uint32_t deviceIndex = 0;
	uint32_t numberOfThreads = 8;

	auto status = Gna2DeviceOpen(deviceIndex);
	ASSERT_EQ(status, Gna2StatusSuccess);

	status = Gna2DeviceSetNumberOfThreads(deviceIndex, numberOfThreads);
	ASSERT_EQ(status, Gna2StatusSuccess);

	status = Gna2DeviceClose(deviceIndex);
	ASSERT_EQ(status, Gna2StatusSuccess);
}

TEST_F(TestGnaDeviceApi, setNumberOfThreadsTooMany)
{
	uint32_t deviceIndex = 0;
	uint32_t numberOfThreads = 128;

	auto status = Gna2DeviceOpen(deviceIndex);
	ASSERT_EQ(status, Gna2StatusSuccess);

	status = Gna2DeviceSetNumberOfThreads(deviceIndex, numberOfThreads);
	ASSERT_EQ(status, Gna2StatusDeviceNumberOfThreadsInvalid);

	status = Gna2DeviceClose(deviceIndex);
	ASSERT_EQ(status, Gna2StatusSuccess);
}

TEST_F(TestGnaDeviceApi, setNumberOfThreadsNoDevice)
{
	uint32_t deviceIndex = 0;
	uint32_t numberOfThreads = 10;

	auto status = Gna2DeviceSetNumberOfThreads(deviceIndex, numberOfThreads);
	ASSERT_EQ(status, Gna2StatusSuccess);
}

