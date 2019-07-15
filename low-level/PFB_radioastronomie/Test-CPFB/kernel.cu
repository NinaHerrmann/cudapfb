
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "gtest/gtest.h"

int main(int argc, char** argv)
{
	cout << "******* TESTs *******" << endl;
	::testing::InitGoogleTest(&argc, argv);
	RUN_ALL_TESTS();
	cout << endl;

	return 0;
}
