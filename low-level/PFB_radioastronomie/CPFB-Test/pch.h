//
// pch.h
// Header for standard system include files.
//

#pragma once

#include "gtest/gtest.h"
//using namespace psrada_cpp_meerkat_fbfuse_criticalpolyphasefilterbank_cuh;
#pragma once

#include "pch.h"

class CriticalPolyphaseFilterbankTester
{
public:
	bool FIR_c_reference();
	bool FIR_cuda_caller();
};