//
// pch.cpp
// Include the standard header and generate the precompiled header.
//

#include "pch.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include "params.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>

bool CriticalPolyphaseFilterbankTester::FIR_c_reference()
{
	// TODO: Implement a c function for a FIR Filter.
	return true;
}
bool CriticalPolyphaseFilterbankTester::FIR_cuda_caller()
{
	// TODO: Implement a c function for a FIR Filter.
	return true;
}
