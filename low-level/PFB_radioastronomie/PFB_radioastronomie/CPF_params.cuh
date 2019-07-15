#pragma once
#include <iostream>
#ifndef pfb_nchans
std::size_t pfb_nchans = 0;
#endif
#ifndef pfb_ntaps
std::size_t pfb_ntaps = 0;
#endif
#ifndef pfb_coeff
float* pfb_coeff;
#endif
#ifndef pfb_SM_Columns
int pfb_SM_Columns;
#endif