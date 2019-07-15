#pragma once
// Settings Debug to true means that some basic information is print on the CMD.
#ifndef DEBUG
#define DEBUG true
#endif
// Setting Write to true means that the intput, the coefficients, the intermediate result after the FIR, and the output is written to a file.
// Moreover, the executiontime and several measurements such as bandwidth and flops per second are written to a file.
// TODO: This is currently tailored to windows systems.
#ifndef WRITE
#define WRITE true
#endif
// The CMDWRITE options prints the input, coefficients, the intermediate fir result, and the output in the cmd.
// ! Do not use this option for big numbers of spectra, channels, and taps! 
#ifndef CMDWRITE
#define CMDWRITE false
#endif
// Generates a Program for Benchmarks.
#ifndef TIMER
#define TIMER true
#endif