
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include "CriticalPolyphaseFilterbank.h"
#include "debug.h"
#include <time.h>

bool checkinput(int argc, char *argv[]) {
	if (argc != 3) {
		printf("You supplied an invalid number of arguments. Must be number of taps, number of channels, and number of spectra.\n");
		exit(0);
	}
	bool checker = true;
	errno = 0;
	char *endptr1, *endptr2, *endptr3;
	long int ntaps = strtol(argv[1], &endptr1, 10);
	long int nchans = strtol(argv[2], &endptr2, 10);
	long int nspectra = strtol(argv[3], &endptr3, 10);
	// Check whether arguments supplied for number of taps is a number.
	if (endptr1 == argv[1]) {
		printf("Invalid number for number of taps: %%s \n", argv[1]);
		checker = false;
	}
	else if (*endptr1) {
		printf("Trailing characters after number of taps:  %%s \n", argv[1]);
		checker = false;
	}
	else if (errno == ERANGE) {
		printf("Number of taps of range:  %%s \n", argv[1]);
		checker = false;
	}
	// Check whether arguments supplied for number of channels is a number.
	if (endptr2 == argv[2]) {
		printf("Invalid number for number of channels: %%s \n", argv[2]);
		checker = false;
	}
	else if (*endptr2) {
		printf("Trailing characters after number of channels:  %%s \n", argv[2]);
		checker = false;
	}
	else if (errno == ERANGE) {
		printf("Number of channels of range:  %%s \n", argv[2]);
		checker = false;
	}
	// Check whether arguments supplied for number of spectra is a number.
	if (endptr3 == argv[3]) {
		printf("Invalid number for number of spectra: %%s \n", argv[3]);
		checker = false;
	}
	else if (*endptr3) {
		printf("Trailing characters after number of spectra:  %%s \n", argv[3]);
		checker = false;
	}
	else if (errno == ERANGE) {
		printf("Number of spectra of range:  %%s \n", argv[3]);
		checker = false;
	}
	// Stop execution in case one of the params is not of right type.
	if (!checker) { exit(0); }
	// TODO any semantic checks required?
	// int ntaps = atoi(argv[1]);
	// int nchans = atoi(argv[2]);
	// int nspectra = atoi(argv[3]);
	return checker;
}

int main(int argc, char *argv[])
{
	// TODO: Get tabs, channels and filter_coefficients from call/Input file.
	checkinput(argc, argv);
	std::size_t ntaps = atoi(argv[1]);
	std::size_t nchans = atoi(argv[2]);
	std::size_t nspectra = atoi(argv[3]);
	// TODO: Read File for input.
	// FILE* stream = ReadTimeSeriesFile(path);
	// TODO Check Memory before starting.

	double cumulative_error, mean_error;

	printf("Welcome everybody, we will start to calculate the PFB...\n\n");

	float *h_input;
	float *h_output;
	float *h_coeff;
	
	int input_size = (nspectra + ntaps - 1)*nchans;
	int output_size = nspectra * nchans;
	int coeff_size = ntaps * nchans;
	printf("Inputsize: %d Outputsize: %d Coefficientsize: %d \n", input_size, output_size, coeff_size);

	if (PRINT) printf("\nHost memory allocation...\t");
	h_input = (float *)malloc(input_size * sizeof(float));
	h_output = (float *)malloc(output_size * sizeof(float));
	h_coeff = (float *)malloc(coeff_size * sizeof(float));
	if (PRINT) printf("done.");

	if (PRINT) printf("\nHost memory memset...\t\t");
	memset(h_output, 0.0, output_size * sizeof(float2));
	if (PRINT) printf("done.");

	if (PRINT) printf("\nRandom data set...\t\t");
	
	srand(time(NULL));
	if (TESTSMALL) printf("Input: [\n");

	for (int i = 0; i < (int)input_size; i++) {
		h_input[i] = rand() / (float)RAND_MAX;
		if (TESTSMALL) {
			printf("%0.6f ", h_input[i]);
			if (i % nchans == 0) {
				printf("\n");
			}
		}
	}
	if (TESTSMALL) printf("]\nCoefficients: [\n");
	for (int i = 0; i < (int)coeff_size; i++) {
		h_coeff[i] = rand() / (float)RAND_MAX;
		if (TESTSMALL) {
			printf("%0.6f ", h_coeff[i]);
			if (i % nchans == 0) {
				printf("\n");
			}
		}
	}
	if (TESTSMALL) printf("]\n");
	const float* constin = h_input;

	CriticalPolyphaseFilterbank PFB = CriticalPolyphaseFilterbank(nchans, ntaps, h_coeff);
	PFB.process(constin, h_output);

	return 0;
}
