#include "CriticalPolyphaseFilterbank.h"
#include <vector>
#include <stddef.h>
#include <cstdint>
#include <time.h>
#include "debug.h"
/**
* @brief Construct a new critically sampled polyphase filterbank
*
* @param[in] nchans The number of spectral channels to produce
* @param[in] ntaps The number of filter taps to use
* @param filter_coefficients The filter coefficients
*
* @detail The number of filter coefficients should be equal to ntaps x nchans.
*/
CriticalPolyphaseFilterbank::CriticalPolyphaseFilterbank(std::size_t nchans, std::size_t ntaps,
	FilterCoefficientsType h_coeff)
{
	pfb_nchans = nchans;
	pfb_ntaps = ntaps;
	pfb_coeff = h_coeff;
}	

/**
* @brief Apply the polyphase filter to a block of timeseries data
*
* @param input The timeseries to process
* @param output The channelised filtered output data
*
* @tparam InputType The input data type
* @tparam OutputType The output data type
*/
void CriticalPolyphaseFilterbank::process(const float *input, float *output) {
	// TODO Last parameter should be spectra however in streaming... ? 
	fir(input, output, pfb_coeff, pfb_nchans, pfb_ntaps, 10);
	fft(output, pfb_nchans);
};
void CriticalPolyphaseFilterbank::_fft(float *buf, float *out, int n, int step)
{
	if (step < n) {
		_fft(out, buf, n, step * 2);
		_fft(out + step, buf + step, n, step * 2);

		for (int i = 0; i < n; i += 2 * step) {
			// TODO float t = float(-I * PI * i / n) * out[i + step];
			float t = 0.0;
			buf[i / 2] = out[i] + t;
			buf[(i + n) / 2] = out[i] - t;
		}
	}
}

void CriticalPolyphaseFilterbank::fft(float *h_output, int nchans)
{
	printf("\n ... Starting to process the FFT ... \n");
	clock_t begin = clock();

	float *out;
	for (int i = 0; i < nchans; i++) out[i] = h_output[i];

	_fft(h_output, out, nchans, 1);
	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	printf("Execution time %0.3f \n\n", time_spent);
}
void CriticalPolyphaseFilterbank::fir(const float *input, float *output, float *filter_coefficients, int ntaps, int nchans, int nspectra)
{
	printf("\n ... Starting to process the FIR ... \n");
	clock_t begin = clock();

	float ftemp;
	ftemp = 0;
	// For each value in the spectras.
	for (int h = 0; h < nchans * nspectra; h++) {
		for (int z = 0; z < ntaps; z++) {
			// For each tap add the multiplied value.
			ftemp += filter_coefficients[h % nchans + z * nchans] * input[h + z * nchans];
		}
		// After all taps have been used write value to output.
		output[h] = ftemp;
		ftemp = 0;
	}
	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	printf("Execution time %0.3f \n\n", time_spent);
	if (TESTSMALL) {
		printf("\n Output fir looks like: \n [\n");
		int c = 0;
		for (int n = 0; n < (nspectra); n++) {
			for (int m = 0; m < nchans; m++) {
				printf(" %d : %.6f", c, output[c]);
				c++;
			}
			printf("\n");
		}
		printf("]\n");
	}
}
