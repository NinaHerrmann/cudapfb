#pragma once
#include <vector>

class CriticalPolyphaseFilterbank
{
public:
	typedef float* FilterCoefficientsType;
	std::size_t pfb_nchans;
	std::size_t pfb_ntaps;
	FilterCoefficientsType pfb_coeff;
public:
	/**
	* @brief Construct a new critically sampled polyphase filterbank
	*
	* @param[in] nchans The number of spectral channels to produce
	* @param[in] ntaps The number of filter taps to use
	* @param filter_coefficients The filter coefficients
	*
	* @detail The number of filter coefficients should be equal to ntaps x nchans.
	*/
	CriticalPolyphaseFilterbank(std::size_t nchans, std::size_t ntaps,
		FilterCoefficientsType filter_coefficients);
	
	// CriticalPolyphaseFilterbank(CriticalPolyphaseFilterbank const &) = delete;
	/**
	* @brief Apply the polyphase filter to a block of timeseries data
	*
	* @param input The timeseries to process
	* @param output The channelised filtered output data
	*
	* @tparam InputType The input data type
	* @tparam OutputType The output data type
	*/
	//template <typename InputType, typename OutputType>
	//void process(const InputType *input, OutputType *output);
	
	void process(const float *input, float *output);
	void _fft(float *buf, float *out, int n, int step);
	void fft(float *h_output, int nchans);
	void fir(const float *input, float *output, float *filter_coefficients, int ntaps, int nchans, int nspectra);
};
