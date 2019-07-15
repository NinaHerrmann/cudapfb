#pragma once
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "debug.h"
#include "params.h"
#include <stdlib.h>
#include "CriticalPolyphaseFilterbank.h"
#include <errno.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <Windows.h>

extern int errno;

char * buildbasepath(const char * timestampchar) {
	char buffer[40];
	const char *basepath = "C:\\Users\\b98\\PFB_OUTPUT\\";
	strcpy(buffer, basepath);
	strcat(buffer, timestampchar);
	CreateDirectoryA(buffer, NULL);

	return buffer;
}

char * buildcustompath(const char *basepath, const char * fileending) {
	char buffer[50];
	strcpy(buffer, basepath);
	strcat(buffer, fileending);
	char * p = new char;
	p = buffer;
	return p;
}

FILE * openfile(char * path) {
	FILE * f = fopen(path, "w+");
	if (f == NULL) {
		int errnum = errno;
		fprintf(stderr, "Error opening file!\nValue of errno: %d\n", errno);
		exit(1);
	}
	return f;
}

void writefloattofile(char * filepath, std::size_t datasize, float* data, bool generaterandom) {
	for (int n = 0; n < datasize; n++) {
		if (generaterandom) 	data[n] = rand() / (float)RAND_MAX;
		if (CMDWRITE) {
			if (n%CHANNELS == 0) {
				printf("\n");
			}
			printf("%0.6f", data[n]);
		}
	}
	if (WRITE) {
		FILE* f = openfile(filepath);
		for (int n = 0; n < datasize; n++) {
			fprintf(f, " %0.6f\n", data[n]);
		}
		fclose(f);
	}
}

void writefloat2tofile(char * filepath, std::size_t datasize, float2* data) {
	int halfcounter = 0;
	for (int n = 0; n < datasize; n++) {
		if (halfcounter > (CHANNELS / 2) && halfcounter != 0) {
			n = n + (halfcounter - 3);
			halfcounter = 0;
			continue;
		}
		halfcounter++;
		if (CMDWRITE) {
			if (n%CHANNELS == 0) {
				printf("\n");
			}
			printf("(%0.6f+%0.6fj)\n", data[n].x, data[n].y);
		}
	}
	halfcounter = 0;
	if (WRITE) {
		FILE* f = openfile(filepath);
		for (int n = 0; n < datasize; n++) {
			if (halfcounter > (CHANNELS / 2) && halfcounter != 0) {
				n = n + (halfcounter - 3);
				halfcounter = 0;
				continue;
			}
			halfcounter++;
			fprintf(f, " (%0.6f+%0.6fj)\n", data[n].x, data[n].y);
		}
		fclose(f);
	}
}

void writefloattofilevector(char * filepath, std::size_t datasize, std::vector<float>& data, bool generaterandom) {
	for (int n = 0; n < datasize; n++) {
		if (generaterandom) data.push_back(rand() / (float)RAND_MAX);
		if (CMDWRITE) {
			if (n%CHANNELS == 0) {
				printf("\n");
			}
			printf("%0.6f", data[n]);
		}
	}
	if (WRITE) {
		FILE* f = openfile(filepath);
		for (int n = 0; n < datasize; n++) {
			fprintf(f, " %0.6f\n", data[n]);
		}
		fclose(f);
	}
}

void generateandwriteinput(std::size_t ntaps, std::size_t nchans, std::size_t nspectra, long timestamp, std::vector<float>& h_input, std::size_t input_size, std::size_t coeff_size, std::vector<float>& h_coeff, bool gen) {
	char buffer2[40];
	char buffer[12];
	const char * timestampchar = "1561096631";

	strcpy(buffer2, buildbasepath(timestampchar));
	char buffer3[50];
	const char *name = "//input";
	strcpy(buffer3, buildcustompath(buffer2, name));

	// Generate and write the input.
	writefloattofilevector(buffer3, input_size, h_input, true);

	strcpy(buffer3, buildcustompath(buffer2, "//coeff"));

	// Generate and write coefficients.
	if (gen) writefloattofilevector(buffer3, coeff_size, h_coeff, true);

	if (WRITE) {
		strcpy(buffer3, buildcustompath(buffer2, "//settings"));
		FILE * f = openfile(buffer3);
		fprintf(f, "%s;%zd;%zd;%zd;%d", timestampchar, nchans, ntaps, nspectra, SM_Columns);
		fclose(f);

		char * timestamppath = "C:\\Users\\b98\\PFB_OUTPUT\\timestamp.txt";
		f = openfile(timestamppath);
		fprintf(f, "%s", timestampchar);
		fclose(f);
	}
}

void writefloat2tofilevector(char * filepath, int datasize, std::vector<float2>& data, bool skipdata) {
	// This might look weird, but actually in the output of the cufftPlan are only half+1 of the output meaningful values than half 0 etc. 
	// So some way was required to write half of number of channels skip other half and start again. 
	int halfcounter = 0;
	for (int n = 0; n < datasize; n++) {
		if (skipdata) {
			if (halfcounter > (CHANNELS / 2) && halfcounter != 0) {
				n = n + (halfcounter - 3);
				halfcounter = 0;
				continue;
			}
			halfcounter++;
		}
		if (CMDWRITE) { printf(" (%0.6f+%0.6fj)\n", data[n].x, data[n].y); }
	}
	if (WRITE) {
		halfcounter = 0;
		FILE* f = openfile(filepath);
		for (int n = 0; n < datasize; n++) {
			if (skipdata) {
				if (halfcounter > (CHANNELS / 2) && halfcounter != 0) {
					n = n + (halfcounter - 3);
					halfcounter = 0;
					continue;
				}
				halfcounter++;
			}
			fprintf(f, " (%0.6f+%0.6fj)\n", data[n].x, data[n].y);
		}
		fclose(f);
	}	

}

bool writedat(char str[], const char *timestamp, int num_blocks, float fir_time, float fft_time, float mem_time_in, float mem_time_out, const int nChannels, const int nTaps, const int nStreams, const int nThreads, const int sh_size) {
	std::ofstream FILEOUT;
	FILEOUT.open(str, std::ofstream::out | std::ofstream::app);
	double flops = 0.0f;
	double bandwidth = 0.0f;
	bandwidth = (3.0*nChannels*nTaps * sizeof(float) + nChannels * 2.0 * sizeof(float))*(num_blocks)*1000.0 / fir_time;
	flops = (4.0*nTaps)*nChannels*(num_blocks)*1000.0 / fir_time;
	//------------------
	FILEOUT << std::fixed << std::setprecision(8) << timestamp << "\t" << nTaps << "\t" << nChannels << "\t" << NSPECTRA << "\t" << num_blocks << "\t" << fir_time / 1000.0 << "\t" << (fir_time + fft_time) / 1000.0 << "\t" << std::scientific << bandwidth << "\t" << flops << "\t" << std::fixed << mem_time_in / 1000.0 << "\t" << mem_time_out / 1000.0 << "\t" << nStreams << "\t" << nThreads << "\t" << sh_size << std::endl;
	FILEOUT.close();
	return 0;
}

void writeoutput(long timestamp, float2 * h_output2, const char *chartimestamp, std::size_t num_blocks, double fir_time, double fft_time, double mem_time_in, double mem_time_out, std::size_t nChannels, std::size_t nTaps, const int nStreams, const int nThreads, const int sh_size) {
	char buffer[12];
	char buffer2[40];
	char buffer3[50];

	const char * timestampchar = "1561096631";

	const char *basepath = "C:\\Users\\b98\\PFB_OUTPUT\\";
	strcpy(buffer2, basepath);
	strcpy(buffer3, basepath);
	strcat(buffer3, "time.dat");
	if (WRITE) writedat(buffer3, timestampchar, num_blocks, fir_time, fft_time, mem_time_in, mem_time_out, nChannels, nTaps, 0, THREADS_PER_BLOCK, DATA_SIZE);

	strcpy(buffer2, buildbasepath(timestampchar));
	strcpy(buffer3, buildcustompath(buffer2, "//output"));
	writefloat2tofile(buffer3, NSPECTRA * CHANNELS, h_output2);
}

