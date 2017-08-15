#include "funset.hpp"
#include <chrono>
#include "common.hpp"

int calculate_histogram_cpu(const unsigned char* data, int length, unsigned int* hist, unsigned int& value, float* elapsed_time)
{
	auto start = std::chrono::steady_clock::now();

	for (int i = 0; i < length; ++i) {
		++hist[data[i]];
	}

	value = 0;
	for (int i = 0; i < 256; ++i) {
		value += hist[i];
	}

	auto end = std::chrono::steady_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
	*elapsed_time = duration.count() * 1.0e-6;

	return 0;
}
