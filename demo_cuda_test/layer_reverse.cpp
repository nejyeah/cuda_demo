#include "funset.hpp"
#include <chrono>
#include <algorithm>

int layer_reverse_cpu(const float* src, float* dst, int length, const std::vector<int>& vec, float* elapsed_time)
{
	auto start = std::chrono::steady_clock::now();

	for (int i = 0; i < length; ++i) {
		auto index1 = (i / vec[0]) % vec[1];
		auto index2 = vec[0] * (vec[1] - 2 * index1 - 1) + i;
		index2 = std::max(0, std::min(length - 1, index2));
		dst[index2] = src[i];
	}

	auto end = std::chrono::steady_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
	*elapsed_time = duration.count() * 1.0e-6;

	return 0;
}
