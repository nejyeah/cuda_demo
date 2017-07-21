#include "funset.hpp"
#include <chrono>

int dot_product_cpu(const float* A, const float* B, float* value, int elements_num, float* elapsed_time)
{
	auto start = std::chrono::steady_clock::now();

	*value = 0.f;
	for (int i = 0; i < elements_num; ++i) {
		(*value) += A[i] * B[i];
	}

	auto end = std::chrono::steady_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
	*elapsed_time = duration.count() * 1.0e-6;

	return 0;
}

