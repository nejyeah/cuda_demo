#include "funset.hpp"
#include <chrono>

int long_vector_add_cpu(const float* A, const float* B, float* C, int elements_num, float* elapsed_time)
{
	auto start = std::chrono::steady_clock::now();

	for (int i = 0; i < elements_num; ++i) {
		C[i] = A[i] + B[i];
	}

	auto end = std::chrono::steady_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
	*elapsed_time = duration.count() * 1.0e-6;

	return 0;
}

