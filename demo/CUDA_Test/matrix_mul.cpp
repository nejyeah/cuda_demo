#include "funset.hpp"
#include <vector>
#include <chrono>
#include "common.hpp"

int matrix_mul_cpu(const float* A, const float* B, float* C, int colsA, int rowsA, int colsB, int rowsB, float* elapsed_time)
{
	auto start = std::chrono::steady_clock::now();

	CHECK(colsA == rowsB);

	for (int y = 0; y < rowsA; ++y) {
		for (int x = 0; x < colsB; ++x) {
			float sum{ 0.f };
			for (int t = 0; t < colsA; ++t) {
				sum += A[y * colsA + t] * B[t * colsB + x];
			}

			C[y * colsB + x] = sum;
		}
	}

	auto end = std::chrono::steady_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
	*elapsed_time = duration.count() * 1.0e-6;

	return 0;
}
