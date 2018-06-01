#include "funset.hpp"
#include <chrono>

int streams_cpu(const int* a, const int* b, int* c, int length, float* elapsed_time)
{
	auto start = std::chrono::steady_clock::now();

	const int N{ length / 20 };

	for (int x = 0; x < 20; ++x) {
		const int* pa = a + x * N;
		const int* pb = b + x * N;
		int* pc = c + x * N;

		for (int idx = 0; idx < N; ++idx) {
			int idx1 = (idx + 1) % 256;
			int idx2 = (idx + 2) % 256;
			float as = (pa[idx] + pa[idx1] + pa[idx2]) / 3.0f;
			float bs = (pb[idx] + pb[idx1] + pb[idx2]) / 3.0f;
			pc[idx] = (as + bs) / 2;
		}
	}

	auto end = std::chrono::steady_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
	*elapsed_time = duration.count() * 1.0e-6;

	return 0;
}
