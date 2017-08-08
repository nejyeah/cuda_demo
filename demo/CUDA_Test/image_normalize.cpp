#include "funset.hpp"
#include <vector>
#include <chrono>
#include "common.hpp"

int image_normalize_cpu(const float* src, float* dst, int width, int height, int channels, float* elapsed_time)
{
	auto start = std::chrono::steady_clock::now();

	const int offset{ width * height };
	for (int i = 0; i < channels; ++i) {
		const float* p1 = src + offset * i;
		float* p2 = dst + offset * i;
		float mean{ 0.f }, sd{ 0.f };

		for (int t = 0; t < offset; ++t) {
			mean += p1[t];
			sd += pow(p1[t], 2.f);
			p2[t] = p1[t];
		}

		mean /= offset;
		sd /= offset;
		sd -= pow(mean, 2.f);
		sd = sqrt(sd);
		if (sd < EPS_) sd = 1.f;

		for (int t = 0; t < offset; ++t) {
			p2[t] = (p1[t] - mean) / sd;
		}
	}

	auto end = std::chrono::steady_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
	*elapsed_time = duration.count() * 1.0e-6;

	return 0;
}


