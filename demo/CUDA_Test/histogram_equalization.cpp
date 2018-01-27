#include "funset.hpp"
#include <chrono>
#include <vector>
#include <algorithm>
#include "common.hpp"

int histogram_equalization_cpu(const unsigned char* src, int width, int height, unsigned char* dst, float* elapsed_time)
{
	TIME_START_CPU

	const int hist_sz{ 256 };
	std::vector<int> hist(hist_sz, 0), lut(hist_sz, 0);
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			++hist[src[y * width + x]];
		}
	}

	int i{ 0 };
	while (!hist[i]) ++i;

	int total{ width * height };
	if (hist[i] == total) {
		unsigned char* p = dst;
		std::for_each(p, p + total, [i](unsigned char& value) { value = i; });
		return 0;
	}

	float scale = (hist_sz - 1.f) / (total - hist[i]);
	int sum = 0;

	for (lut[i++] = 0; i < hist_sz; ++i) {
		sum += hist[i];
		lut[i] = static_cast<unsigned char>(sum * scale + 0.5f);
	}

	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			dst[y * width + x] = static_cast<unsigned char>(lut[src[y * width + x]]);
		}
	}

	TIME_END_CPU

	return 0;
}

