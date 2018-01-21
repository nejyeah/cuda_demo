#include "funset.hpp"
#include <chrono>
#include <vector>
#include "common.hpp"

int histogram_equalization_cpu(const unsigned char* src, int width, int height, unsigned char* dst, float* elapsed_time)
{
	TIME_START_CPU

	std::vector<int> hist(256, 0);
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			++hist[src[y * width + x]];
		}
	}

	TIME_END_CPU

	return 0;
}

