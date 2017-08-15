#include "funset.hpp"
#include <cmath>
#include <chrono>

int ripple_cpu(unsigned char* ptr, int width, int height, int ticks, float* elapsed_time)
{
	auto start = std::chrono::steady_clock::now();

	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			int offset = x + y * width;
			float fx = x - width / 2;
			float fy = y - height / 2;
			float d = sqrtf(fx * fx + fy * fy);
			unsigned char grey = (unsigned char)(128.0f + 127.0f * cos(d / 10.0f - ticks / 7.0f) / (d / 10.0f + 1.0f));

			ptr[offset * 4 + 0] = grey;
			ptr[offset * 4 + 1] = grey;
			ptr[offset * 4 + 2] = grey;
			ptr[offset * 4 + 3] = 255;
		}
	}

	auto end = std::chrono::steady_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
	*elapsed_time = duration.count() * 1.0e-6;

	return 0;
}
