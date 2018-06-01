#include "funset.hpp"
#include <chrono>
#include "common.hpp"

int green_ball_cpu(unsigned char* ptr, int width, int height, float* elapsed_time)
{
	auto start = std::chrono::steady_clock::now();

	const float period{ 128.0f };
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			int offset = x + y * width;
			unsigned char grey = (unsigned char)(255 * (sinf(x * 2.0f * PI / period) + 1.0f) *
				(sinf(y * 2.0f * PI / period) + 1.0f) / 4.0f) ;

			ptr[offset * 4 + 0] = 0;
			ptr[offset * 4 + 1] = grey;
			ptr[offset * 4 + 2] = 0;
			ptr[offset * 4 + 3] = 255;
		}
	}

	auto end = std::chrono::steady_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
	*elapsed_time = duration.count() * 1.0e-6;

	return 0;
}
