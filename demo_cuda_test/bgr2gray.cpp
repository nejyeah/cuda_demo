#include "funset.hpp"
#include <chrono>
#include "common.hpp"

int bgr2gray_cpu(const unsigned char* src, int width, int height, unsigned char* dst, float* elapsed_time)
{
	TIME_START_CPU

	const int R2Y{ 4899 }, G2Y{ 9617 }, B2Y{ 1868 }, yuv_shift{ 14 };

	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			dst[y * width + x] = (unsigned char)((src[y*width * 3 + 3 * x + 0] * B2Y +
				src[y*width * 3 + 3 * x + 1] * G2Y + src[y*width * 3 + 3 * x + 2] * R2Y) >> yuv_shift);
		}
	}

	TIME_END_CPU

	return 0;
}