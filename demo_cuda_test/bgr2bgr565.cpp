#include "funset.hpp"
#include <chrono>
#include "common.hpp"

int bgr2bgr565_cpu(const unsigned char* src, int width, int height, unsigned char* dst, float* elapsed_time)
{
	TIME_START_CPU

	for (int y = 0; y < height; ++y) {
		const unsigned char* p1 = src + y * width * 3;
		unsigned char* p2 = dst + y * width * 2;

		for (int x = 0; x < width; ++x, p1+=3) {
			((unsigned short*)p2)[x] = (unsigned short)((p1[0] >> 3) | ((p1[1] & ~3) << 3) | ((p1[2] & ~7) << 8));
		}
	}

	TIME_END_CPU

	return 0;
}
