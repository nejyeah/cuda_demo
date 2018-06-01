#include "funset.hpp"
#include <chrono>
#include <memory>
#include <vector>

static void copy_const_kernel(float* iptr, const float* cptr, int width, int height)
{
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			int offset = x + y * width;
			if (cptr[offset] != 0) iptr[offset] = cptr[offset];
		}
	}
}

static void blend_kernel(float* outSrc, const float* inSrc, int width, int height, float speed)
{
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			int offset = x + y * width;

			int left = offset - 1;
			int right = offset + 1;
			if (x == 0) ++left;
			if (x == width - 1) --right;

			int top = offset - height;
			int bottom = offset + height;
			if (y == 0) top += height;
			if (y == height - 1) bottom -= height;

			outSrc[offset] = inSrc[offset] + speed * (inSrc[top] + inSrc[bottom] + inSrc[left] + inSrc[right] - inSrc[offset] * 4);
		}
	}
}

static unsigned char value(float n1, float n2, int hue)
{
	if (hue > 360) hue -= 360;
	else if (hue < 0) hue += 360;

	if (hue < 60)
		return (unsigned char)(255 * (n1 + (n2 - n1)*hue / 60));
	if (hue < 180)
		return (unsigned char)(255 * n2);
	if (hue < 240)
		return (unsigned char)(255 * (n1 + (n2 - n1)*(240 - hue) / 60));
	return (unsigned char)(255 * n1);
}

static void float_to_color(unsigned char *optr, const float *outSrc, int width, int height)
{
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			int offset = x + y * width;

			float l = outSrc[offset];
			float s = 1;
			int h = (180 + (int)(360.0f * outSrc[offset])) % 360;
			float m1, m2;

			if (l <= 0.5f) m2 = l * (1 + s);
			else m2 = l + s - l * s;
			m1 = 2 * l - m2;

			optr[offset * 4 + 0] = value(m1, m2, h + 120);
			optr[offset * 4 + 1] = value(m1, m2, h);
			optr[offset * 4 + 2] = value(m1, m2, h - 120);
			optr[offset * 4 + 3] = 255;
		}
	}
}

int heat_conduction_cpu(unsigned char* ptr, int width, int height, const float* src, float speed, float* elapsed_time)
{
	auto start = std::chrono::steady_clock::now();

	std::vector<float> inSrc(width*height, 0.f);
	std::vector<float> outSrc(width*height, 0.f);

	for (int i = 0; i < 90; ++i) {
		copy_const_kernel(inSrc.data(), src, width, height);
		blend_kernel(outSrc.data(), inSrc.data(), width, height, speed);
		std::swap(inSrc, outSrc);
	}

	float_to_color(ptr, inSrc.data(), width, height);

	auto end = std::chrono::steady_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
	*elapsed_time = duration.count() * 1.0e-6;

	return 0;
}
