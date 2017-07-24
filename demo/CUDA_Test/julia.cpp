#include "funset.hpp"
#include "common.hpp"
#include <chrono>

struct Complex {
	float r, i;
	Complex(float a, float b) : r(a), i(b) {}
	float magnitude2() { return r * r + i * i; }
	Complex operator * (const Complex& a) { return Complex(r*a.r - i*a.i, i*a.r + r*a.i); }
	Complex operator + (const Complex& a) { return Complex(r + a.r, i + a.i); }
};

static int julia(int x, int y, int width, int height, float scale)
{
	float jx = scale * (float)(width / 2 - x) / (width / 2);
	float jy = scale * (float)(height / 2 - y) / (height / 2);

	Complex c(-0.8, 0.156);
	Complex a(jx, jy);

	for (int i = 0; i < 200; ++i) {
		a = a * a + c;
		if (a.magnitude2() > 1000)
			return 0;
	}

	return 1;
}

int julia_cpu(unsigned char* ptr, int width, int height, float scale, float* elapsed_time)
{
	auto start = std::chrono::steady_clock::now();

	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			int offset = x + y * width;

			int julia_value = julia(x, y, width, height, scale);
			ptr[offset * 4 + 0] = 255 * julia_value;
			ptr[offset * 4 + 1] = 0;
			ptr[offset * 4 + 2] = 0;
			ptr[offset * 4 + 3] = 255;
		}
	}

	auto end = std::chrono::steady_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
	*elapsed_time = duration.count() * 1.0e-6;

	return 0;
}
