#include "funset.hpp"
#include <limits.h>
#include <chrono>
#include <vector>
#include <algorithm>
#include <memory>
#include "common.hpp"

namespace {

typedef struct Rect {
	int x, y, width, height;
} Rect;
typedef struct Size {
	int width, height;
} Size;
typedef struct Point {
	int x, y;
} Point;

int borderInterpolate(int p, int len)
{
	if ((unsigned)p < (unsigned)len) {
		;
	} else {
		if (len == 1) return 0;

		int delta = 1;
		do {
			if (p < 0) p = -p - 1 + delta;
			else p = len - 1 - (p - len) - delta;
		} while ((unsigned)p >= (unsigned)len);
	}

	return p;
}

inline unsigned char saturate_cast(float v)
{
	int iv = (int)(v + (v >= 0 ? 0.5f : -0.5f));
	return (unsigned char)((unsigned)v <= UCHAR_MAX ? v : v > 0 ? UCHAR_MAX : 0);
}

void filter2D(const unsigned char** src, unsigned char* dst, int dststep, int count, int width, int ksize)
{
	std::vector<Point> coords;
	std::vector<float> coeffs;
	if (ksize == 1) {
		coords = { { 1, 0 }, { 0, 1 }, { 1, 1 }, { 2, 1 }, { 1, 2 } }; // kernel non zero position: (x, y)
		coeffs = { 1.f, 1.f, -4.f, 1.f, 1.f }; // kernel non zero value: 1, 1, -4, 1, 1
	} else {
		coords = { { 0, 0 }, { 2, 0 }, { 1, 1 }, { 0, 2 }, { 2, 2 } }; // kernel non zero position: (x, y)
		coeffs = { 2.f, 2.f, -8.f, 2.f, 2.f }; // kernel non zero value: 2, 2, -8, 2, 2
	}

	std::vector<unsigned char*> ptrs(coords.size());

	float _delta{ 0.f };
	const Point* pt = &coords[0];
	const float* kf = (const float*)&coeffs[0];
	const unsigned char** kp = (const unsigned char**)&ptrs[0];
	int nz = (int)coords.size();

	for (; count > 0; count--, dst += dststep, src++) {
		unsigned char* D = (unsigned char*)dst;

		for (int k = 0; k < nz; k++)
			kp[k] = (const unsigned char*)src[pt[k].y] + pt[k].x;

		for (int i = 0; i < width; i++) {
			float s0 = _delta;
			for (int k = 0; k < nz; k++)
				s0 += kf[k] * kp[k][i];
			D[i] = saturate_cast(s0);
		}
	}
}

int Laplacian_(const unsigned char* src_, unsigned char* dst_, int width_, int height_, int ksize_)
{
	const unsigned char* src = src_;
	unsigned char* dst = dst_;
	const Size ksize{ 3, 3 };
	const int maxBufRows = ksize.height + 3;
	const Point anchor{ 1, 1 };
	const Rect roi{ 0, 0, width_, height_ };
	const int dx1{ 1 }, dx2{ 1 };

	int borderLength = std::max(ksize.width - 1, 1);
	std::vector<int> borderTab(borderLength);
	borderTab[0] = borderInterpolate(-dx1, width_);
	borderTab[1] = borderInterpolate(width_, width_);
	std::vector<unsigned char*> rows(maxBufRows);

	const int* btab = &borderTab[0];
	int srcstep{ width_ }, dststep{ width_ };
	std::vector<unsigned char> ringBuf((width_ + ksize.width - 1) * maxBufRows, 0);
	int bufStep{ width_ + ksize.width - 1 };
	int startY = std::max(roi.y - anchor.y, 0), startY0 = startY, rowCount{ 0 }, dstY{ 0 };
	int endY = std::min(roi.y + roi.height + ksize.height - anchor.y - 1, height_);
	int esz = 1;
	unsigned char** brows = &rows[0];
	int bufRows = (int)rows.size();
	int kwidth = ksize.width;
	int kheight = ksize.height, ay = anchor.y;
	int _dx1 = dx1, _dx2 = dx2;
	int width1 = roi.width + kwidth - 1;
	int dy = 0, i = 0;
	int count = endY - startY;

	for (;; dst += dststep * i, dy += i) {
		int dcount = bufRows - ay - startY - rowCount + roi.y;
		dcount = dcount > 0 ? dcount : bufRows - kheight + 1;
		dcount = std::min(dcount, count);
		count -= dcount;

		for (; dcount-- > 0; src += srcstep) {
			int bi = (startY - startY0 + rowCount) % bufRows;
			unsigned char* brow = &ringBuf[0] + bi*bufStep;
			unsigned char* row = brow;

			if (++rowCount > bufRows) {
				--rowCount;
				++startY;
			}

			memcpy(row + _dx1*esz, src, (width1 - _dx2 - _dx1)*esz);

			for (i = 0; i < _dx1*esz; i++)
				row[i] = src[btab[i]];
			for (i = 0; i < _dx2*esz; i++)
				row[i + (width1 - _dx2)*esz] = src[btab[i + _dx1*esz]];
		}

		int max_i = std::min(bufRows, roi.height - (dstY + dy) + (kheight - 1));
		for (i = 0; i < max_i; i++) {
			int srcY = borderInterpolate(dstY + dy + i + roi.y - ay, height_);

			if (srcY < startY) return -1;
			if (srcY >= startY + rowCount) break;
			int bi = (srcY - startY0) % bufRows;
			brows[i] = &ringBuf[0] + bi*bufStep;
		}

		if (i < kheight) break;
		i -= kheight - 1;

		filter2D((const unsigned char**)brows, dst, dststep, i, roi.width, ksize_);
	}

	dstY += dy;
	if (dstY > roi.height) return -1;

	return 0;
}

int Laplacian(const unsigned char* src_, unsigned char* dst_, int width_, int height_, int ksize_)
{
	const int kernel_size{ 3 };
	std::vector<float> kernel;
	if (ksize_ == 1) kernel = { 0.f, 1.f, 0.f, 1.f, -4.f, 1.f, 0.f, 1.f, 0.f };
	else kernel = { 2.f, 0.f, 2.f, 0.f, -8.f, 0.f, 2.f, 0.f, 2.f };
	int new_width = width_ + kernel_size - 1, new_height = height_ + kernel_size - 1;
	std::unique_ptr<unsigned char[]> data(new unsigned char[new_width * new_height]);
	unsigned char* p = data.get();

	for (int y = 0; y < new_height; ++y) {
		if (y != 0 && y != new_height - 1) {
			for (int x = 0; x < new_width; ++x) {
				if (x == 0) {
					p[y * new_width + x] = src_[(y - 1) * width_ + 1];
				} else if (x == new_width - 1) {
					p[y * new_width + x] = src_[(y - 1) * width_ + (width_ - 1 - 1)];
				} else {
					p[y * new_width + x] = src_[(y - 1) * width_ + (x - 1)];
				}
			}
		}

		if (y == new_height - 1) {
			for (int x = 0; x < new_width; ++x) {
				p[y * new_width + x] = p[(y - 2) * new_width + x];
			}

			for (int x = 0; x < new_width; ++x) { // y = 0
				p[x] = p[2 * new_width + x];
			}
		}
	}

	for (int y = 1; y < new_height - 1; ++y) {
		for (int x = 1; x < new_width - 1; ++x) {
			float value{ 0.f };
			int count{ 0 };
			for (int m = -1; m <= 1; ++m) {
				for (int n = -1; n <= 1; ++n) {
					value += p[(y + m) * new_width + (x + n)] * kernel[count++];
				}
			}

			if (value < 0.) dst_[(y - 1) * width_ + (x - 1)] = 0;
			else if (value > 255.) dst_[(y - 1) * width_ + (x - 1)] = 255;
			else dst_[(y - 1) * width_ + (x - 1)] = static_cast<unsigned char>(value);
		}
	}

	return 0;
}

} // namespace

int laplacian_cpu(const unsigned char* src, int width, int height, int ksize, unsigned char* dst, float* elapsed_time)
{
	int ret{ -1 };
	// ksize == 1: kernel={ 0, 1, 0, 1, -4, 1, 0, 1, 0 }
	// ksize == 3: kernel={ 2, 0, 2, 0, -8, 0, 2, 0, 2 }
	CHECK(ksize == 1 || ksize == 3);
	//TIME_START_CPU

	ret = Laplacian(src, dst, width, height, ksize);

	//TIME_END_CPU

	return ret;
}
