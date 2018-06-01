#include "funset.hpp"
#include <chrono>
#include <memory>
#include "common.hpp"

// 通过一个数据结构对球面建模
struct Sphere {
	float r, b, g;
	float radius;
	float x, y, z;
	float hit(float ox, float oy, float *n)
	{
		float dx = ox - x;
		float dy = oy - y;
		if (dx*dx + dy*dy < radius*radius) {
			float dz = sqrtf(radius*radius - dx*dx - dy*dy);
			*n = dz / sqrtf(radius * radius);
			return dz + z;
		}
		return -INF;
	}
};

int ray_tracking_cpu(const float* a, const float* b, const float* c, int sphere_num, unsigned char* ptr, int width, int height, float* elapsed_time)
{
	auto start = std::chrono::steady_clock::now();

	std::unique_ptr<Sphere[]> spheres(new Sphere[sphere_num]);
	for (int i = 0, t = 0; i < sphere_num; ++i, t+=3) {
		spheres[i].r = a[t];
		spheres[i].g = a[t+1];
		spheres[i].b = a[t+2];
		spheres[i].x = b[t];
		spheres[i].y = b[t+1];
		spheres[i].z = b[t+2];
		spheres[i].radius = c[i];
	}

	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			int offset = x + y * width;
			float ox{ (x - width / 2.f) };
			float oy{ (y - height / 2.f) };
			float r{ 0 }, g{ 0 }, b{ 0 };
			float maxz{ -INF };

			for (int i = 0; i < sphere_num; ++i) {
				float n;
				float t = spheres[i].hit(ox, oy, &n);
				if (t > maxz) {
					float fscale = n;
					r = spheres[i].r * fscale;
					g = spheres[i].g * fscale;
					b = spheres[i].b * fscale;
					maxz = t;
				}
			}

			ptr[offset * 4 + 0] = static_cast<unsigned char>(r * 255);
			ptr[offset * 4 + 1] = static_cast<unsigned char>(g * 255);
			ptr[offset * 4 + 2] = static_cast<unsigned char>(b * 255);
			ptr[offset * 4 + 3] = 255;
		}
	}

	auto end = std::chrono::steady_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
	*elapsed_time = duration.count() * 1.0e-6;

	return 0;
}
