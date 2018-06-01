#include "funset.hpp"
#include <vector>
#include <chrono>
#include "common.hpp"

int layer_prior_vbox_cpu(float* dst, int length, const std::vector<float>& vec1, const std::vector<float>& vec2,
	const std::vector<float>& vec3, float* elapsed_time)
{
	TIME_START_CPU

	int layer_width = (int)vec1[0];
	int layer_height = (int)vec1[1];
	int image_width = (int)vec1[2];
	int image_height = (int)vec1[3];
	float offset = vec1[4];
	float step = vec1[5];
	int num_priors = (int)vec1[6];
	float width = vec1[7];

	CHECK(length == layer_width * layer_height * num_priors * 4 * 2);
	CHECK(vec1.size() == 8);
	CHECK(vec2.size() == num_priors);
	CHECK(vec3.size() == 4);

	float* top_data = dst;
	int idx = 0;

	for (int h = 0; h < layer_height; ++h) {
		for (int w = 0; w < layer_width; ++w) {
			float center_x = (w + offset) * step;
			float center_y = (h + offset) * step;

			for (int s = 0; s < num_priors; ++s) {
				float box_width = width;
				float box_height = vec2[s];

				top_data[idx++] = (center_x - box_width / 2.) / image_width;
				top_data[idx++] = (center_y - box_height / 2.) / image_height;
				top_data[idx++] = (center_x + box_width / 2.) / image_width;
				top_data[idx++] = (center_y + box_height / 2.) / image_height;
			}
		}
	}

	int len = layer_width * layer_height * num_priors;
	for (int i = 0; i < len; ++i) {
		for (int j = 0; j < 4; ++j) {
			top_data[idx++] = vec3[j];
		}
	}

	TIME_END_CPU

	return 0;
}
