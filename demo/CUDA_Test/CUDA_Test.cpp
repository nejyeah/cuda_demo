#include <iostream>
#include "funset.hpp"

int main()
{
	int ret = test_calculate_histogram();

	if (ret == 0) fprintf(stderr, "***** test success *****\n");
	else fprintf(stderr, "===== test fail =====\n");

	return 0;
}
