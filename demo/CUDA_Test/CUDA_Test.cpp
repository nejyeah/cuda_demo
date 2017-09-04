#include <cstdio>
#include <iostream>
#include "funset.hpp"

int main()
{
	int ret = test_layer_reverse();

	if (ret == 0) fprintf(stderr, "***** test success *****\n");
	else fprintf(stderr, "===== test fail =====\n");

	return 0;
}
