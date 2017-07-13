#include <iostream>
#include "simple.hpp"

int main()
{
	int ret = test_vector_add();

	if (ret == 0) fprintf(stderr, "***** test success *****\n");
	else fprintf(stderr, "===== test fail =====\n");

	return 0;
}
