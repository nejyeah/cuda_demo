#include <iostream>
#include "funset.hpp"

int main()
{
	int ret = test_long_vector_add();

	if (ret == 0) fprintf(stderr, "***** test success *****\n");
	else fprintf(stderr, "===== test fail =====\n");

	return 0;
}
