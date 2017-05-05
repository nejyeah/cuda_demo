#include <iostream>
#include "simple.hpp"

int main()
{
	int ret = test_vectorAdd();

	if (ret == 0) fprintf(stderr, "***** test success *****\n");
	else fprintf(stderr, "===== test fail =====\n");

	return 0;
}
