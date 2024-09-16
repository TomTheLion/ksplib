#include <iostream>
#include "b.h"

void print_from_a()
{
	std::cout << "hello from a\n";
}

void print_from_b_from_a()
{
	print_from_b();
}