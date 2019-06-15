#ifndef MANDEL_H 
#define MANDEL_H

#include <complex.h>

void prepare (int *res_matrix, const int w, const int h, std::complex<float> c0, const float del_y, const float del_x, const int threads);

#endif 
