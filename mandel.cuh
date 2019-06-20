#ifndef MANDEL_H 
#define MANDEL_H

#include <thrust/complex.h>

__host__ void prepare (int *res_matrix, const int start, const int w, const int work_size, thrust::complex<float> c0, const float del_y, const float del_x, const int threads);

#endif 
