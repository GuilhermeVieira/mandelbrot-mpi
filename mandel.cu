#include <iostream>
#include "mandel.cuh"
#define INTER_LIMIT 255

__device__ int get_inter (thrust::complex<float> c) {
    int i;
    thrust::complex<float> z(0.0, 0.0);

    for (i = 0; i < INTER_LIMIT; ++i) {
        if (thrust::abs(z) > 2 ) {
            break;
        }
        z = thrust::pow(z, 2) + c;
    }
    return i;
}

__global__ void fill_matrix (int *res, const int start, const int w, const int work_size, thrust::complex<float> c0, const float del_x, const float del_y, const int threads, const int blocks, const int offset) {
    thrust::complex<float> del(0, 0);
    unsigned int k = threadIdx.x + blockIdx.x*threads + blocks*threads*offset;
    if (k >= work_size)
        return;
    del.real(del_x * ((start + k) % w));
    del.imag(del_y * ((start + k) / w));
    res[k] = get_inter(c0 + del);
    return;
}

__host__ void prepare (int *res_matrix, const int start, const int w, const int work_size, thrust::complex<float> c0, const float del_x, const float del_y, const int threads) {
    int *d_res_matrix; 
    int *d_w; 
    int *d_start;
    int *d_work_size;
    float *d_del_x;
    float *d_del_y;
    thrust::complex<float> *d_c0;  
    
    cudaSetDevice(0);

    if (cudaSuccess != cudaMallocManaged((void **) &d_res_matrix, sizeof(int)*work_size)) {
        std::cerr << "Could not allocate memory";
        exit(EXIT_FAILURE);
    }
    if (cudaSuccess != cudaMallocManaged((void **) &d_start, sizeof(int))) {
    	std::cerr << "Could not allocate memory";
	exit(EXIT_FAILURE);
    }
    if (cudaSuccess != cudaMallocManaged((void **) &d_w, sizeof(int))) {
        std::cerr << "Could not allocate memory";
        exit(EXIT_FAILURE);
    }
    if (cudaSuccess != cudaMallocManaged((void **) &d_work_size, sizeof(int))) {
    	std::cerr << "Could not allocate memory";
	exit(EXIT_FAILURE);
    }
    if (cudaSuccess != cudaMallocManaged((void **) &d_c0, sizeof(thrust::complex<float>)) ) {
        std::cerr << "Could not allocate memory";
        exit(EXIT_FAILURE);
    }
    if (cudaSuccess != cudaMallocManaged((void **) &d_del_y, sizeof(float))) {
        std::cerr << "Could not allocate memory";
        exit(EXIT_FAILURE);
    }
    if (cudaSuccess != cudaMallocManaged((void **) &d_del_x, sizeof(float))) {
        std::cerr << "Could not allocate memory";
        exit(EXIT_FAILURE);
    }
    if (cudaSuccess != cudaMemcpy(d_start, &start, sizeof(int), cudaMemcpyHostToDevice)) {
    	std::cerr << "Could not copy memory";
	exit(EXIT_FAILURE);
    }
    if (cudaSuccess != cudaMemcpy(d_w, &w, sizeof(int), cudaMemcpyHostToDevice)) {
        std::cerr << "Could not copy memory";
        exit(EXIT_FAILURE);
    }
    if (cudaSuccess != cudaMemcpy(d_work_size, &work_size, sizeof(int), cudaMemcpyHostToDevice)) {
    	std::cerr << "Could not copy memory";
	exit(EXIT_FAILURE);
    }
    if (cudaSuccess != cudaMemcpy(d_c0, &c0, sizeof(thrust::complex<float>), cudaMemcpyHostToDevice)) {
        std::cerr << "Could not copy memory";
        exit(EXIT_FAILURE);
    }
    if (cudaSuccess != cudaMemcpy(d_del_y, &del_y, sizeof(float), cudaMemcpyHostToDevice)) {
        std::cerr << "Could not copy memory";
        exit(EXIT_FAILURE);
    }
    if (cudaSuccess != cudaMemcpy(d_del_x, &del_x, sizeof(float), cudaMemcpyHostToDevice)) {
        std::cerr << "Could not copy memory";
        exit(EXIT_FAILURE);
    }
    
    int block = 1024;
    int max = (work_size/ (threads*block)) + 1;
    for (int i = 0; i < max; ++i) {
        fill_matrix<<<block, threads>>> (d_res_matrix, *d_start, *d_w, *d_work_size, *d_c0, *d_del_x, *d_del_y, threads, block, i);
        cudaDeviceSynchronize();
    }
    
    if (cudaSuccess != cudaMemcpy(res_matrix, d_res_matrix, sizeof(int)*work_size, cudaMemcpyDeviceToHost)) {
        std::cerr << "Could not copy memory";
        exit(EXIT_FAILURE);
    }
    return;
}
