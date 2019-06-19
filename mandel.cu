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

__global__ void fill_matrix (int *res, const int w, const int h, thrust::complex<float> c0, const float del_y, const float del_x, const int threads, const int blocks, const int offset) {
    thrust::complex<float> del(0, 0);
    unsigned int k = threadIdx.x + blockIdx.x*threads + blocks*threads*offset;
    if (k >= w*h)
        return;
    del.real(del_x * (k%w));
    del.imag(del_y * (k/w));
    res[k] = get_inter(c0 + del);
    return;
}

__host__ void prepare (int *res_matrix, const int w, const int h, thrust::complex<float> c0, const float del_y, const float del_x, const int threads) {
    int *d_res_matrix; 
    int *d_w; 
    int *d_h;
    thrust::complex<float> *d_c0; 
    float *d_del_y; 
    float *d_del_x; 
    
    cudaSetDevice(0);

    if (cudaSuccess != cudaMallocManaged((void **) &d_res_matrix, sizeof(int)*w*h)) {
        std::cerr << "Could not allocate memory";
        exit(EXIT_FAILURE);
    }
    if (cudaSuccess != cudaMallocManaged((void **) &d_w, sizeof(int))) {
        std::cerr << "Could not allocate memory";
        exit(EXIT_FAILURE);
    }
    if (cudaSuccess != cudaMallocManaged((void **) &d_h, sizeof(int))) {
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

    if (cudaSuccess != cudaMemcpy(d_w, &w, sizeof(int), cudaMemcpyHostToDevice)) {
        std::cerr << "Could not copy memory";
        exit(EXIT_FAILURE);
    }
    if (cudaSuccess != cudaMemcpy(d_h, &h, sizeof(int), cudaMemcpyHostToDevice)) {
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
    int max = ((w*h) / (threads*block)) + 1;
    for (int i = 0; i < max; ++i) {
        fill_matrix<<<block, threads>>> (d_res_matrix, *d_w, *d_h, *d_c0, *d_del_y, *d_del_x, threads, block, i);
        cudaDeviceSynchronize();
    }
    
    if (cudaSuccess != cudaMemcpy(res_matrix, d_res_matrix, sizeof(int)*w*h, cudaMemcpyDeviceToHost)) {
        std::cerr << "Could not copy memory";
        exit(EXIT_FAILURE);
    }
    return;
}
