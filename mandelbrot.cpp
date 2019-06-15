#include <iostream> //cout, cerr
#include <complex>
#include <string> //stof

#include "mandel.cuh"

#include <png++/png.hpp>

#define INTER_LIMIT 255

#define DIE(...) { \
        std::cerr << __VA_ARGS__; \
        std::exit (EXIT_FAILURE); \
}


int get_inter(std::complex<float> c) {
    std::complex<float> z(0.0, 0.0);
    int i;
    for (i = 0; i < INTER_LIMIT; i++) {
        if (std::abs(z) > 2) {
            break;
        }
        z = std::pow(z, 2) + c;
    }
    return i;
}

void fill_matrix(int *res, const int w, const int h, std::complex<float> c0, const float del_y, const float del_x, const int threads){
    std::complex<float> del(0, 0);
    #pragma omp parallel for num_threads(threads)
        for (int k = 0; k < h*w; ++k) {
            del.real(del_x * (k%w));
            del.imag(del_y * (k/w));
            res[k] = get_inter(c0 + del);
        }  
    return;
}

void create_picture(int *matrix, const std::string file_name, const int w, const int  h) {
    png::image< png::rgb_pixel > image(w, h);
    for (png::uint_32 i = 0; i < image.get_height(); ++i) {
        for (png::uint_32 j = 0; j < image.get_width(); ++j) {
            image[i][j] = png::rgb_pixel(255 - matrix[i*w +j], matrix[i*w +j], 255 -  matrix[i*w +j]);
        }
    }
    image.write(file_name);
    return;
}

int main(int argc, char** argv) {
    if (argc != 10) {
        DIE("Wrong number of arguments\n");
    }
    std::complex<float> c0(std::stof(argv[1]), std::stof(argv[2]));
    std::complex<float> c1(std::stof(argv[3]), std::stof(argv[4]));
    const int w = std::stoi(argv[5]);
    const int h = std::stoi(argv[6]);
    const std::string comp_flag = argv[7];
    const int num_threads = std::stoi(argv[8]);
    const std::string file_name = argv[9];
    
    const float del_x = (c1.real() - c0.real()) / (w - 1);
    const float del_y = (c1.imag() - c0.imag()) / (h - 1);
    int *res = new int[w*h];
    
    if (comp_flag.compare("CPU") == 0) {
        fill_matrix(res, w, h, c0, del_y, del_x, num_threads);
    }
    else if (comp_flag.compare("GPU") == 0) {
        prepare(res, w, h, c0, del_y, del_x, num_threads);
    } 
    else {
        DIE("Neither CPU nor GPU selected.\n");
    }
    create_picture(res, file_name, w, h);
    
    delete[] res;

    return 0;
}