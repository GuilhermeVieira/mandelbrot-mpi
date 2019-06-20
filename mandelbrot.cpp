#include <iostream> //cout, cerr
#include <complex>
#include <string> //stof
//#include"mandel.cuh"
#include <png++/png.hpp>
#include <mpi.h>

// Defines a task of mandelbrot set calculation over some pixels in the image. 
// The work must be done in the interval [start, end[
struct task {
    int start, end;
};

// These tags are used to identify the messages. TCP does not guarantee that
// two sent messages sent will be received in the correct order.
#define TAG_TASK    0
#define TAG_RESULT  1

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
        for (int i = 0; i < h * w; ++i) {
            del.real(del_x * (i % w));
            del.imag(del_y * (i / w));
            res[i] = get_inter(c0 + del);
        }  
    return;
}

void work(int *result, int start, const int w, int work_size, std::complex<float> c0, const float del_y, const float del_x) {
    std::complex<float> del(0, 0);
    for (int i = 0; i < work_size; ++i) {
        del.real(del_x * ((start + i) % w));
        del.imag(del_y * ((start + i) / w));
        result[i] = get_inter(c0 + del);
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

// Pseudo-master process. It will send the messages with the tasks, do its work and retrieve the
// results from the slaves.
static void master_fill_matrix(int *res, const int w, const int h, std::complex<float> c0, const float del_y, const float del_x, const int threads, int num_slaves)
{
    struct task *tasks = new struct task[num_slaves];
    int err = 0;
    int threads_with_one_more_work = h*w % num_slaves;
    // Distribute processes between processes.
    for (int i = 0; i < num_slaves; ++i) {
        int work_size = h*w / num_slaves;
        if (i < threads_with_one_more_work) {
            work_size += 1;
        }
        tasks[i].start = i * work_size;
        tasks[i].end = (i + 1) * work_size;
        // If it's not the master process, we send the task to the respective process
        if (i != 0) {
            err |= MPI_Send(&tasks[i],   // Buffer to send
                    sizeof(struct task), // How many elements. Note that there is a hack here.
                    MPI_CHAR,            // Type of element. Note that there is a hack here.
                    i,                   // For which process
                    TAG_TASK,            // Tag of the message. Remember that TCP do not guarantee order.
                    MPI_COMM_WORLD       // In the context of the entire world.
            );
        }
    }
    // Master processing work
    work(res, 0, w, tasks[0].end, c0, del_y, del_x); 
    // Gather the results
    for (int i = 1; i < num_slaves; ++i) {
        int work_size = tasks[i].end - tasks[i].start;
        int *result = new int[work_size];
        err |= MPI_Recv(result,   // Buffer to write to. You must ensure that the message fits here.
                work_size,        // How many elements.
                MPI_INT,          // Type.
                i,                // From which process
                TAG_RESULT,       // Tag of message. Again, remember that TCP do not guarantee order.
                MPI_COMM_WORLD,   // In the context of the entire world.
                MPI_STATUS_IGNORE // Ignore the status return.
        );
        if (err) {
            DIE("There was an MPI error in the master.\n");
        }
        // Process result
        for (int j = 0; j < work_size; ++j) {
            res[(i * work_size) + j] = result[j];
        }
        delete[] result;
    }
    delete[] tasks;
    return;
}

void slave_fill_matrix(const int w, std::complex<float> c0, const float del_y, const float del_x) {
    struct task recv;
    int err = 0;
    int *result;
    int work_size;
    // Wait for task
    err |= MPI_Recv(&recv,
            sizeof(struct task),
            MPI_CHAR,
            0,
            TAG_TASK,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE
    );
    // Mandelbrot work
    work_size = recv.end - recv.start;
    result = new int[work_size];
    work (result, recv.start, w, work_size, c0, del_y, del_x);
    // Send vector
    err |= MPI_Send(result,
            work_size,
            MPI_INT,
            0,
            TAG_RESULT,
            MPI_COMM_WORLD
    );
    delete[] result;
    if (err) {
        DIE("There was an MPI error in one slave.\n");  
    }
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
    int world_size, taskid;
    int err = 0;
    // Send attributes to all processes, and initialize MPI.
    err |= MPI_Init(&argc, &argv);
    // Get how many processes there are in the world MPI_COMM_WORLD
    err |= MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    // Which process am I?
    err |= MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    if (err) {
        DIE("There was an MPI initialization error.\n");
    }
    // Pseudo-master process
    if (taskid == 0) {
        int *res = new int[w*h];
        int *res_test = new int[w*h]; // FOR TESTING PURPOSES
        if (comp_flag.compare("CPU") == 0) {
            master_fill_matrix(res, w, h, c0, del_y, del_x, num_threads, world_size);
            fill_matrix(res_test, w, h, c0, del_y, del_x, num_threads);
        }
        else if (comp_flag.compare("GPU") == 0) {
            //prepare(res, w, h, c0, del_y, del_x, num_threads);
            DIE("Comprou GPU AMD!¯\\_(ツ)_/¯");
        } 
        else {
            DIE("Neither CPU nor GPU selected.\n");
        }
        // FOR TEST PURPOSES
        for (int i = 0; i < w*h; ++i)
            if (res[i] != res_test[i])
                DIE("DEU MUITO RUIM!!! =(")
        create_picture(res, file_name, w, h);
        delete[] res;
        delete[] res_test;
    }
    // Slave processes
    else {
        // USE OF THREADS ON CPU??
        slave_fill_matrix(w, c0, del_y, del_x);
    }
    MPI_Finalize();

    return 0;
}