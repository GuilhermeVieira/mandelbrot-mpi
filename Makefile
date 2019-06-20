CFLAGS=-Wall -Wextra -pedantic -fopenmp -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcuda -lcudart -lpthread `libpng-config --ldflags`
NVCC=nvcc
CC=mpic++

all: mandel.o
	$(CC) mandelbrot.cpp $(CFLAGS) -g -O2 -o dmbrot

mandel.o: mandel.cu
	$(NVCC) mandel.cu -c

.PHONY: cpu
cpu:
	./dmbrot -2.0 -2.0 2.9 2.0 1920 1080 CPU 4 cpu.png

.PHONY: gpu
gpu:
	./dmbrot -2.0 -2.0 2.9 2.0 1920 1080 GPU 32 gpu.png

.PHONY: clean
clean:
	rm *.o
	rm dmbrot
	rm *.png
