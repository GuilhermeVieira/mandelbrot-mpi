CFLAGS=-Wall -Wextra -pedantic -fopenmp -I/usr/local/cuda/inlcude -L/usr/local/cuda/lib64 -lcuda -lcudart -lpthread `libpng-config --ldflags`
NVCC=nvcc
CC=g++
LDFLAGS=-lm -lmpi
INCLUDES=-I/usr/lib/x86_64-linux-gnu/openmpi/include/

all: mandel.o
	$(CC) mandelbrot.cpp mandel.o $(CFLAGS) $(LDFLAGS) -g -O2 -o dmbrot

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
