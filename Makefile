CFLAGS=-Wall -Wextra -pedantic -fopenmp -I/usr/local/cuda/inlcude -L/usr/local/cuda/lib64 -lcuda -lcudart -lpthread `libpng-config --ldflags`
NVCC=nvcc
CC=g++

all: mandel.o
	$(CC) mandelbrot.cpp mandel.o $(CFLAGS) -g -O2 -o mbrot

mandel.o: mandel.cu
	$(NVCC) mandel.cu -c

.PHONY: cpu
cpu:
	./mbrot -2.0 -2.0 2.9 2.0 1920 1080 CPU 4 cpu.png

.PHONY: gpu
gpu:
	./mbrot -2.0 -2.0 2.9 2.0 1920 1080 GPU 32 gpu.png


.PHONY: clean
clean:
	rm *.o
	rm mbrot
	rm *.png
