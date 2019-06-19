CFLAGS=-Wall -Wextra -pedantic -fopenmp `libpng-config --ldflags`
CC=g++
LDFLAGS=-lm -lmpi

all:
	$(CC) mandelbrot.cpp $(CFLAGS) $(LDFLAGS) -g -O2 -o dmbrot

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