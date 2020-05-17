obj : test
PYTHON: obj

CC = gcc
CPP = g++

FLAGS = --std=c++11 -mavx -mavx2 -O3

test: gemm_calculator_int16.o test.o
	g++ -o test gemm_calculator_int16.o test.o

gemm_calculator_int16.o: gemm_calculator_int16.cc gemm_calculator_int16.h
	g++ $(FLAGS) -c gemm_calculator_int16.cc

test.o: test.cc
	g++ $(FLAGS) -c test.cc

PYTHON: clean
clean:
	rm -f test *.o

