CC = gcc
ARCH=sm_35

main : driver1.cu
	nvcc -O3 -std=c++11 -ccbin=$(CC)  driver1.cu -arch=$(ARCH) -o main 

clean :
	rm -f main


