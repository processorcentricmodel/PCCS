#include<stdio.h>
#include <malloc.h>
#include <stdlib.h>
#define N 1000
 
void MatrixMul(int *A, int *B, int *C, int Width) {
    int i, j, k;
	for(i=0; i<Width; i++)
		for(j=0; j<Width; j++){
			int s=0;
			for(k=0; k<Width; k++) 
				s+=A[i*Width+k]*B[k*Width+j];
			C[i*Width+j]=s;
		}
}
 
#define TILE_WIDTH 16
 
__global__ void KernelMatrixMul(int* Md, int* Nd, int* Pd, int Width)
{
    int x = threadIdx.x+blockIdx.x*blockDim.x;
    int y = threadIdx.y+blockIdx.y*blockDim.y;
 
	int Pvalue = 0;
	for (int k = 0; k < Width; ++k)
		Pvalue+=Md[y * Width + k]*Nd[k * Width + x];
	Pd[y*Width + x] = Pvalue;
 
}
 
int main(){
	int *A=(int*)malloc(N*N*sizeof(int));
    int *B=(int*)malloc(N*N*sizeof(int));
    int *C=(int*)malloc(N*N*sizeof(int));
	int i;
	for(i=0;i<N*N;i++){
		A[i] = 1;
		B[i] = 2;
	}
 
	//MatrixMul(A,B,C,N);
	
	int *dev_A,*dev_B,*dev_C;
	dim3 dimGrid(N/TILE_WIDTH,N/TILE_WIDTH);
	dim3 dimBlock(TILE_WIDTH,TILE_WIDTH);
	cudaMalloc((void**)&dev_A,N*N*sizeof(int));
	cudaMalloc((void**)&dev_B,N*N*sizeof(int));
	cudaMalloc((void**)&dev_C,N*N*sizeof(int));
	cudaMemcpy(dev_A,A,N*N*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_B,B,N*N*sizeof(int),cudaMemcpyHostToDevice);
	KernelMatrixMul<<<dimGrid,dimBlock>>>(dev_A,dev_B,dev_C,N);
	cudaThreadSynchronize();
	cudaMemcpy(C,dev_C,N*N*sizeof(int),cudaMemcpyDeviceToHost);
	cudaFree(dev_A);
	cudaFree(dev_B);
	cudaFree(dev_C);
	
	int m,n;
	for(m=0;m<N;m++){
		for(n=0;n<N;n++)
			printf("C[%d][%d] = %d\n",m,n,C[m*N+n]);
	}
 
	return 0;
}
