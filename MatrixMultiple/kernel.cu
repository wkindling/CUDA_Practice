#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#define N1 2
#define N2 3
#define N3 2


using namespace std;


__global__ void multiple(float* A, float* B, float* C)
{
	int ix = threadIdx.x + blockDim.x*blockIdx.x;
	int iy = threadIdx.y + blockDim.y*blockIdx.y;
	int index = ix * N3 + iy;
	C[index] = 0;
	for (int i = 0; i < N2; i++)
	{
		C[index] += A[ix*N2 + i] * B[i*N3 + iy]; 		//C[ix,iy]+=A[ix,i]*B[i,iy]
	}
}

int main()
{
	cudaSetDevice(0);
		
	float* A_host=(float*)malloc(N1*N2*sizeof(float));
	float* B_host=(float*)malloc(N2*N3*sizeof(float));
	float* C_host=(float*)malloc(N1*N3*sizeof(float));

	for (int i = 0; i < N1; i++)
	{
		for (int j = 0; j < N2; j++)
		{
			cin >> A_host[i*N2 + j];
		}
	}	
	
	for (int i = 0; i < N2; i++)
	{
		for (int j = 0; j < N3; j++)
		{
			cin >> B_host[i*N3 + j];
		}
	}

	float* A_dev = NULL;
	float* B_dev = NULL;
	float* C_dev = NULL;
	cudaMalloc((void**)&A_dev, N1*N2 * sizeof(float));
	cudaMalloc((void**)&B_dev, N2*N3 * sizeof(float));
	cudaMalloc((void**)&C_dev, N1*N3 * sizeof(float));

	cudaMemcpy(A_dev, A_host, sizeof(float)*N1*N2, cudaMemcpyHostToDevice);
	cudaMemcpy(B_dev, B_host, sizeof(float)*N2*N3, cudaMemcpyHostToDevice);
	
	int dimx = N1;
	int dimy = N3;

	dim3 block(dimx, dimy);
	dim3 grid(1);

	multiple << <grid, block >> > (A_dev, B_dev, C_dev);
	
	cudaDeviceSynchronize();
	cudaMemcpy(C_host, C_dev, N1*N3 * sizeof(float), cudaMemcpyDeviceToHost);

	cout <<endl;
	for (int i = 0; i < N1; i++)
	{
		for (int j = 0; j < N3; j++)
		{
			cout << C_host[i*N3+j] << " ";
		}
		cout << endl;
	}

	cudaFree(A_dev);
	cudaFree(B_dev);
	cudaFree(C_dev);
	free(A_host);
	free(B_host);
	free(C_host);

	cudaDeviceReset();
	return 0;
}