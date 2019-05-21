//kernelVBO.cu

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>


__global__ void kernel(float4 *pos, unsigned int width, unsigned int height, float time)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	float u = x / (float)width;
	float v = y / (float)height;

	u = u * 2.0f - 1.0f;
	v = v * 2.0f - 1.0f;

	float freq = 4.0f;

	float w = sinf(u*freq + time) * cosf(v*freq + time) * 0.5f;

	pos[y*width + x] = make_float4(u, w, v, 1.0f);

}

extern "C" void launch_kernel(float4 *pos, unsigned int mesh_width, unsigned int mesh_height, float time)
{
	dim3 block(8, 8, 1);
	dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
	kernel << <grid, block >> > (pos, mesh_width, mesh_height, time);

	cudaThreadSynchronize();
}