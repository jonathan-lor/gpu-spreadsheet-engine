#include <cuda_runtime.h>
#include "../include/spreadsheet.hpp"
#include <iostream>

using std::vector;

// kernel to scale table by x factor
__global__ void scaleKernel(double* data, int N, double scale) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if(idx < N) data[idx] *= scale;
} 

void SpreadsheetGrid::applyScaleCPU(double scale) {
	for(size_t i = 0; i < rows; i++) {
		for(size_t j = 0; j < cols; j++) {
			grid[i][j].value *= scale;
		}
	}
}

void SpreadsheetGrid::applyScaleCUDA(double scale) {
	size_t totalCells = rows * cols;
	size_t size = totalCells * sizeof(double);

	// turn grid into 1d array (is ths most efficient?) maybe 2d indexing is better for cpu cache locality?
	vector<double> flat(totalCells);
	for(size_t i = 0; i < rows; i++)  {
		for(size_t j = 0; j < cols; j++) {
			flat[j + i * cols] = grid[i][j].value;
		}
	}	

	// gpu allocation - consider cudaMallocManaged for performance maybe?
	double* d_data;
	cudaMalloc(&d_data, size);
	cudaMemcpy(d_data, flat.data(), size, cudaMemcpyHostToDevice);

	// kernel launch - maybe make threads per block variable?
	int threadsPerBlock = 256;
	int blocks = (totalCells + threadsPerBlock - 1) / threadsPerBlock; // n is totalCells here
	scaleKernel<<<blocks, threadsPerBlock>>>(d_data, totalCells, scale);

	cudaMemcpy(flat.data(), d_data, size, cudaMemcpyDeviceToHost);

	// write back to grid
	for(size_t i = 0; i < rows; i++) {
		for(size_t j = 0; j < cols; j++) {
			grid[i][j].value = flat[j + i * cols];
		}
	}
}
