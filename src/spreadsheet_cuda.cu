#include <cuda_runtime.h>
#include "../include/spreadsheet.hpp"
#include <iostream>

using std::vector;


// SCALE OPERATIONS
// kernel to scale table by x factor
__global__ void scaleKernel(double* data, int N, double scale) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if(idx < N) data[idx] *= scale;
} 

void SpreadsheetGrid::applyScaleCPU(double scale) {
	for (size_t i = 0; i < rows; ++i) {
		for (size_t j = 0; j < cols; ++j) {
			grid[i * cols + j].value *= scale;
		}
	}
}


void SpreadsheetGrid::applyScaleCUDA(double scale) {
	size_t totalCells = rows * cols;
	size_t size = totalCells * sizeof(double);

	// Flatten values for GPU
	vector<double> flat(totalCells);
	for (size_t idx = 0; idx < totalCells; ++idx) {
		flat[idx] = grid[idx].value;
	}

	// Allocate GPU memory
	double* d_data;
	cudaMalloc(&d_data, size);
	cudaMemcpy(d_data, flat.data(), size, cudaMemcpyHostToDevice);

	// Launch kernel
	int threadsPerBlock = 256;
	int blocks = (totalCells + threadsPerBlock - 1) / threadsPerBlock;
	scaleKernel<<<blocks, threadsPerBlock>>>(d_data, totalCells, scale);

	// Copy result back
	cudaMemcpy(flat.data(), d_data, size, cudaMemcpyDeviceToHost);
	cudaFree(d_data);

	// Update grid
	for (size_t idx = 0; idx < totalCells; ++idx) {
		grid[idx].value = flat[idx];
	}
}

// ROW SUM OPERATIONS


__global__ void rowSumKernelShared(const double* __restrict__ data, double* rowSums, int rows, int cols) {
    extern __shared__ double shared[];

    int row = blockIdx.x; // one block per row
    int tid = threadIdx.x;
    int idx = row * cols + tid;

    // Load data into shared memory
    shared[tid] = (tid < cols) ? data[idx] : 0.0;
    __syncthreads();

    // Parallel reduction within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && tid + stride < cols) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    // Write result to rowSums
    if (tid == 0) {
        rowSums[row] = shared[0];
    }
}

vector<double> SpreadsheetGrid::rowSumGPU() const {
    size_t totalCells = rows * cols;

    // Allocate unified memory
    double* data;
    double* result;
    cudaMallocManaged(&data, totalCells * sizeof(double));
    cudaMallocManaged(&result, rows * sizeof(double));

    // Copy grid values into data
    for (size_t i = 0; i < totalCells; ++i) {
        data[i] = grid[i].value;
    }

    // Launch kernel: one block per row, threads = next power of 2 >= cols
    int threads = 1;
    while (threads < cols) threads <<= 1;

    rowSumKernelShared<<<rows, threads, threads * sizeof(double)>>>(data, result, rows, cols);
    cudaDeviceSynchronize();

    // Copy back results
    std::vector<double> rowSums(rows);
    for (size_t i = 0; i < rows; ++i) {
        rowSums[i] = result[i];
    }

    cudaFree(data);
    cudaFree(result);
    return rowSums;
}


vector<double> SpreadsheetGrid::rowSumCPU() const {
    vector<double> rowSums(rows, 0.0);

    for (size_t i = 0; i < rows; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < cols; ++j) {
            sum += grid[i * cols + j].value;
        }
        rowSums[i] = sum;
    }

    return rowSums;
}


// NORMALIZE

__global__ void normalizeRowsKernel(double* data, const double* rowSums, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx >= total) return;

    int row = idx / cols;
    double sum = rowSums[row];
    data[idx] = sum > 1e-8 ? data[idx] / sum : 0.0;
}

void SpreadsheetGrid::normalizeRowsGPU() {
    size_t total = rows * cols;
    size_t dataSize = total * sizeof(double);

    double* data;
    cudaMallocManaged(&data, dataSize);
    for (size_t i = 0; i < total; ++i)
        data[i] = grid[i].value;

    auto sums = rowSumGPU(); // reuse existing
    double* d_sums;
    cudaMallocManaged(&d_sums, rows * sizeof(double));
    std::copy(sums.begin(), sums.end(), d_sums);

    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    normalizeRowsKernel<<<blocks, threads>>>(data, d_sums, rows, cols);
    cudaDeviceSynchronize();

    for (size_t i = 0; i < total; ++i)
        grid[i].value = data[i];

    cudaFree(data);
    cudaFree(d_sums);
}

void SpreadsheetGrid::normalizeRowsCPU() {
    for (size_t i = 0; i < rows; ++i) {
        double sum = 0.0;

        // Compute sum of row i
        for (size_t j = 0; j < cols; ++j) {
            sum += grid[i * cols + j].value;
        }

        // Normalize row i
        if (sum > 1e-8) { // avoid division by zero
            for (size_t j = 0; j < cols; ++j) {
                grid[i * cols + j].value /= sum;
            }
        } else {
            // Optional: set all values to zero
            for (size_t j = 0; j < cols; ++j) {
                grid[i * cols + j].value = 0.0;
            }
        }
    }
}

