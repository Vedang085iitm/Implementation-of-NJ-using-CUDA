#include <bits/stdc++.h>
#include <numeric>
#include <chrono>
#include <cuda_runtime.h>
#include <cuda.h>
using namespace std;

using std::cin;
using std::cout;

__global__ void rowsum(int *d_matrix, int *d_sums, int n)
{
    long block_Idx = blockIdx.x + (gridDim.x) * blockIdx.y + (gridDim.y * gridDim.x) * blockIdx.z;
    long thread_Idx = threadIdx.x + (blockDim.x) * threadIdx.y + (blockDim.y * blockDim.x) * threadIdx.z;
    long block_Capacity = blockDim.x * blockDim.y * blockDim.z;
    long i = block_Idx * block_Capacity + thread_Idx;

    if (i < n)
    {
        d_sums[i] = 0; // Initialize the sum to 0
        for (int j = 0; j < n; ++j)
        {
            atomicAdd(&d_sums[i], d_matrix[i * n + j]);
        }
    }
}

__global__ void neighborJoiningMatrix(int *d_matrix, int *d_rowSums, int *d_njMatrix, int n)
{
    long block_Idx = blockIdx.x + (gridDim.x) * blockIdx.y + (gridDim.y * gridDim.x) * blockIdx.z;
    long thread_Idx = threadIdx.x + (blockDim.x) * threadIdx.y + (blockDim.y * blockDim.x) * threadIdx.z;
    long block_Capacity = blockDim.x * blockDim.y * blockDim.z;
    long arr_Idx = block_Idx * block_Capacity + thread_Idx;

    if (arr_Idx < n * n)
    {
        int i = arr_Idx / n;
        int j = arr_Idx % n;

        if (i == j)
        {
            d_njMatrix[arr_Idx] = 0;
        }
        else
        {
            d_njMatrix[arr_Idx] = (n - 2) * d_matrix[arr_Idx] - d_rowSums[i] - d_rowSums[j];
        }
    }
}

int main()
{
    int n;
    cout << "Enter the size of the matrix: " << endl;
    cin >> n;
    cout << "Enter the matrix: " << endl;
    int *matrix = new int[n * n];
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            cin >> matrix[i * n + j];
        }
    }
    int *sums = new int[n];
    int *d_matrix;
    int *d_sums;
    cudaMalloc(&d_matrix, n * n * sizeof(int));
    cudaMalloc(&d_sums, n * sizeof(int));
    cudaMemcpy(d_matrix, matrix, n * n * sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockSize(512);
    dim3 numBlocks((n + blockSize.x - 1) / blockSize.x);

    rowsum<<<numBlocks, blockSize>>>(d_matrix, d_sums, n);
    cudaDeviceSynchronize();
    cout << "Row sums: " << endl;
    cudaMemcpy(sums, d_sums, n * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; ++i)
    {
        cout << sums[i] << " ";
    }
    cout << endl;

    int *njMatrix = new int[n * n];
    int *d_njMatrix;
    cudaMalloc(&d_njMatrix, n * n * sizeof(int));
    neighborJoiningMatrix<<<numBlocks, blockSize>>>(d_matrix, d_sums, d_njMatrix, n);
    cudaDeviceSynchronize();
    cudaMemcpy(njMatrix, d_njMatrix, n * n * sizeof(int), cudaMemcpyDeviceToHost);
    cout << "Neighbor joining matrix: " << endl;
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            cout << njMatrix[i * n + j] << " ";
        }
        cout << endl;
    }

    delete[] matrix;
    delete[] sums;
    cudaFree(d_matrix);
    cudaFree(d_sums);
    cudaFree(d_njMatrix);
    
    return 0;
}