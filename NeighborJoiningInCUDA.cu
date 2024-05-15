#include <vector>
#include <iostream>
#include <numeric>
#include <chrono>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <fstream>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
# define ll long long

using namespace std;

using std::cin;
using std::cout;

__global__ void getsum(int *rowsums, int *matrix, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * n) {
        int row = idx / n;
        int col = idx % n;
        if (row <= col) {
            atomicAdd(&rowsums[row], matrix[idx]);
            if (row != col) {
                atomicAdd(&rowsums[col], matrix[idx]);
            }
        }
    }
}


__global__ void create_nj(int *matrix, int *rowsums, int *njMat, int n) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n * n) {
        int i = idx / n;
        int j = idx % n;
        int value = matrix[idx];
        if (i == j) {
            njMat[idx] = 0;
        } else {
            int nj_value = (n - 2) * value - rowsums[i] - rowsums[j];
            njMat[idx] = nj_value;
        }
    }
}

__global__ void findMin(int * njMat , int * min , int n){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < n*n){
        atomicMin(min , njMat[idx]);
    } 
}

__global__ void getMinidx(int * njMat , int * min_idx ,  int minVal , int n){
    int idx = blockDim.x *blockIdx.x + threadIdx.x;
    if(idx < n*n){
        if(njMat[idx] == minVal){
            min_idx[0] = idx;
        }
    }
}

__global__ void makeNew(int * old_matrix , int * new_matrix , int i , int j , int n){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < n*n){
        int row = idx / n;
        int col = idx % n;
        if(row==i || row==j || col==i || col==j) return;
        if(i > j){
            int tmp = j;
            j = i;
            i = tmp;
        }
        if(row < i) row++;
        else if(row > j) row--;
        if(col < i) col++;
        else if(col > j) col--;
        new_matrix[(n-1)*row + col] = old_matrix[idx];
    }
}

__global__ void makeMerge(int * new_matrix , int * old_matrix , int * map , int i , int j , int n){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < n - 1){
        if(idx == 0){
            new_matrix[0] = 0;
        }
        else{
            int actual_idx = map[idx];
            new_matrix[idx] = (old_matrix[n * i  + actual_idx] + old_matrix[ n * j + actual_idx] - old_matrix[n * i  + j]) / 2;
            new_matrix[(n-1)*idx] = new_matrix[idx];
        }
    }
}

pair<int, int> calculateLimbLengths(int * matrix, int i, int j, int delta , int n) {
    int limbLengthI = (matrix[n*i +j] + delta) / 2;
    int limbLengthJ = (matrix[n*i +j] - delta) / 2;
    return make_pair(limbLengthI, limbLengthJ);
}

__global__ void copy_mat(int * to , int * from , int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n*n){
        int row = idx / n;
        int col = idx % n;
        to[row*n + col] = from[row*n + col];
    }
}

int main(int argc, char **argv) {
    int n;
    cin >> n;
    int *matrix = new int[n * n];
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            cin >> matrix[i * n + j];
        }
    }

    int top = n;
    vector<int> prev(n, 0);
    vector<int> next(n, 0);
    vector<vector<int>> tree(n);
    vector<vector<int>> edgeWeights(1e4, vector<int>(1e4, 0));
    int *new_mat = new int[n * n];
    int *rowsums = new int[n];
    int *njMatrix = new int[n * n];
    int *d_old_mat;
    int *d_new_mat;
    int *d_rowsums;
    int *d_njMatrix;

    // CUDA streams creation
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    int minVal, minIdx, minIdxI, minIdxJ, delta;
    minVal = 1e9;
    int *d_minVal, *d_minIdx;

    cudaMalloc(&d_njMatrix, n * n * sizeof(int));
    cudaMalloc(&d_old_mat, n * n * sizeof(int));
    cudaMalloc(&d_new_mat, n * n * sizeof(int));
    cudaMalloc(&d_rowsums, n * sizeof(int));
    cudaMalloc(&d_minVal, sizeof(int));
    cudaMalloc(&d_minIdx, sizeof(int));

    cudaMemcpy(d_minVal, &minVal, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_old_mat, matrix, n * n * sizeof(int), cudaMemcpyHostToDevice);

    for (int i = 0; i < n; i++) prev[i] = i;

    int num_blocks;
    auto start = chrono::high_resolution_clock::now();

    while (n - 2) {

        num_blocks = (n * n + 1023) / 1024;
        cudaMemsetAsync(d_rowsums, 0, n * sizeof(int), stream1);
        getsum<<<num_blocks, 1024, 0, stream1>>>(d_rowsums, d_old_mat, n);
        cudaStreamSynchronize(stream1);

        create_nj<<<num_blocks, 1024, 0, stream1>>>(d_old_mat, d_rowsums, d_njMatrix, n);

        minVal = 1e9;
        cudaMemcpyAsync(d_minVal, &minVal, sizeof(int), cudaMemcpyHostToDevice, stream1);
        findMin<<<num_blocks, 1024, 0, stream1>>>(d_njMatrix, d_minVal, n);
        cudaMemcpyAsync(&minVal, d_minVal, sizeof(int), cudaMemcpyDeviceToHost, stream1);
        cudaStreamSynchronize(stream1);

        getMinidx<<<num_blocks, 1024, 0, stream2>>>(d_njMatrix, d_minIdx, minVal, n);
        cudaMemcpyAsync(&minIdx, d_minIdx, sizeof(int), cudaMemcpyDeviceToHost, stream2);
        cudaStreamSynchronize(stream2);
        minIdxI = minIdx / n;
        minIdxJ = minIdx % n;
        delta = (rowsums[minIdxI] - rowsums[minIdxJ]) / (n - 2);

        makeNew<<<num_blocks, 1024, 0, stream1>>>(d_old_mat, d_new_mat, minIdxI, minIdxJ, n);
        cudaStreamSynchronize(stream1);

        cudaMemcpyAsync(new_mat, d_new_mat, (n - 1) * (n - 1) * sizeof(int), cudaMemcpyDeviceToHost, stream1);
        new_mat[0] = 0;
        ll ct = 0;
        for (ll m = 1; m < n - 1; m++) {
            while (ct == minIdxI || ct == minIdxJ) ct++;
            new_mat[m] = (matrix[minIdxI * n + ct] + matrix[minIdxJ * n + ct] - matrix[minIdxI * n + minIdxJ]) / 2;
            new_mat[m * (n - 1)] = new_mat[m];
            ct++;
        }

        tree.push_back(vector<int>());

        pair<int, int> pr = calculateLimbLengths(matrix, minIdxI, minIdxJ, delta, n);

        edgeWeights[prev[minIdxI]][top] = pr.first;
        edgeWeights[prev[minIdxJ]][top] = pr.second;
        edgeWeights[top][prev[minIdxI]] = pr.first;
        edgeWeights[top][prev[minIdxJ]] = pr.second;
        tree[top].push_back(prev[minIdxI]);
        tree[top].push_back(prev[minIdxJ]);
        tree[prev[minIdxI]].push_back(top);
        tree[prev[minIdxJ]].push_back(top);

        next[0] = top;
        ll cnt = 1;
        for (ll i = 0; i < n; i++) {
            if (i != minIdxI && i != minIdxJ) next[cnt++] = prev[i];
        }
        prev = next;
        num_blocks = ((n - 1) * (n - 1) + 1023) / 1024;
        cudaMemcpyAsync(d_new_mat, new_mat, (n - 1) * (n - 1) * sizeof(int), cudaMemcpyHostToDevice, stream1);
        copy_mat<<<num_blocks, 1024, 0, stream1>>>(d_old_mat, d_new_mat, n - 1);
        cudaStreamSynchronize(stream1);
        cudaMemcpyAsync(matrix, d_old_mat, (n - 1) * (n - 1) * sizeof(int), cudaMemcpyDeviceToHost, stream1);
        top++;
        n--;
    }

    tree[next[0]].push_back(next[1]);
    tree[next[1]].push_back(next[0]);
    edgeWeights[next[0]][next[1]] = new_mat[1];
    edgeWeights[next[1]][next[0]] = new_mat[1];

    auto stop = chrono::high_resolution_clock::now();

    auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);

    ofstream file("out1.txt");
    if (!file.is_open()) {
        cout << "Error: Unable to open file for writing." << endl;
        return 1;
    }
    for(int i = 0 ; i < top ; i++){
        file << i << " - ";
        for(auto v : tree[i]){
            file << v << " ";
        }
        file << endl;
    }   

    cout << "Time taken by function: " << duration.count() << " microseconds" << endl;

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaFree(d_old_mat);
    cudaFree(d_new_mat);
    cudaFree(d_rowsums);
    cudaFree(d_njMatrix);
    cudaFree(d_minVal);
    cudaFree(d_minIdx);
 
     return 0;
}