#include <vector>
#include <iostream>
#include <numeric>
#include <chrono>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <fstream>
# define ll long long

using namespace std;

using std::cin;
using std::cout;


__global__ void getsum(int * rowsums , int * matrix , int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n*n){    
        int row = idx/n;
        atomicAdd(&rowsums[row] , matrix[idx]);
    }
}


__global__ void create_nj(int *matrix, int *rowsums, int *njMat, int n) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n * n) {
        int i = idx / n;
        int j = idx % n;
        int value = matrix[idx];  // Fetch matrix element once

        if (i == j) {
            njMat[idx] = 0;
        } else {
            int nj_value = (n - 2) * value - rowsums[i] - rowsums[j];
            njMat[idx] = nj_value;
        }
    }
}


__global__ void findMin(int *njMat, int *result, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    // Initialize the result with the maximum integer value
    if (threadIdx.x == 0 && blockIdx.x == 0)
        *result = INT_MAX;
    
    __syncthreads();

    // Each thread finds the minimum value it has access to
    int minValue = (idx < n * n) ? njMat[idx] : INT_MAX;
    
    // Reduction to find the minimum value among all elements
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        minValue = min(minValue, __shfl_down_sync(0xFFFFFFFF, minValue, stride));
    }

    // The first thread in each block updates the result
    if (threadIdx.x == 0) {
        atomicMin(result, minValue);
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

__global__ void makeNew(int *old_matrix, int *new_matrix, int i, int j, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n * n) {
        int row = idx / n;
        int col = idx % n;
        int new_row, new_col;

        if (row < i) {
            new_row = row;
        } else if (row > j) {
            new_row = row - 1;
        } else {
            return; // Skip the elements in rows i and j
        }

        if (col < i) {
            new_col = col;
        } else if (col > j) {
            new_col = col - 1;
        } else {
            return; // Skip the elements in columns i and j
        }

        new_matrix[new_row * (n - 1) + new_col] = old_matrix[idx];
    }
}


__global__ void makeMerge(int *new_matrix, int *old_matrix, int *map, int i, int j, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // Only calculate for valid indices
    if (idx < n - 1) {
        int actual_idx = map[idx + 1]; // Offset idx by 1 since idx == 0 is handled differently
        int old_matrix_i_actual_idx = n * i + actual_idx;
        int old_matrix_j_actual_idx = n * j + actual_idx;
        int old_matrix_i_j = old_matrix[n * i + j];

        // Handle first element of new_matrix separately
        if (idx == 0) {
            new_matrix[0] = 0;
        } else {
            // Calculate new_matrix values
            int sum = old_matrix[old_matrix_i_actual_idx] + old_matrix[old_matrix_j_actual_idx] - old_matrix_i_j;
            int new_val = sum / 2;
            new_matrix[idx] = new_val;
            new_matrix[(n - 1) * idx] = new_val;
        }
    }
}


pair<int, int> calculateLimbLengths(int * matrix, int i, int j, int delta , int n) {
    int limbLengthI = (matrix[n*i +j] + delta) / 2;
    int limbLengthJ = (matrix[n*i +j] - delta) / 2;
    return make_pair(limbLengthI, limbLengthJ);
}

int main(int argc, char **argv){
    cout << "Print -" << endl;
    int n;
    cin >> n;
    cout << "Print- " << endl;
    int *matrix = new int[n * n];
    for (int i = 0; i < n; ++i){
        for (int j = 0; j < n; ++j){
            cin >> matrix[i * n + j];
        }
    }

    ofstream outfile("cuda.out");

    

    int top = n;
    vector<int> prev(n , 0);
    vector<int> next(n , 0);
    vector<vector<int>> tree(n);
    vector<vector<int>> edgeWeights(1e4 , vector<int> (1e4 , 0));
    int * new_mat = new int[n*n];
    int * rowsums = new int[n];
    int * njMatrix = new int[n * n];
    int * d_old_mat;
    int * d_new_mat;
    int * d_rowsums;
    int * d_njMatrix;

    int minVal , minIdx , minIdxI , minIdxJ , delta;
    minVal = 1e9;
    int * d_minVal , * d_minIdx;
    
    cudaMalloc(&d_njMatrix, n * n * sizeof(int));
    cudaMalloc(&d_old_mat, n * n * sizeof(int));
    cudaMalloc(&d_new_mat, n * n * sizeof(int));
    cudaMalloc(&d_rowsums, n * sizeof(int));
    cudaMalloc(&d_minVal, sizeof(int));
    cudaMalloc(&d_minIdx , sizeof(int));

    cudaMemcpy(d_minVal , &minVal , sizeof(int) , cudaMemcpyHostToDevice);

    cudaMemcpy(d_old_mat, matrix, n * n * sizeof(int), cudaMemcpyHostToDevice);

    for(int i = 0 ; i < n ; i ++) prev[i] = i;

    int num_blocks;
    auto start = chrono::high_resolution_clock::now();

    while(n - 2){

        // get row sums
        num_blocks = (n*n + 1023) / 1024;
        cudaMemset(d_rowsums , 0 , n*sizeof(int));

        getsum<<< num_blocks , 1024 >>> (d_rowsums , d_old_mat , n);
        cudaDeviceSynchronize();
        cudaMemcpy(rowsums , d_rowsums , n*sizeof(int) , cudaMemcpyDeviceToHost);

        // create nj matrix
        create_nj <<< num_blocks , 1024 >>> (d_old_mat , d_rowsums , d_njMatrix , n);
        cudaDeviceSynchronize();
        cudaMemcpy(njMatrix , d_njMatrix , n*n*sizeof(int) , cudaMemcpyDeviceToHost);

        //find Minval
        minVal = 1e5;
        cudaMemcpy(d_minVal , &minVal , sizeof(int) , cudaMemcpyHostToDevice);
        findMin <<< num_blocks , 1024 >>> (d_njMatrix , d_minVal , n);
        cudaDeviceSynchronize();
        cudaMemcpy(&minVal , d_minVal , sizeof(int) , cudaMemcpyDeviceToHost);

        //find MinIndices
        getMinidx <<< num_blocks , 1024 >>> (d_njMatrix , d_minIdx , minVal , n);
        cudaDeviceSynchronize();
        cudaMemcpy(&minIdx , d_minIdx , sizeof(int) , cudaMemcpyDeviceToHost);
        minIdxI = minIdx / n;
        minIdxJ = minIdx % n;
        delta = (rowsums[minIdxI] - rowsums[minIdxJ]) / (n-2);
        cout << "Min Indices - " << minIdxI << " " << minIdxJ << endl;

        // make newMatrix
        makeNew <<< num_blocks , 1024 >>> (d_old_mat , d_new_mat , minIdxI , minIdxJ , n);
        cudaDeviceSynchronize();

        cudaMemcpy(new_mat , d_new_mat , (n-1)*(n-1)*sizeof(int) ,  cudaMemcpyDeviceToHost);
        
        new_mat[0] = 0;
        ll ct = 0;
        for(ll m = 1 ; m < n - 1 ; m++){
            while(ct==minIdxI || ct==minIdxJ) ct++;
            new_mat[m] = (matrix[minIdxI * n + ct] + matrix[minIdxJ * n + ct] - matrix[minIdxI *n + minIdxJ]) / 2;
            new_mat[m*(n-1)] = new_mat[m];
            ct++;  
        }

        tree.push_back(vector<int>());
        
        pair<int,int> pr = calculateLimbLengths(matrix , minIdxI , minIdxJ , delta , n);

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
        for(ll i = 0 ; i < n ; i++){
            if(i!=minIdxI && i!=minIdxJ) next[cnt++] = prev[i];
        }
        cout << "OldMat" << endl;
        for(int i = 0 ; i < n ; i ++){
            for(int j = 0 ; j < n ; j ++){
                cout << matrix[n*i + j] << " "; 
            }
            cout << endl;
        }
        cout << "New Mat" << endl;
        for(int i = 0 ; i < n -1 ; i++){
            for(int j = 0 ; j < n - 1 ; j++){
                cout << new_mat[i*(n-1) + j] << " ";
            }
            cout << endl;
        }
        prev = next;
        matrix = new_mat;
        cudaMemcpy(d_old_mat , matrix , (n-1)*(n-1)*sizeof(int) , cudaMemcpyHostToDevice);
        cout << "After copy " << endl;
        for(int i = 0 ; i < n -1 ; i++){
            for(int j = 0 ; j < n - 1 ; j++){
                cout << matrix[i*(n-1) + j] << " ";
            }
            cout << endl;
        }
        top++;
        n--;
    }
    tree[next[0]].push_back(next[1]);
    tree[next[1]].push_back(next[0]);
    edgeWeights[next[0]][next[1]] = new_mat[1];
    edgeWeights[next[1]][next[0]] = new_mat[1];

    auto stop = chrono::high_resolution_clock::now();
    
    auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);

    for(int i = 0 ; i < top ; i++){
        cout << i << " - ";
        for(auto v : tree[i]){
            cout << v << " ";
        }
        cout << endl;
    }

    for(int i = 0 ; i < top ; i++){
        for(int j = 0 ; j < top ; j++){
            cout << edgeWeights[i][j] << " ";
        }
        cout << endl;
    }

    cout << "Time taken by function: " << duration.count() << " microseconds" << endl;
}