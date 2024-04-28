
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

// __global__ void getsum(int * rowsums , int * matrix , int n){
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if(idx < n*n){    
//         int row = idx/n;
//         atomicAdd(&rowsums[row] , matrix[idx]);
//     }
// }
__global__ void getsum(int *rowsums, int *matrix, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * n) {
        // Code goes here
    
        int row = idx / n;
        int col = idx % n;
        if (row <= col) {  // Compute only upper triangular portion or diagonal
            atomicAdd(&rowsums[row], matrix[idx]);
            if (row != col) {  // If not on the diagonal, mirror the sum
                atomicAdd(&rowsums[col], matrix[idx]);
            }
        }
    }
}

// __global__ void getsum(int *rowsums, int *matrix, int n) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if(idx < n*n){
//         int row = idx/n;
//         int col = idx%n;
//         if(col <= row){ // only consider elements in the lower triangular part
//             atomicAdd(&rowsums[row], matrix[idx]*2);
//         }
//     }
// }

//**********************************************************************************************************************************//


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
// /The kernel is parallelized such that each thread handles one element of the matrix, which can lead to significant speedup when dealing with large matrices. The kernel is also optimized to avoid redundant memory accesses by caching the matrix element in a register and only accessing it once.

//**********************************************************************************************************************************//




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
//The threads perform a reduction operation to find the minimum value among all elements. This is done using a loop that iteratively halves the stride and uses the __shfl_down_sync function to compare values from different threads. The min function is used to keep the smaller of the two values.

// After the reduction, the first thread in each block (threadIdx.x == 0) uses the atomicMin function to update result with the minimum value found in that block. The atomicMin function ensures that this update is done atomically, i.e., without interference from other threads.

// The kernel is parallelized such that each thread handles one element of the matrix, and the reduction operation allows the minimum value to be found efficiently. This approach is scalable and can handle matrices of any size.
//**********************************************************************************************************************************//



__global__ void getMinidx(const int *njMat, int *min_idx, int minVal, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n * n && njMat[idx] == minVal) {
        int row = idx / n;
        int col = idx % n;
        min_idx[0] = row * n + col;
    }
}


//**********************************************************************************************************************************//



// __global__ void makeNew(int * old_matrix , int * new_matrix , int i , int j , int n){
//     int idx = blockDim.x*blockIdx.x + threadIdx.x;
//     if(idx < n*n){
//         int row = idx / n;
//         int col = idx % n;
//         if(row==i || row ==j || col==i || col==j){
//             return;
//         }
//         if(i > j){
//             int tmp = j;
//             j = i;
//             i = tmp;
//         }
//         if(row < i){
//             if(col < i){
//                 new_matrix[(n-1)*(1 + row) + (1 + col)] = old_matrix[idx];
//             }
//             else if(col > j){
//                 new_matrix[(n-1)*(1 + row) + (col - 1)] = old_matrix[idx];
//             }
//             else{
//                 new_matrix[(n-1)*(1 + row) + (col)] = old_matrix[idx];
//             }
//         }
//         else if(row > j){
//             if(col < i){
//                 new_matrix[(n-1)*(row - 1) + (1 + col)] = old_matrix[idx];
//             }
//             else if(col > j){
//                 new_matrix[(n-1)*(row - 1) + (col - 1)] = old_matrix[idx];
//             }
//             else{
//                 new_matrix[(n-1)*(row - 1) + (col)] = old_matrix[idx];
//             }
//         }
//         else{
//             if(col < i){
//                 new_matrix[(n-1)*(row) + (1 + col)] = old_matrix[idx];
//             }
//             else if(col > j){
//                 new_matrix[(n-1)*(row) + (col - 1)] = old_matrix[idx];
//             }
//             else{
//                 new_matrix[(n-1)*(row) + (col)] = old_matrix[idx];
//             }
//         }
//     }
// }

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

__global__ void makeMerge(int *new_matrix, const int *old_matrix, const int *map, int i, int j, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (idx < n - 1) {
        int actual_idx = map[idx];
        
        if (idx == 0) {
            new_matrix[0] = 0;
            new_matrix[(n - 1) * idx] = 0;  
        } else {
            int old_ij = old_matrix[n * i + j];  // Cache old_matrix[n * i + j]
            int old_ia = old_matrix[n * i + actual_idx];  // Cache old_matrix[n * i + actual_idx]
            int old_ja = old_matrix[n * j + actual_idx];  // Cache old_matrix[n * j + actual_idx]
            
            new_matrix[idx] = (old_ia + old_ja - old_ij) / 2;
            new_matrix[(n - 1) * idx] = new_matrix[idx];
        }
    }
}
//The kernel is parallelized such that each thread handles one row of the new matrix, which can lead to significant speedup when dealing with large matrices. The kernel is also optimized to avoid redundant memory accesses by caching the matrix elements in registers and only accessing them once.


pair<int, int> calculateLimbLengths(int * matrix, int i, int j, int delta , int n) {
    int limbLengthI = (matrix[n*i +j] + delta) / 2;
    int limbLengthJ = (matrix[n*i +j] - delta) / 2;
    return make_pair(limbLengthI, limbLengthJ);
}

__global__ void copy_mat(int *to, const int *from, int n) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int idx = tid; idx < n * n; idx += stride) {
        
        to[idx] = from[idx];
    }
}
//The function then enters a loop, where each thread starts from its unique ID (tid) and increments by the stride. This means that each thread handles every stride-th element of the matrices, starting from its unique ID. This approach ensures that the work of copying the matrix is evenly distributed across all the threads, which can lead to significant speedup when dealing with large matrices.

int main(int argc, char **argv){
    // cout << "Enter the number of vertices -" << endl;
    int n;
    cin >> n;
    // cout << "Enter the distance matrix - " << endl;
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

    int * debug_old = new int[n*n];

    int num_blocks;
    auto start = chrono::high_resolution_clock::now();

    while(n - 2){

        // get row sums
        num_blocks = (n*n + 1023) / 1024;
        cudaMemset(d_rowsums , 0 , n*sizeof(int));
        getsum<<< num_blocks , 1024 >>> (d_rowsums , d_old_mat , n);
        cudaDeviceSynchronize();
        cudaMemcpy(rowsums , d_rowsums , n*sizeof(int) , cudaMemcpyDeviceToHost);

        cudaMemcpy(debug_old , d_old_mat , n*n*sizeof(int) , cudaMemcpyDeviceToHost);

        // create nj matrix
        create_nj <<< num_blocks , 1024 >>> (d_old_mat , d_rowsums , d_njMatrix , n);
        cudaDeviceSynchronize();
        cudaMemcpy(njMatrix , d_njMatrix , n*n*sizeof(int) , cudaMemcpyDeviceToHost);

        cudaMemcpy(debug_old , d_old_mat , n*n*sizeof(int) , cudaMemcpyDeviceToHost);
        minVal = 1e9;
        cudaMemcpy(d_minVal , &minVal , sizeof(int) , cudaMemcpyHostToDevice);
        findMin <<< num_blocks , 1024 >>> (d_njMatrix , d_minVal , n);
        cudaDeviceSynchronize();
        cudaMemcpy(&minVal , d_minVal , sizeof(int) , cudaMemcpyDeviceToHost);

        cudaMemcpy(debug_old , d_old_mat , n*n*sizeof(int) , cudaMemcpyDeviceToHost);

        //find MinIndices
        getMinidx <<< num_blocks , 1024 >>> (d_njMatrix , d_minIdx , minVal , n);
        cudaDeviceSynchronize();
        cudaMemcpy(&minIdx , d_minIdx , sizeof(int) , cudaMemcpyDeviceToHost);
        minIdxI = minIdx / n;
        minIdxJ = minIdx % n;
        delta = (rowsums[minIdxI] - rowsums[minIdxJ]) / (n-2);
        cudaMemcpy(debug_old , d_old_mat , n*n*sizeof(int) , cudaMemcpyDeviceToHost);

        // make newMatrix
        makeNew <<< num_blocks , 1024 >>> (d_old_mat , d_new_mat , minIdxI , minIdxJ , n);
        cudaDeviceSynchronize();
        cudaMemcpy(debug_old , d_old_mat , n*n*sizeof(int) , cudaMemcpyDeviceToHost);

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
        prev = next;
        num_blocks = ((n-1)*(n-1) + 1023) / 1024;
        cudaMemcpy(d_new_mat , new_mat , (n-1)*(n-1)*sizeof(int) , cudaMemcpyHostToDevice);
        copy_mat<<< num_blocks , 1024 >>> (d_old_mat , d_new_mat , n -1);
        cudaDeviceSynchronize();
        cudaMemcpy(matrix , d_old_mat , (n-1)*(n-1)*sizeof(int) , cudaMemcpyDeviceToHost);
        top++;
        n--;
    }
    tree[next[0]].push_back(next[1]);
    tree[next[1]].push_back(next[0]);
    edgeWeights[next[0]][next[1]] = new_mat[1];
    edgeWeights[next[1]][next[0]] = new_mat[1];

    auto stop = chrono::high_resolution_clock::now();
    
    auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);

    ofstream file("treeinput.txt");
if (!file.is_open()) {
    cerr << "Could not open file: treeinput.txt" << endl;
    return 1;
}

    for(int i = 0 ; i < top ; i++){
        file << i << " ->";
        for(auto v : tree[i]){
            file << v << " ";
        }
        file << endl;
    }

    // for(int i = 0 ; i < top ; i++){
    //     for(int j = 0 ; j < top ; j++){
    //         cout << edgeWeights[i][j] << " ";
    //     }
    //     cout << endl;
    // }
    cout<<"Code executed successfully"<<endl;

    cout << "Time taken by function: " << duration.count() << " microseconds" << endl;
}