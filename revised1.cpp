#include <bits/stdc++.h>
#include <numeric>
#include <chrono>
#include <fstream>
using namespace std;
typedef long ll;

// vector<ll> rowsum(vector<vector<ll>> matrix, ll n)
// {
    
//     vector<ll> sums(n, 0);
//     for(ll i=0;i<n;i++)
//     {
//         for(ll j=0;j<n;j++)
//         {
//             sums[i] += matrix[i][j];
//         }
//     }
//     return sums;
// }
vector<float> rowsum(vector<vector<float>> matrix, int n)
{
    vector<float> sums(n, 0.0f);
    for(ll i=0; i<n; i++)
    {
        for(ll j=0; j<n; j++)
        {
            sums[i] += matrix[i][j];
        }
    }
    return sums;
}

// vector<vector<ll>> neighborJoiningMatrix(vector<vector<ll>>& matrix, vector<ll>& rowSums, ll n) {
//     vector<vector<ll>> njMatrix(n, vector<ll>(n, 0));

//     for(ll i = 0; i < n; i++) {
//         for(ll j = 0; j < n; j++) {
//             if(i == j) {
//                 njMatrix[i][j] = 0;
//                 continue;
//             }
//             njMatrix[i][j] = (n - 2) * matrix[i][j] - rowSums[i] - rowSums[j];
//         }
//     }
//     return njMatrix;
// }
vector<vector<float>> neighborJoiningMatrix(vector<vector<float>>& matrix, vector<float>& rowSums, ll n) {
    vector<vector<float>> njMatrix(n, vector<float>(n, 0.0f));

    for(ll i = 0; i < n; i++) {
        for(ll j = 0; j < n; j++) {
            if(i == j) {
                njMatrix[i][j] = 0;
                continue;
            }
            njMatrix[i][j] = (n - 2) * matrix[i][j] - rowSums[i] - rowSums[j];
        }
    }
    return njMatrix;
}
// pair<ll, pair<ll, ll>> findMinAndComputeDelta(vector<vector<ll>>& njMatrix, vector<ll>& rowSums, ll n) {
//     ll minVal = LLONG_MAX;
//     pair<ll, ll> minIndices;
//     for(ll i = 0; i < n; i++) {
//         for(ll j = 0; j < n; j++) {
//             if(i != j && njMatrix[i][j] < minVal) {
//                 minVal = njMatrix[i][j];
//                 minIndices = make_pair(i, j);
//             }
//         }
//     }

//     ll delta = (rowSums[minIndices.first] - rowSums[minIndices.second]) / (n - 2);
//     return make_pair(delta, minIndices);
// }
pair<float, pair<ll, ll>> findMinAndComputeDelta(vector<vector<float>>& njMatrix, vector<float>& rowSums, ll n) {
    float minVal = LLONG_MAX*1.0;
    pair<ll, ll> minIndices;
    for(ll i = 0; i < n; i++) {
        for(ll j = 0; j < n; j++) {
            if(i != j && njMatrix[i][j] < minVal) {
                minVal = njMatrix[i][j];
                minIndices = make_pair(i, j);
            }
        }
    }

    float delta = (rowSums[minIndices.first] - rowSums[minIndices.second]) / (n - 2);
    return make_pair(delta, minIndices);
}



pair<float, float> calculateLimbLengths(vector<vector<float>>& D, int i, int j, float delta) {
    float limbLengthI = (D[i][j] + delta) / 2.0f;
    float limbLengthJ = (D[i][j] - delta) / 2.0f;

    return make_pair(limbLengthI, limbLengthJ);
}


vector<vector<float>> formNewMatrix(vector<vector<float>>& D, ll i, ll j) {
    float n = D.size();
    vector<vector<float>> D_prime(n - 1, vector<float>(n - 1 , 0));
    ll rct = 1;
    ll cct;
    for(ll k = 0 ; k < n ; k++){
        if(k!=i && k!=j){
            cct = 1;
            for(ll m = 0 ; m < n ; m++){
                if(m!=i && m!=j){
                    D_prime[rct][cct] = D[k][m];
                    cct++;
                }
            }
            rct++;
        }
    }
    D_prime[0][0] = 0;
    ll ct = 0;
    for(ll m = 1 ; m < n - 1 ; m++){
        while(ct==i || ct==j) ct++;
        D_prime[0][m] = (D[i][ct] + D[j][ct] - D[i][j]) / 2;
        D_prime[m][0] = D_prime[0][m];
        ct++;  
    }
    return D_prime;
}

// void compute(vector<vector<ll>> & matrix , vector<vector<ll>> &tree , vector<vector<ll>> & edgeWeights , int n){
//     ll top = n;  
//     vector<ll> prev(n , 0);   
//     vector<ll> next(n , 0);   
//     vector<ll> sums;          
//     vector<vector<ll>> njMat;   
//     pair<ll,pair<ll,ll>> minDel;  
//     vector<vector<ll>> oldMat = matrix; 
//     vector<vector<ll>> newMat;
//     for(ll i = 0 ; i < n ; i ++) prev[i] = i; 
//     while(n-2){ 
//         sums = rowsum(oldMat , n); 
//         njMat = neighborJoiningMatrix(oldMat , sums , n);  
//         minDel = findMinAndComputeDelta(njMat , sums , n);
//         ll i_node = minDel.second.first; 
//         ll j_node = minDel.second.second; 
//         ll delta = minDel.first; 
//         newMat = formNewMatrix(oldMat , i_node , j_node); 
//         tree.push_back(vector<ll>()); // pushed in merged node 
//         pair<ll,ll> pr = calculateLimbLengths(oldMat , i_node , j_node , delta); 
//         edgeWeights[prev[i_node]][top] = pr.first;
//         edgeWeights[prev[j_node]][top] = pr.second;
//         edgeWeights[top][prev[i_node]] = pr.first;
//         edgeWeights[top][prev[j_node]] = pr.second;
//         tree[top].push_back(prev[i_node]);
//         tree[top].push_back(prev[j_node]);
//         tree[prev[i_node]].push_back(top);
//         tree[prev[j_node]].push_back(top);
//         // add edge weights
//         next[0] = top;
//         ll ct = 1;
//         for(ll i = 0 ; i < n ; i++){
//             if(i!=i_node && i!=j_node) next[ct++] = prev[i];
//         }
//         prev = next;
//         oldMat = newMat;
//         top++;
//         n--;
//     }
//     tree[next[0]].push_back(next[1]);
//     tree[next[1]].push_back(next[0]);
//     edgeWeights[next[0]][next[1]] = newMat[0][1];
//     edgeWeights[next[1]][next[0]] = newMat[0][1];
// }
void compute(vector<vector<float>> & matrix , vector<vector<ll>> &tree , vector<vector<float>> & edgeWeights , ll n){
    ll top = n;  
    vector<ll> prev(n , 0);   
    vector<ll> next(n , 0);   
    vector<float> sums;          
    vector<vector<float>> njMat;   
    pair<float,pair<ll,ll>> minDel;  
    vector<vector<float>> oldMat = matrix; 
    vector<vector<float>> newMat;
    for(ll i = 0 ; i < n ; i ++) prev[i] = i; 
    while(n-2){ 
        sums = rowsum(oldMat , n); 
        njMat = neighborJoiningMatrix(oldMat , sums , n);  
        minDel = findMinAndComputeDelta(njMat , sums , n);
        ll i_node = minDel.second.first; 
        ll j_node = minDel.second.second; 
        float delta = minDel.first; 
        newMat = formNewMatrix(oldMat , i_node , j_node); 
        tree.push_back(vector<ll>()); // pushed in merged node 
        pair<float,float> pr = calculateLimbLengths(oldMat , i_node , j_node , delta); 
        edgeWeights[prev[i_node]][top] = pr.first;
        edgeWeights[prev[j_node]][top] = pr.second;
        edgeWeights[top][prev[i_node]] = pr.first;
        edgeWeights[top][prev[j_node]] = pr.second;
        tree[top].push_back(prev[i_node]);
        tree[top].push_back(prev[j_node]);
        tree[prev[i_node]].push_back(top);
        tree[prev[j_node]].push_back(top);
        next[0] = top;
        ll ct = 1;
        for(ll i = 0 ; i < n ; i++){
            if(i!=i_node && i!=j_node) next[ct++] = prev[i];
        }
        prev = next;
        oldMat = newMat;
        top++;
        n--;
    }
    tree[next[0]].push_back(next[1]);
    tree[next[1]].push_back(next[0]);
    edgeWeights[next[0]][next[1]] = newMat[0][1];
    edgeWeights[next[1]][next[0]] = newMat[0][1];
}


int main(int argc, char** argv){
    ll n;
    cin>>n;
    vector<vector<float>> matrix(n, vector<float>(n));
    for(ll i=0;i<n;i++){
        for(ll j=0;j<n;j++){
            cin>>matrix[i][j];
        }
    }
    // ofstream outfile("cpp.out");

    vector<vector<ll>> tree(n);

    vector<vector<float>> edgeWeights(1e4, vector<float> (1e4 , 0));

    auto start = chrono::high_resolution_clock::now();

    compute(matrix , tree , edgeWeights , n);

    auto stop = chrono::high_resolution_clock::now();
    
    auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);

    cout << "Printing the tree - \n" << endl;

    ofstream file("treeinput.txt");
if (!file.is_open()) {
    cerr << "Could not open file: treeinput.txt" << endl;
    return 1;
}

for(int i = 0 ; i < tree.size() ; i++){
    file << i << " ->";
    for(auto v : tree[i]){
        file << v << " ";
    }
    file << endl;
}

file.close();
    //weights can be queried from edgeWeight array using the corresponding vertices
    
    cout << "Time taken by function: " << duration.count() << " microseconds" << endl;
    return 0;
}