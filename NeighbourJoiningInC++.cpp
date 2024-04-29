#include <bits/stdc++.h>
#include <numeric>
#include <chrono>
using namespace std;
typedef long ll;

vector<ll> rowsum(vector<vector<ll>> matrix, ll n)
{
    
    vector<ll> sums(n, 0);
    for(ll i=0;i<n;i++)
    {
        for(ll j=0;j<n;j++)
        {
            sums[i] += matrix[i][j];
        }
    }
    return sums;
}

//here we will check the time of the above functions, the onne which is slower will be used

vector<vector<ll>> neighborJoiningMatrix(vector<vector<ll>>& matrix, vector<ll>& rowSums, ll n) {
    vector<vector<ll>> njMatrix(n, vector<ll>(n, 0));

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

pair<ll, pair<ll, ll>> findMinAndComputeDelta(vector<vector<ll>>& njMatrix, vector<ll>& rowSums, ll n) {
    ll minVal = LLONG_MAX;
    pair<ll, ll> minIndices;
    for(ll i = 0; i < n; i++) {
        for(ll j = 0; j < n; j++) {
            if(i != j && njMatrix[i][j] < minVal) {
                minVal = njMatrix[i][j];
                minIndices = make_pair(i, j);
            }
        }
    }

    ll delta = (rowSums[minIndices.first] - rowSums[minIndices.second]) / (n - 2);
    return make_pair(delta, minIndices);
}



pair<ll, ll> calculateLimbLengths(vector<vector<ll>>& D, ll i, ll j, ll delta) {
    ll limbLengthI = (D[i][j] + delta) / 2;
    ll limbLengthJ = (D[i][j] - delta) / 2;

    return make_pair(limbLengthI, limbLengthJ);
}


vector<vector<ll>> formNewMatrix(vector<vector<ll>>& D, ll i, ll j) {
    ll n = D.size();
    vector<vector<ll>> D_prime(n - 1, vector<ll>(n - 1 , 0));
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

void compute(vector<vector<ll>> & matrix , vector<vector<ll>> &tree , int n){
    ll top = n;
    vector<ll> prev(n , 0);
    vector<ll> next(n , 0);
    vector<ll> sums;
    vector<vector<ll>> njMat;
    pair<ll,pair<ll,ll>> minDel;
    vector<vector<ll>> oldMat = matrix;
    vector<vector<ll>> newMat;
    // vector<vector<ll>> edgeWeights(100, vector<ll> (100 , 0));
    // think how to store weights
    for(ll i = 0 ; i < n ; i ++) prev[i] = i;
    while(n-2){
        sums = rowsum(oldMat , n);
        njMat = neighborJoiningMatrix(oldMat , sums , n);
        minDel = findMinAndComputeDelta(njMat , sums , n);
        ll i_node = minDel.second.first;
        ll j_node = minDel.second.second;
        ll delta = minDel.first;
        newMat = formNewMatrix(oldMat , i_node , j_node);
        tree.push_back(vector<ll>()); // pushed in merged node
        pair<ll,ll> pr = calculateLimbLengths(oldMat , i_node , j_node , delta);
        // edgeWeights[prev[i_node]][top] = pr.first;
        // edgeWeights[prev[j_node]][top] = pr.second;
        // edgeWeights[top][prev[i_node]] = pr.first;
        // edgeWeights[top][prev[j_node]] = pr.second;
        tree[top].push_back(prev[i_node]);
        tree[top].push_back(prev[j_node]);
        tree[prev[i_node]].push_back(top);
        tree[prev[j_node]].push_back(top);
        // add edge weights
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
    // edgeWeights[next[0]][next[1]] = newMat[0][1];
    // edgeWeights[next[1]][next[0]] = newMat[0][1];

    // for(int i = 0 ; i < top ; i++){
    //     for(int j = 0 ; j < top ; j++){
    //         cout << edgeWeights[i][j] << " ";
    //     }
    //     cout << endl;
    // }
}


int main(int argc , char ** argv){
    ll n;
    cout<<"Enter Size of the matrix: ";
    cin>>n;
    //define 2d vector
    vector<vector<ll>> matrix(n, vector<ll>(n));
    cout<<"Enter the elements of the matrix: ";
    for(ll i=0;i<n;i++){
        for(ll j=0;j<n;j++){
            cin>>matrix[i][j];
        }
    }

    vector<vector<ll>> tree(n);

    auto start = chrono::high_resolution_clock::now();

    compute(matrix , tree , n);

    auto stop = chrono::high_resolution_clock::now();
    
    auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
    ofstream file("out.txt");
    if(file.is_open()){
        cout << "-1" << endl;
    }

    for(int i = 0 ; i < tree.size() ; i++){
        file << i << " - ";
        for(auto v : tree[i]){
            file << v << " ";
        }
        file << endl;
    }

    cout << "Time taken by function: " << duration.count() << " microseconds" << endl;
    return 0;
}