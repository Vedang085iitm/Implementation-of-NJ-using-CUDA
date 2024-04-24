#include <bits/stdc++.h>
#include <numeric>
#include <chrono>
#include <thread>
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
    cout << n << endl;
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


int main(){
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

    auto start = chrono::high_resolution_clock::now();

    vector<ll> sums = rowsum(matrix, n);

    cout << "The sum of each row is:\n";
    for (const auto& sum : sums) {
        cout << sum << '\n';
    }
    vector<vector<ll>> njMatrix = neighborJoiningMatrix(matrix, sums, n);
    cout << "The neighbor joining matrix is:\n";
    for (const auto& row : njMatrix) {
        for (const auto& elem : row) {
            cout << elem << ' ';
        }
        cout << '\n';
    }
    pair<ll, pair<ll, ll>> minAndDelta = findMinAndComputeDelta(njMatrix, sums, n);
    cout << "The minimum value in the neighbor joining matrix is: " << minAndDelta.first << '\n';

    vector<vector<ll>> newMatrix = formNewMatrix(matrix, minAndDelta.second.first, minAndDelta.second.second);
    cout << "The new matrix after removing the rows and columns corresponding to the minimum value is:\n";
    for (const auto& row : newMatrix) {
        for (const auto& elem : row) {
            cout << elem << ' ';
        }
        cout << '\n';
    }
    unsigned int nn = std::thread::hardware_concurrency();
    std::cout << "Number of concurrent threads supported: " << nn << "\n";
   

    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
    

    cout << "Time taken by function: " << duration.count() << " microseconds" << endl;
    return 0;
}