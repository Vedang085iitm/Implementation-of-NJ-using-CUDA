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
// vector<ll> rowsum(vector<vector<ll>> matrix, ll n)
// {
//     vector<ll> sums(n, 0);
//     for(ll i=0;i<n;i++)
//     {
//         sums[i] = accumulate(matrix[i].begin(), matrix[i].end(), 0LL);
//     }
//     return sums;
// }
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


int main()
{
    ll n;
    cout<<"Enter Size of the matrix: ";
    cin>>n;
    //define 2d vector
    vector<vector<ll>> matrix(n, vector<ll>(n));
    cout<<"Enter the elements of the matrix: ";
    for(ll i=0;i<n;i++)
    {
        for(ll j=0;j<n;j++)
        {
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
    auto stop = chrono::high_resolution_clock::now();
auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);

cout << "Time taken by function: " << duration.count() << " microseconds" << endl;
    return 0;
    


}
