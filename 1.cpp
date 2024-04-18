#include <bits/stdc++.h>
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

    return 0;
    


}
