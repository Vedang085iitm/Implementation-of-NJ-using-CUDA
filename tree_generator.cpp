#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

using namespace std;
#include <fstream>
#include <sstream>
#include <vector>
#include <utility>

vector<pair<int, vector<int>>> readInputFromFile(const string &filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Could not open file: " << filename << endl;
        return {};
    }

    vector<pair<int, vector<int>>> input;
    string line;
    while (getline(file, line)) {
        stringstream ss(line);
        int parent;
        ss >> parent;
        ss.ignore(3); // Skip " ->"

        vector<int> children;
        int child;
        while (ss >> child) {
            children.push_back(child);
        }

        input.push_back({parent, children});
    }

    return input;
}


int main() {
vector<pair<int, vector<int>>> input = readInputFromFile("treeinput.txt");
    // Find the maximum node value
    int maxNode = 0;
    for (const auto &item : input) {
        maxNode = max(maxNode, item.first);
        for (int child : item.second) {
            maxNode = max(maxNode, child);
        }
    }

    // Declare the adjacencyList variable and resize it
    vector<vector<int>> adjacencyList(maxNode + 1);

    // Constructing the adjacency list
    for (auto &item : input) {
        int node = item.first;
        vector<int> adjacentNodes = item.second;
        adjacencyList[node] = adjacentNodes;
    }

// Write the DOT code for the graph structure
ofstream outputFile("graph.dot");

if (!outputFile) {
    cerr << "Error: Unable to open output file." << endl;
    return 1;
}

outputFile << "graph {" << endl;
for (int node = 0; node < adjacencyList.size(); ++node) {
    for (int adjacentNode : adjacencyList[node]) {
        outputFile << node << " -- " << adjacentNode << ";" << endl;
    }
}
outputFile << "}" << endl;

outputFile.close();

    cout << "Graph DOT file generated successfully." << endl;

    return 0;
}
