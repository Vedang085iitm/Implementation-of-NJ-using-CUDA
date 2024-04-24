#include <iostream>
#include <fstream>

using namespace std;

class TreeNode {
public:
    int value;
    TreeNode *left;
    TreeNode *right;

    TreeNode(int val) : value(val), left(nullptr), right(nullptr) {}
};

// Function to generate a Graphviz representation of the tree
void generateGraphviz(TreeNode *root, ofstream &file) {
    if (root == nullptr) return;

    file << "\t" << root->value << ";\n";

    if (root->left != nullptr) {
        file << "\t" << root->value << " -> " << root->left->value << ";\n";
        generateGraphviz(root->left, file);
    }
    if (root->right != nullptr) {
        file << "\t" << root->value << " -> " << root->right->value << ";\n";
        generateGraphviz(root->right, file);
    }
}

// Function to generate a Graphviz file
void generateGraphvizFile(TreeNode *root, const string &filename) {
    ofstream file(filename);

    file << "digraph Tree {\n";
    generateGraphviz(root, file);
    file << "}\n";

    file.close();
}

int main() {
    // Constructing a sample tree
    TreeNode *root = new TreeNode(1);
    root->left = new TreeNode(2);
    root->right = new TreeNode(3);
    root->left->left = new TreeNode(4);
    root->left->right = new TreeNode(5);
    root->right->left = new TreeNode(6);
    root->right->right = new TreeNode(7);

    // Generate the Graphviz file
    generateGraphvizFile(root, "tree.dot");

    // Convert the Graphviz file to an image using the 'dot' command
    system("dot -Tpng tree.dot -o tree.png");

    // Clean up
    delete root->left->left;
    delete root->left->right;
    delete root->right->left;
    delete root->right->right;
    delete root->left;
    delete root->right;
    delete root;

    return 0;
}
