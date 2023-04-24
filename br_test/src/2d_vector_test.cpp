#include <iostream>
#include <vector>

int main() {
    int row = 3;
    int col = 4;
    std::vector<std::vector<int> > test_vec(row, std::vector<int>{1, 2, 3, 4});
    std::vector<float> feature;
    
    size_t offset;
    int *prev = NULL;
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            std::cout << "test_vec[" << i << "]" << "[" << j << "]: " << test_vec[i][j]
                      << "  ptr: " << &(test_vec[i][j]) << std::endl;
            offset = (size_t)(&(test_vec[i][j]) - prev);
            prev = &(test_vec[i][j]);
            std::cout << "offset: " << offset << std::endl;
        }
    }
    feature.insert(feature.end(), 5, 0);
    for (int i = 0; i < 5; i++) {
        std::cout << feature[i] << std::endl;
    }
    return 0;
}