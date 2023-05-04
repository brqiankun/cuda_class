#include<iostream>
#include<string>
#include<vector>

using namespace std;

int main() {
    printf("unsigned int : %d\n",sizeof(unsigned int));
    printf("int : %d\n",sizeof(int));
    printf("float : %d\n",sizeof(float));
    printf("double : %d\n",sizeof(double));

    string dict[10] = {"", "", "abc", "def", "ghi",
                           "jkl", "mno", "pqrs", "tuv", 
                           "wxyz"};
    
    for(int i = 0; i < dict[2].size(); i++) {
            cout<<*(&(*(dict + 2)) + i);
        }

    return 0;

    
}