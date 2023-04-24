#include <iostream>

struct A {
    virtual void foo();
    void bar();
    virtual ~A();
};

void A::foo() {
    std::cout << "A::foo();\n";
}

A::~A() {
    std::cout << "A::~A();\n";
}

struct B : A {
    void foo() override;
    // void bar() override; 
    ~B() override;
    void override();
};

void B::foo() {
    std::cout << "B::foo();\n";
}

B::~B() {
    std::cout << "B::~B();\n";
}

void B::override() {
    std::cout << "B::override();\n";
}

int main() {
    B b;
    b.foo();
    b.override();
    int override{42};
    std::cout << "override: " << override << "\n";
    return 0;
}


