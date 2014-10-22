#include "solver.h"

#include <iostream>

template<typename real>
struct Foo {
    const std::string name() {
        return "Foo";
    }
    real xlen() const { return 2; }
    real ylen() const { return 1; }
};

int main() {
    try {
        auto ctx = std::make_shared<solver_context<float> >();
        Solver<float, 2, Foo<float> > s(/* M = */50, /* N = */100, /* C = */.15, Foo<float>(), ctx);

        s.perform_step();
        s.save("out");
    } catch (const std::exception &e) {
        std::cerr << "Exception msg = " << e.what() << std::endl;
    }
    return 0;
}
