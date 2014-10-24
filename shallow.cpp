#include "solver.h"

#include <iostream>
#include <cmath>

template<typename real>
struct Foo {
    const std::string name() {
        return "foo";
    }
    real xlen() const { return 2; }
    real ylen() const { return 1; }

    void initial(const real x, const real y, sloped<real> &b,
            sloped<real> &h, sloped<real> &hu, sloped<real> &hv)
    {
        b.v = b.vx = b.vy = 0;
        hu.v = hu.vx = hu.vy = 0;
        hv.v = hv.vx = hv.vy = 0;

        const double sigma = 0.1;
        const double x0 = 0.25;
        const double y0 = 0.25;

        double ev = 0.1 * exp(-(pow(x - x0, 2) + pow(y - y0, 2)) / pow(sigma, 2));

        h.v = 1 + ev;
        h.vx = -2 * (x - x0) / pow(sigma, 2) * ev;
        h.vy = -2 * (y - y0) / pow(sigma, 2) * ev;
    }
};

int main() {
    try {
        auto ctx = std::make_shared<solver_context<float> >();
        Solver<float, 2, Foo<float> > s(/* M = */50, /* N = */100, /* C = */.15, Foo<float>(), ctx);

        while (s.step() < 10) {
            s.perform_step();
            s.save("out/");
        }
    } catch (const std::exception &e) {
        std::cerr << "Exception msg = " << e.what() << std::endl;
    }
    return 0;
}
