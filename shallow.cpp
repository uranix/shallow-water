#include "solver.h"

#include <iostream>
#include <cmath>

template<typename real>
struct Flood {
    const std::string name() {
        return "flood";
    }
    real xlen() const { return 2; }
    real ylen() const { return 1; }
    real tmax() const { return 50; }
    real next(real t) const { return t + 0.05; }

    void initial(const real x, const real y, sloped<real> &b,
            sloped<real> &h, sloped<real> &hu, sloped<real> &hv)
    {
        hu.v = hu.vx = hu.vy = 0;
        hv.v = hv.vx = hv.vy = 0;

        b.v  = 0.2 * x;
        b.vx = 0.2;
        b.vy = 0;

        real xdry = 1;

        if (x < xdry) {
            real ev = exp(-100 * (pow(x - 0.25, 2) + pow(y - 0.25, 2)));
            h.v  = 0.2 * (1 - x) + ev;
            h.vx = -b.vx - 100 * 2 * (x - 0.25) * ev;
            h.vy = -b.vy - 100 * 2 * (y - 0.25) * ev;
        } else {
            h.v = 0;
            h.vx = 0;
            h.vy = 0;
        }
    }
};

template<typename real>
struct FlatFlood {
    const std::string name() {
        return "flat";
    }
    real xlen() const { return 2; }
    real ylen() const { return 1; }
    real tmax() const { return 50; }
    real next(real t) const { return t + 0.05; }

    void initial(const real x, const real y, sloped<real> &b,
            sloped<real> &h, sloped<real> &hu, sloped<real> &hv)
    {
        hu.v = hu.vx = hu.vy = 0;
        hv.v = hv.vx = hv.vy = 0;

        b.v  = 0;
        b.vx = 0;
        b.vy = 0;

        h.v  = 0 * exp(-10 * x);
        h.vx = 0 * -10 * exp(-10 * x);
        h.vy = 0;
    }
};

template<typename real>
struct Balance {
    const std::string name() {
        return "balance";
    }
    real xlen() const { return 2; }
    real ylen() const { return 1; }
    real tmax() const { return 50; }
    real next(real t) const { return t + 0.05; }

    void initial(const real x, const real y, sloped<real> &b,
            sloped<real> &h, sloped<real> &hu, sloped<real> &hv)
    {
        hu.v = hu.vx = hu.vy = 0;
        hv.v = hv.vx = hv.vy = 0;

        b.v  = 0.1 * (3 * x + 2 * y);
        b.vx = 0.3;
        b.vy = 0.2;

        h.v  = 1 - b.v;
        h.vx = -b.vx;
        h.vy = -b.vy;
    }
};

template<typename real>
struct Neat {
    const std::string name() {
        return "neat";
    }
    real xlen() const { return 3; }
    real ylen() const { return 1; }
    real tmax() const { return 50; }
    real next(real t) const { return t + 0.05; }

    void initial(const real x, const real y, sloped<real> &b,
            sloped<real> &h, sloped<real> &hu, sloped<real> &hv)
    {
        hu.v = hu.vx = hu.vy = 0;
        hv.v = hv.vx = hv.vy = 0;

        b.v  = 0.1 * (3 * x + 2 * y);
        b.vx = 0.3;
        b.vy = 0.2;

        real x0 = 0.25;
        real y0 = 0.15;
        real ev = exp(-(pow(x - x0, 2) + pow(y - y0, 2)) / 0.01);

        h.v  = 2.5 - b.v + ev;
        h.vx = -b.vx + ev * 2 * (x0 - x) / 0.01;
        h.vy = -b.vy + ev * 2 * (y0 - y) / 0.01;
    }
};

int main() {
    try {
        auto ctx = std::make_shared<solver_context<float> >();
//        Solver<float, 2, Balance<float> > s(/* M = */50, /* N = */25, /* C = */.3, Balance<float>(), ctx);
        Solver<float, 2, Flood<float> > s(/* M = */400, /* N = */200, /* C = */.15, Flood<float>(), ctx);
//        Solver<float, 2, FlatFlood<float> > s(/* M = */200, /* N = */1, /* C = */.3, FlatFlood<float>(), ctx);
//        Solver<float, 2, Neat<float> > s(/* M = */150, /* N = */50, /* C = */.3, Neat<float>(), ctx);
        s.save("out/");

        float tout = 0;
        while (s.time() < s.tmax()) {
            if (s.time() >= tout)
            {
                tout = s.next(tout);
                std::cout << "Saving step #" << s.step() << ", t = " << s.time() << std::endl;
                s.save("out/");
            }
            s.perform_step();
        }
    } catch (const std::exception &e) {
        std::cerr << "Exception msg = " << e.what() << std::endl;
    }
    return 0;
}
