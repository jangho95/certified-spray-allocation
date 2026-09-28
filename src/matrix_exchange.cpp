#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    try {
        if (argc != 4) throw std::runtime_error("input output greedy|exchange required");
        std::ifstream in(argv[1], std::ios::binary);
        int64_t n, B, C;
        in.read(reinterpret_cast<char*>(&n), 8);
        in.read(reinterpret_cast<char*>(&B), 8);
        in.read(reinterpret_cast<char*>(&C), 8);
        if (!in || n < 1 || n > 5000 || B < n || B > n*C) throw std::runtime_error("invalid header");
        std::vector<double> H(n*n), q(n), x(n), g(n);
        in.read(reinterpret_cast<char*>(H.data()), 8*n*n);
        in.read(reinterpret_cast<char*>(q.data()), 8*n);
        in.read(reinterpret_cast<char*>(x.data()), 8*n);
        if (!in) throw std::runtime_error("truncated input");
        auto gradient = [&]() {
            for (int i=0; i<n; ++i) {
                g[i]=q[i];
                for (int j=0; j<n; ++j) g[i]+=H[i*n+j]*x[j];
            }
        };
        if (std::string(argv[3]) == "greedy") {
            std::fill(x.begin(), x.end(), 1.0);
            gradient();
            for (int64_t t=n; t<B; ++t) {
                int j=-1;
                double cost=INFINITY;
                for (int k=0; k<n; ++k) if (x[k]<C) {
                    double v=g[k]+0.5*H[k*n+k];
                    if (v<cost) {cost=v; j=k;}
                }
                if (j<0) throw std::runtime_error("greedy infeasible");
                ++x[j];
                for (int k=0; k<n; ++k) g[k]+=H[j*n+k];
            }
        }
        int64_t total=0;
        for (double v:x) {
            if (v!=std::round(v) || v<1 || v>C) throw std::runtime_error("invalid allocation");
            total+=static_cast<int64_t>(v);
        }
        if (total!=B) throw std::runtime_error("invalid budget");
        gradient();
        int moves=0;
        double best=0;
        for (; moves<100000; ++moves) {
            int a=-1,b=-1;
            auto scan=[&]() {
                best=0; a=b=-1;
                for (int i=0; i<n; ++i) if (x[i]>1)
                    for (int j=0; j<n; ++j) if (i!=j && x[j]<C) {
                        double d=g[j]-g[i]+0.5*(H[i*n+i]+H[j*n+j]-2*H[i*n+j]);
                        if (d<best) {best=d; a=i; b=j;}
                    }
            };
            scan();
            if (best>=-1e-11) {gradient(); scan(); if (best>=-1e-11) break;}
            --x[a]; ++x[b];
            for (int k=0; k<n; ++k) g[k]+=H[b*n+k]-H[a*n+k];
            if (moves%100==99) gradient();
        }
        std::ofstream out(argv[2], std::ios::binary);
        out.write(reinterpret_cast<const char*>(x.data()),8*n);
        if (!out) throw std::runtime_error("output failed");
        std::cout<<"{\"moves\":"<<moves<<",\"converged\":"<<(moves<100000?"true":"false")
                 <<",\"min_delta\":"<<best<<"}\n";
        return moves<100000 ? 0 : 2;
    } catch (const std::exception& e) {std::cerr<<e.what()<<"\n"; return 1;}
}
