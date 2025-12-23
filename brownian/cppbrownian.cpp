#include <iostream>
#include <random>
#include <matplot/matplot.h>
#include <iomanip>

std::vector<std::vector<double>> sval (std::vector<double> var, double avg, int steps, double dtime,double meanvalue){
    std::vector<std::vector<double>> bv;
    for (int i=0;i<var.size();++i){
        bv.push_back(std::vector<double> (steps,meanvalue));
    }
    for (int k=0;k<var.size();++k){
        for (int i=0; i<steps-1;++i){
            double m=avg-(std::sqrt(dtime)*std::sqrt(dtime))/2;
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<double> dist(0.0, 1.0);
            bv[k][i+1]=bv[k][i]+(m+dtime+var[k]*dist(gen));

        }
    }
    return bv;
}
int main() {
    double mu = 1.0;
    int n = 100;
    double dt = 0.01;
    
    std::vector<double> sigma(9);
    for (int i = 0; i < 9; ++i) {
        sigma[i] = 0.5 + i * (4.5 - 0.5) / 8.0;
    }
    

    std::vector<std::vector<double>> bv = sval(sigma, mu, n, dt, 10);

    size_t n_lines = bv.size();
    size_t steps   = bv[0].size();

    std::vector<double> x(steps);
    std::iota(x.begin(), x.end(), 0);

    matplot::hold(matplot::on);

    for (size_t i = 0; i < n_lines; ++i) {
        std::stringstream ss;
        ss << "σ = " << std::fixed << std::setprecision(2) << sigma[i];

        auto p = matplot::plot(x, bv[i]);
        p->marker("o");
        p->marker_size(6);
        p->line_width(1.5);
        p->display_name(ss.str());
    }

    matplot::legend();
    matplot::title("Brownian Motion with Different Variance");
    matplot::xlabel("Time");
    matplot::ylabel("Asset Value");

    matplot::show();
    return 0;
}
