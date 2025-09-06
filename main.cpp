// rbm.cpp
// g++ rbm.cpp -std=c++17 -O2 -I /usr/include/eigen3 -o rbm
#include <iostream>
#include <random>
#include <Eigen/Dense>
#include "./DataRead/readData.h"

using namespace Eigen;

struct RBM {
    int v_dim;  // 可见层单元数
    int h_dim;  // 隐含层单元数
    MatrixXd W; // [v_dim, h_dim]
    VectorXd b_v; // [v_dim]
    VectorXd b_h; // [h_dim]
    std::mt19937 gen;

    RBM(int v, int h) : v_dim(v), h_dim(h), gen(std::random_device{}()) {
        std::normal_distribution<> nd(0.0, 0.01);
        W = MatrixXd(v_dim, h_dim).unaryExpr([&](double){ return nd(gen); });
        b_v = VectorXd::Zero(v_dim);
        b_h = VectorXd::Zero(h_dim);
    }

    // sigmoid
    static double sigmoid(double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }

    // 给定 v，计算 p(h=1|v)
    VectorXd prob_h_given_v(const VectorXd &v) {
        return ((W.transpose() * v) + b_h).unaryExpr([](double x){ return sigmoid(x); });
    }

    // 给定 h，计算 p(v=1|h)
    VectorXd prob_v_given_h(const VectorXd &h) {
        return ((W * h) + b_v).unaryExpr([](double x){ return sigmoid(x); });
    }

    // 伯努利采样
    VectorXd sample_bernoulli(const VectorXd &p) {
        VectorXd out(p.size());
        std::uniform_real_distribution<> ud(0.0, 1.0);
        for(int i=0; i<p.size(); i++) out(i) = (ud(gen) < p(i)) ? 1.0 : 0.0;
        return out;
    }

    // CD-1 更新
    void update(const VectorXd &v0, double lr) {
        // 正相
        VectorXd ph0 = prob_h_given_v(v0);
        VectorXd h0 = sample_bernoulli(ph0);

        // 负相
        VectorXd pv1 = prob_v_given_h(h0);
        VectorXd v1 = sample_bernoulli(pv1);
        VectorXd ph1 = prob_h_given_v(v1);

        // 梯度
        MatrixXd pos_grad = v0 * ph0.transpose(); // 外积
        MatrixXd neg_grad = v1 * ph1.transpose();

        // 更新参数
        W += lr * (pos_grad - neg_grad);
        b_v += lr * (v0 - v1);
        b_h += lr * (ph0 - ph1);
    }

    // 计算重构误差
    double reconstruction_error(const VectorXd &v0) {
        VectorXd ph = prob_h_given_v(v0);
        VectorXd pv = prob_v_given_h(ph);
        return (v0 - pv).squaredNorm();
    }
};

struct DBN {
    std::vector<RBM> layers;

    DBN(int v, int h) {
        layers.push_back(RBM(v, h));
    }

    void add_layer(int h) {
        layers.push_back(RBM(layers[layers.size()-1].h_dim, h));
    }

    void update(const VectorXd &v0, double lr) {
        auto cur_v = v0;
        for (int i = 0; i < layers.size()-1; i++) {
            cur_v = layers[i].prob_h_given_v(cur_v);
        }
        layers[layers.size()-1].update(cur_v, lr);
    }

    double reconstruction_error(const VectorXd &v0) {
        auto cur_v = v0;
        for (int i = 0; i < layers.size(); i++) {
            cur_v = layers[i].prob_h_given_v(cur_v);
        }
        for (int i = layers.size()-1; i >= 0; i--) {
            cur_v = layers[i].prob_v_given_h(cur_v);
        }
        return (v0 - cur_v).squaredNorm();
    }
};

int main() {
    int v_dim = 784;   // 可见单元数
    int h_dim = 10;   // 隐含单元数
    int epochs = 20;
    int n_samples = 1000;
    double lr = 0.05;

    DBN rbm(v_dim, h_dim);

    auto testInData = readData::readData::readImageData("../Data/t10k-images.idx3-ubyte");
    auto testOutData = readData::readData::readTagData("../Data/t10k-labels.idx1-ubyte");

    std::vector<VectorXd> data;
    for(int n=0; n<n_samples; n++) {
        VectorXd v(v_dim);
        for(int i=0; i<v_dim; i++) {
            v(i) = testInData[n][i] > 0.5;
        }
        data.push_back(v);
    }

    // 训练
    for(int epoch=1; epoch<=epochs; epoch++) {
        double total_err = 0.0;
        for(auto &v : data) {
            rbm.update(v, lr);
            total_err += rbm.reconstruction_error(v);
        }
        std::cout << "Epoch " << epoch << " avg recon error = "
                  << total_err / n_samples << std::endl;
    }
    //
    // rbm.add_layer(30);
    //
    // for(int epoch=1; epoch<=epochs+30; epoch++) {
    //     double total_err = 0.0;
    //     for(auto &v : data) {
    //         rbm.update(v, lr);
    //         total_err += rbm.reconstruction_error(v);
    //     }
    //     std::cout << "Epoch " << epoch << " avg recon error = "
    //               << total_err / n_samples << std::endl;
    // }

    // for(int i = 0; i < 100; i++) {
    //
    //     auto res = rbm.prob_h_given_v(data[i]);
    //     std::cout<<res<<std::endl;
    //     std::cout<<testOutData[i]<<std::endl;
    //
    // }

    return 0;
}
