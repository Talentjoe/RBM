#pragma once
#include <vector>
#include <random>
#include <Eigen/Dense>

struct RBM {
    int v_dim;
    int h_dim;
    Eigen::MatrixXd W;
    Eigen::VectorXd b_v;
    Eigen::VectorXd b_h;
    std::mt19937 gen;

    RBM(int v, int h);

    Eigen::VectorXd prob_h_given_v(const Eigen::VectorXd &v) const;
    Eigen::VectorXd prob_v_given_h(const Eigen::VectorXd &h) const;
    Eigen::VectorXd sample_bernoulli(const Eigen::VectorXd &p);

    void update(const Eigen::VectorXd &v0, double lr);
    double reconstruction_error(const Eigen::VectorXd &v0) const;

private:
    static double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
};

struct DBN {
    std::vector<RBM> layers;

    DBN(int v_dim, int h_dim);

    void add_layer(int h_dim);
    void update(const Eigen::VectorXd &v0, double lr);
    double reconstruction_error(const Eigen::VectorXd &v0) const;
    std::vector<std::vector<float>> get_hidden_layer_value(const std::vector<std::vector<float>> &v) const;
};
