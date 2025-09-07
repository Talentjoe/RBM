#include "RBM.h"

using namespace Eigen;
using namespace std;

RBM::RBM(int v, int h) : v_dim(v), h_dim(h), gen(random_device{}()) {
    normal_distribution<> nd(0.0, 0.01);
    W = MatrixXd(v_dim, h_dim).unaryExpr([&](double){ return nd(gen); });
    b_v = VectorXd::Zero(v_dim);
    b_h = VectorXd::Zero(h_dim);
}

VectorXd RBM::prob_h_given_v(const VectorXd &v) const {
    return ((W.transpose() * v) + b_h).unaryExpr(&RBM::sigmoid);
}

VectorXd RBM::prob_v_given_h(const VectorXd &h) const {
    return ((W * h) + b_v).unaryExpr(&RBM::sigmoid);
}

VectorXd RBM::sample_bernoulli(const VectorXd &p) {
    VectorXd out(p.size());
    uniform_real_distribution<> ud(0.0, 1.0);
    for (int i = 0; i < p.size(); i++)
        out(i) = (ud(gen) < p(i)) ? 1.0 : 0.0;
    return out;
}

void RBM::update(const VectorXd &v0, double lr) {
    VectorXd ph0 = prob_h_given_v(v0);
    VectorXd h0 = sample_bernoulli(ph0);

    VectorXd pv1 = prob_v_given_h(h0);
    VectorXd v1 = sample_bernoulli(pv1);
    VectorXd ph1 = prob_h_given_v(v1);

    W += lr * (v0 * ph0.transpose() - v1 * ph1.transpose());
    b_v += lr * (v0 - v1);
    b_h += lr * (ph0 - ph1);
}

double RBM::reconstruction_error(const VectorXd &v0) const {
    VectorXd ph = prob_h_given_v(v0);
    VectorXd pv = prob_v_given_h(ph);
    return (v0 - pv).squaredNorm();
}

// ---------------- DBN ----------------

DBN::DBN(int v_dim, int h_dim) {
    layers.emplace_back(v_dim, h_dim);
}

void DBN::add_layer(int h_dim) {
    layers.emplace_back(layers.back().h_dim, h_dim);
}

void DBN::update(const VectorXd &v0, double lr) {
    VectorXd cur_v = v0;
    for (size_t i = 0; i < layers.size() - 1; i++)
        cur_v = layers[i].prob_h_given_v(cur_v);
    layers.back().update(cur_v, lr);
}

double DBN::reconstruction_error(const VectorXd &v0) const {
    VectorXd cur_v = v0;
    for (const auto &layer : layers)
        cur_v = layer.prob_h_given_v(cur_v);
    for (auto it = layers.rbegin(); it != layers.rend(); ++it)
        cur_v = it->prob_v_given_h(cur_v);
    return (v0 - cur_v).squaredNorm();
}

vector<vector<float>> DBN::get_hidden_layer_value(const vector<vector<float>> &v) const {
    vector<vector<float>> hidden_layer_value;
    hidden_layer_value.reserve(v.size());

    for (const auto &element : v) {
        VectorXd v0 = Map<const VectorXf>(element.data(), element.size()).cast<double>();
        for (const auto &layer : layers)
            v0 = layer.prob_h_given_v(v0);

        vector<float> result(v0.size());
        for (int i = 0; i < v0.size(); ++i)
            result[i] = static_cast<float>(v0[i]);

        hidden_layer_value.push_back(result);
    }
    return hidden_layer_value;
}
