#include <iostream>
#include <Eigen/Dense>
#include "./CudaNN/NNCore.cuh"
#include "./DataRead/readData.h"
#include "RBM/RBM.h"

using namespace std;
using namespace Eigen;

int main() {
    int v_dim = 784;
    int h_dim = 60;
    int epochs = 20;
    int n_samples = 1000;
    double lr = 0.05;

    DBN dbn(v_dim, h_dim);

    // 读取数据
    auto trainInData = readData::readData::readImageData("../Data/train-images.idx3-ubyte");
    auto trainOutData = readData::readData::readTagData("../Data/train-labels.idx1-ubyte");
    auto testInData  = readData::readData::readImageData("../Data/t10k-images.idx3-ubyte");
    auto testOutData = readData::readData::readTagData("../Data/t10k-labels.idx1-ubyte");

    vector<VectorXd> data;
    data.reserve(n_samples);
    for (int n = 0; n < n_samples; n++) {
        VectorXd v(v_dim);
        for (int i = 0; i < v_dim; i++)
            v(i) = trainInData[n][i] > 0.5;
        data.push_back(v);
    }

    // 预训练 DBN
    for (int epoch = 1; epoch <= epochs; epoch++) {
        double total_err = 0.0;
        for (auto &v : data) {
            dbn.update(v, lr);
            total_err += dbn.reconstruction_error(v);
        }
        cout << "Epoch " << epoch << " avg recon error = "
             << total_err / n_samples << endl;
    }

    // 提取特征
    vector<vector<float>> hiddenTrain = dbn.get_hidden_layer_value(trainInData);
    vector<vector<float>> hiddenTest  = dbn.get_hidden_layer_value(testInData);

    // 初始化 NN
    float Srate = 0.1;
    vector<NN::NNCore::LayerStructure> layerStructure = {
        {h_dim, ""},
        {128, "ReLU"},
        {64, "ReLU"},
        {10, "sigmoid"}
    };
    auto *nn = new NN::NNCore(layerStructure, Srate);

    int termsOfTrain = 1;
    for (int j = 0; j < termsOfTrain; j++) {
        cout << "Epoch: " << j << endl;
        vector<vector<float>> wrongData;
        vector<int> correctData;

        nn->train_with_retrain(hiddenTrain, trainOutData, wrongData, correctData, true);

        float acc = nn->test(hiddenTest, testOutData);
        nn->save("Model_Epoch" + to_string(j) + "_With_Rate_" + to_string(acc*100) + "%.module");

        Srate *= 0.75;
        nn->changeStudyRate(Srate);
    }

    delete nn;
    return 0;
}
/*
 Epoch 1 avg recon error = 42.7371
Epoch 2 avg recon error = 31.863
Epoch 3 avg recon error = 28.9144
Epoch 4 avg recon error = 27.3352
Epoch 5 avg recon error = 26.3493
Epoch 6 avg recon error = 25.5316
Epoch 7 avg recon error = 25.0211
Epoch 8 avg recon error = 24.6383
Epoch 9 avg recon error = 24.3691
Epoch 10 avg recon error = 24.0192
Epoch 11 avg recon error = 23.7713
Epoch 12 avg recon error = 23.5062
Epoch 13 avg recon error = 23.3033
Epoch 14 avg recon error = 23.0396
Epoch 15 avg recon error = 22.9455
Epoch 16 avg recon error = 22.7056
Epoch 17 avg recon error = 22.6213
Epoch 18 avg recon error = 22.4512
Epoch 19 avg recon error = 22.4203
Epoch 20 avg recon error = 22.2215
RESIZED And Inited
Epoch: 0
Progress:       0% Correct Percentage:       0%
Progress: 1.66667% Correct Percentage: 59.6404%
Progress: 3.33333% Correct Percentage:  70.015%
Progress:       5% Correct Percentage: 73.9087%
Progress: 6.66667% Correct Percentage: 77.0058%
Progress: 8.33333% Correct Percentage: 78.5243%
Progress:      10% Correct Percentage:   79.72%
Progress: 11.6667% Correct Percentage: 80.6885%
Progress: 13.3333% Correct Percentage: 81.3148%
Progress:      15% Correct Percentage: 81.6687%
Progress: 16.6667% Correct Percentage: 82.3318%
Progress: 18.3333% Correct Percentage: 82.9016%
Progress:      20% Correct Percentage: 83.1431%
Progress: 21.6667% Correct Percentage: 83.3013%
Progress: 23.3333% Correct Percentage: 83.5012%
Progress:      25% Correct Percentage: 83.4678%
Progress: 26.6667% Correct Percentage: 83.6635%
Progress: 28.3333% Correct Percentage: 83.8774%
Progress:      30% Correct Percentage: 83.9509%
Progress: 31.6667% Correct Percentage: 84.2271%
Progress: 33.3333% Correct Percentage: 84.4458%
Progress:      35% Correct Percentage: 84.5341%
Progress: 36.6667% Correct Percentage: 84.7643%
Progress: 38.3333% Correct Percentage: 84.7833%
Progress:      40% Correct Percentage: 84.9131%
Progress: 41.6667% Correct Percentage: 84.9966%
Progress: 43.3333% Correct Percentage: 85.1352%
Progress:      45% Correct Percentage: 85.2117%
Progress: 46.6667% Correct Percentage: 85.2755%
Progress: 48.3333% Correct Percentage: 85.4177%
Progress:      50% Correct Percentage: 85.5105%
Progress: 51.6667% Correct Percentage: 85.6198%
Progress: 53.3333% Correct Percentage: 85.6286%
Progress:      55% Correct Percentage:  85.652%
Progress: 56.6667% Correct Percentage: 85.7945%
Progress: 58.3333% Correct Percentage: 85.8718%
Progress:      60% Correct Percentage: 85.9643%
Progress: 61.6667% Correct Percentage: 86.0679%
Progress: 63.3333% Correct Percentage: 86.1135%
Progress:      65% Correct Percentage: 86.1952%
Progress: 66.6667% Correct Percentage: 86.2403%
Progress: 68.3333% Correct Percentage: 86.3223%
Progress:      70% Correct Percentage: 86.3765%
Progress: 71.6667% Correct Percentage: 86.3864%
Progress: 73.3333% Correct Percentage: 86.4617%
Progress:      75% Correct Percentage: 86.5314%
Progress: 76.6667% Correct Percentage: 86.5807%
Progress: 78.3333% Correct Percentage: 86.6237%
Progress:      80% Correct Percentage: 86.6669%
Progress: 81.6667% Correct Percentage:  86.737%
Progress: 83.3333% Correct Percentage: 86.7143%
Progress:      85% Correct Percentage: 86.7709%
Progress: 86.6667% Correct Percentage: 86.8676%
Progress: 88.3333% Correct Percentage: 86.9135%
Progress:      90% Correct Percentage: 86.9743%
Progress: 91.6667% Correct Percentage: 87.0202%
Progress: 93.3333% Correct Percentage: 87.0842%
Progress:      95% Correct Percentage: 87.1388%
Progress: 96.6667% Correct Percentage: 87.1933%
Progress: 98.3333% Correct Percentage: 87.3206%
Finish Training 60000 Data
With Accuracy: 87.435%
Progress:       0% Correct Percentage:     100%
Progress:      10% Correct Percentage:  88.012%
Progress:      20% Correct Percentage: 86.9565%
Progress:      30% Correct Percentage:  86.871%
Progress:      40% Correct Percentage: 86.7533%
Progress:      50% Correct Percentage: 87.0226%
Progress:      60% Correct Percentage: 87.8354%
Progress:      70% Correct Percentage: 88.3874%
Progress:      80% Correct Percentage: 88.8639%
Progress:      90% Correct Percentage: 89.5234%
Finish Testing 10000 Data
With Accuracy: 89.54%
*/