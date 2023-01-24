#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;

int main() {
    // create data matrix
    MatrixXd data(3, 2);
    data << 1, 2,
            3, 4,
            5, 6;

    // create labels vector
    VectorXd labels(3);
    labels << 1, -1, 1;

    // create weight vector with random values
    VectorXd weights(2);
    weights = VectorXd::Random(2);

    // set learning rate
    double learning_rate = 0.1;

    // set number of iterations
    int iterations = 100;

    // train model
    for (int i = 0; i < iterations; i++) {
        for (int j = 0; j < data.rows(); j++) {
            VectorXd x = data.row(j);
            double y = labels(j);

            double prediction = x.dot(weights);
            double error = y - prediction;

            weights = weights + learning_rate * error * x;
        }
    }

    // print weights
    std::cout << "Weights: " << weights << std::endl;

    return 0;
}
