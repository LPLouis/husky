/***
    This is the implementation of the l2 loss dual coordinate descent method by Hsieh. et. al 2008
    The codes follow the original implementation available in LibLinear with some modifications

    problem specification:
        f(a) = 0.5* \alpha^TQ\alpha + 0.5/C * \alpha^T\alpha + \beta^T\alpha
        LB[i] <= \alpha[i]

    Setting \beta = [-1, -1, -1, ..., -1] gives the standard l2 loss SVM problem
    Note this can only work on single machine
    Note in this implementation w^T = [w^T b] x_i^T = [x_i^T 1]

    parameters:
    train
    type: string
    info: path to training data, LIBLINEAR format

    test
    type: string
    info: path to testing data, LIBLINEAR format

    format
    type: string
    info: the data format of the input file: libsvm/tsv

    configuration example:
    train=/path/to/training/data
    test=/path/to/testing/data
    format=libsvm
    C=1
    is_sparse=true
    max_iter=200
***/

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include "boost/tokenizer.hpp"

#include "core/engine.hpp"
#include "lib/ml/data_loader.hpp"
#include "lib/ml/feature_label.hpp"
#include "lib/ml/parameter.hpp"

using husky::lib::Aggregator;
using husky::lib::AggregatorFactory;
using husky::lib::DenseVector;
using husky::lib::SparseVector;
using ObjT = husky::lib::ml::LabeledPointHObj<double, double, true>;

#define EQUAL_VALUE(a, b) (a - b < 1.0e-12) && (a - b > -1.0e-12)
#define NOT_EQUAL(a, b) (a - b > 1.0e-12) || (a - b < -1.0e-12)
#define INF std::numeric_limits<double>::max()
#define EPS 1.0e-6

template <typename T>
inline void swap(T& a, T& b) {
    T t = a;
    a = b;
    b = t;
}

template <typename T, bool is_sparse>
T self_dot_product(const husky::lib::Vector<T, is_sparse>& v) {
    T ret = 0;
    for (auto it = v.begin_value(); it != v.end_value(); it++) {
        ret += (*it) * (*it);
    }
    return ret;
}

template <bool is_sparse = true>
void dcd_svm_no_shrinking_l2(int num_features, double C, int max_iter, DenseVector<double>& alpha,
                             const DenseVector<double>& LB, const DenseVector<double>& beta,
                             husky::ObjList<ObjT>& train_set) {
    // Declaration and Initialization
    const auto& xy_vector = train_set.get_data();
    int l = xy_vector.size();
    int n = num_features;

    double diag = 0.5 / C;
    double UB = INF;

    double* QD = new double[l];
    int* index = new int[l];

    int iter = 200;
    int i, k;

    double G, PG, PGmax_new;
    double diff;

    DenseVector<double> w(n, 0);

    for (i = 0; i < l; i++) {
        QD[i] = self_dot_product(xy_vector[i].x) + diag;
        index[i] = i;
        w += xy_vector[i].x * xy_vector[i].y * alpha[i];
    }

    iter = 0;
    while (iter < max_iter) {
        auto start = std::chrono::steady_clock::now();

        PGmax_new = 0.0;

        for (i = 0; i < l; i++) {
            int j = i + std::rand() % (l - i);
            swap(index[i], index[j]);
        }

        bool optimal = true;
        for (k = 0; k < l; k++) {
            i = index[k];
            int yi = xy_vector[i].y;
            auto& xi = xy_vector[i].x;

            G = w.dot(xi) * yi + beta[i] + diag * alpha[i];
            if (EQUAL_VALUE(alpha[i], LB[i])) {
                PG = std::min(0.0, G);
            } else {
                PG = G;
            }

            if (EQUAL_VALUE(PG, 0.0)) {
                continue;
            } else {
                optimal = false;
                double alpha_old = alpha[i];
                alpha[i] = std::min(std::max(alpha[i] - G / QD[i], LB[i]), UB);
                diff = yi * (alpha[i] - alpha_old);
                w += xi * diff;
            }
        }
        if (optimal) {
            break;
        }
        iter++;
        auto end = std::chrono::steady_clock::now();

        // objective
        double obj = 0;
        for (i = 0; i < n; i++) {
            obj += w[i] * w[i];
        }
        for (i = 0; i < l; i++) {
            obj += alpha[i] * (alpha[i] * diag + 2 * beta[i]);
        }
        obj /= 2;
    }

    delete[] index;
    delete[] QD;
}

template <bool is_sparse = true>
void dcd_svm_with_shrinking_l2(int num_features, double C, int max_iter, DenseVector<double>& alpha,
                               const DenseVector<double>& LB, const DenseVector<double>& beta,
                               husky::ObjList<ObjT>& train_set) {
    // Declaration and Initialization
    const auto& xy_vector = train_set.get_data();
    int l = xy_vector.size();
    int n = num_features;

    double diag = 0.5 / C;
    double UB = INF;

    double* QD = new double[l];
    int* index = new int[l];

    int iter = 0;

    double G, PG, PGmax_new, PGmin_new;
    double PGmax_old = INF;
    double PGmin_old = -INF;

    int i, k;
    int active_size = l;

    double diff;

    DenseVector<double> w(n, 0);

    for (i = 0; i < l; i++) {
        QD[i] = self_dot_product(xy_vector[i].x) + diag;
        index[i] = i;
        w += xy_vector[i].x * xy_vector[i].y * alpha[i];
    }

    while (iter < max_iter) {
        auto start = std::chrono::steady_clock::now();

        PGmax_new = 0;
        PGmin_new = 0;

        for (i = 0; i < active_size; i++) {
            int j = i + std::rand() % (active_size - i);
            swap(index[i], index[j]);
        }

        for (k = 0; k < active_size; k++) {
            i = index[k];
            int yi = xy_vector[i].y;
            auto& xi = xy_vector[i].x;

            G = (w.dot(xi)) * yi + beta[i] + diag * alpha[i];

            PG = 0;
            if (EQUAL_VALUE(alpha[i], LB[i])) {
                if (G > PGmax_old) {
                    active_size--;
                    swap(index[k], index[active_size]);
                    k--;
                    continue;
                } else if (G < 0) {
                    PG = G;
                    PGmin_new = std::min(PGmin_new, PG);
                }
            } else {
                PG = G;
                PGmax_new = std::max(PGmax_new, PG);
                PGmin_new = std::min(PGmin_new, PG);
            }

            if (EQUAL_VALUE(PG, 0.0)) {
                continue;
            } else {
                double alpha_old = alpha[i];
                alpha[i] = std::min(std::max(alpha[i] - G / QD[i], LB[i]), UB);
                diff = yi * (alpha[i] - alpha_old);
                w += xi * diff;
            }
        }

        iter++;
        if (PGmax_new - PGmin_new <= EPS) {
            if (active_size == l) {
                break;
            } else {
                active_size = l;
                PGmax_old = INF;
                PGmin_old = -INF;
                continue;
            }
        }
        auto end = std::chrono::steady_clock::now();

        PGmax_old = PGmax_new;
        PGmin_old = PGmin_new;
        if (PGmax_old == 0) {
            PGmax_old = INF;
        }
        if (PGmin_old == 0) {
            PGmin_old = -INF;
        }
        // objective
        double obj = 0;
        for (i = 0; i < n; i++) {
            obj += w[i] * w[i];
        }
        // assume that beta is local and alpha is global
        for (i = 0; i < l; i++) {
            obj += alpha[i] * (alpha[i] * diag + 2 * beta[i]);
        }
        obj /= 2;
    }

    delete[] index;
    delete[] QD;
}

void job_runner() {
    auto& train_set = husky::ObjListStore::create_objlist<ObjT>("train_set");
    auto& test_set = husky::ObjListStore::create_objlist<ObjT>("test_set");

    auto format_str = husky::Context::get_param("format");
    husky::lib::ml::DataFormat format;
    if (format_str == "libsvm") {
        format = husky::lib::ml::kLIBSVMFormat;
    } else if (format_str == "tsv") {
        format = husky::lib::ml::kTSVFormat;
    }

    // load data
    int num_samples = 0;
    int num_features = husky::lib::ml::load_data(husky::Context::get_param("train"), train_set, format);
    num_features =
        std::max(num_features, husky::lib::ml::load_data(husky::Context::get_param("test"), test_set, format));

    // get model config parameters
    double C = std::stod(husky::Context::get_param("C"));
    int max_iter = std::stoi(husky::Context::get_param("max_iter"));

    // append 1 to the end of every sample
    for (auto& labeled_point : train_set.get_data()) {
        labeled_point.x.resize(num_features + 1);
        labeled_point.x.set(num_features, 1);
        num_samples += 1;
    }

    for (auto& labeled_point : test_set.get_data()) {
        labeled_point.x.resize(num_features + 1);
        labeled_point.x.set(num_features, 1);
    }

    num_features += 1;
    husky::LOG_I << "number of samples: " + std::to_string(num_samples);
    husky::LOG_I << "number of features: " + std::to_string(num_features);

    // parameter specification
    DenseVector<double> alpha(num_samples, 1.0 / num_samples);
    DenseVector<double> lower_bound(num_samples, 0.0);
    DenseVector<double> beta(num_samples, -1.0);
    DenseVector<double> w(num_features, 0);

    int num_test_samples = test_set.get_data().size();
    husky::LOG_I << "------------------l2 loss dual coordinate descent with no shringking------------------";
    for (C = 0.01; C < 11.0; C *= 10) {
        auto start = std::chrono::steady_clock::now();
        dcd_svm_no_shrinking_l2(num_features, C, max_iter, alpha, lower_bound, beta, train_set);

        int i = 0;
        for (auto& labeled_point : train_set.get_data()) {
            w += labeled_point.x * alpha[i++] * labeled_point.y;
        }

        double error = 0;
        double indicator;
        for (auto& labeled_point : test_set.get_data()) {
            indicator = w.dot(labeled_point.x);
            indicator *= labeled_point.y;
            if (indicator < 0) {
                error += 1;
            }
        }
        husky::LOG_I << "Classification accuracy on testing set with C = " + std::to_string(C) + " : " +
                            std::to_string(1.0 - static_cast<double>(error / num_test_samples));
        auto end = std::chrono::steady_clock::now();
        husky::LOG_I << "Time elapsed: " +
                            std::to_string(
                                std::chrono::duration_cast<std::chrono::duration<float>>(end - start).count());
    }

    husky::LOG_I << "------------------l2 loss dual coordinate descent with shringking------------------";
    for (C = 0.01; C < 11.0; C *= 10) {
        auto start = std::chrono::steady_clock::now();
        dcd_svm_with_shrinking_l2(num_features, C, max_iter, alpha, lower_bound, beta, train_set);

        int i = 0;
        for (auto& labeled_point : train_set.get_data()) {
            w += labeled_point.x * alpha[i++] * labeled_point.y;
        }

        double error = 0;
        double indicator;
        for (auto& labeled_point : test_set.get_data()) {
            indicator = w.dot(labeled_point.x);
            indicator *= labeled_point.y;
            if (indicator < 0) {
                error += 1;
            }
        }
        husky::LOG_I << "Classification accuracy on testing set with C = " + std::to_string(C) + " : " +
                            std::to_string(1.0 - static_cast<double>(error / num_test_samples));
        auto end = std::chrono::steady_clock::now();
        husky::LOG_I << "Time elapsed: " +
                            std::to_string(
                                std::chrono::duration_cast<std::chrono::duration<float>>(end - start).count());
    }
}

void init() {
    if (husky::Context::get_param("is_sparse") == "true") {
        job_runner();
    } else {
        husky::LOG_I << "Dense data format is not supported";
    }
}

int main(int argc, char** argv) {
    std::vector<std::string> args({"hdfs_namenode", "hdfs_namenode_port", "train", "test", "C", "format", "is_sparse", "max_iter"});
    if (husky::init_with_args(argc, argv, args)) {
        husky::run_job(init);
        return 0;
    }
    return 1;
}
