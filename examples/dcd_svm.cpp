/***
    This is the implementation of the l2 loss dual coordinate descent method by Hsieh. et. al 2008
    The codes follow the original implementation available in LibLinear with some modifications

    problem specification:
        f(a) = 0.5* \alpha^TQ\alpha + 0.5/C * \alpha^T\alpha + \beta^T\alpha
        lower_bound[i] <= \alpha[i]

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

class problem {
   public:
    double C;
    int n;  // number of features (appended 1 included)
    int l;
    int max_iter;
    husky::ObjList<ObjT>* train_set;
    husky::ObjList<ObjT>* test_set;
};

class solution {
   public:
    double dual_obj;
    DenseVector<double> alpha;
    DenseVector<double> w;
};

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

void initialize(problem* prob) {
    auto& train_set = husky::ObjListStore::create_objlist<ObjT>("train_set");
    auto& test_set = husky::ObjListStore::create_objlist<ObjT>("test_set");
    prob->train_set = &train_set;
    prob->test_set = &test_set;

    auto format_str = husky::Context::get_param("format");
    husky::lib::ml::DataFormat format;
    if (format_str == "libsvm") {
        format = husky::lib::ml::kLIBSVMFormat;
    } else if (format_str == "tsv") {
        format = husky::lib::ml::kTSVFormat;
    }

    // load data
    int n = husky::lib::ml::load_data(husky::Context::get_param("train"), train_set, format);
    n = std::max(n, husky::lib::ml::load_data(husky::Context::get_param("test"), test_set, format));

    // get model config parameters
    prob->C = std::stod(husky::Context::get_param("C"));
    prob->max_iter = std::stoi(husky::Context::get_param("max_iter"));

    auto& train_set_data = train_set.get_data();

    for (auto& labeled_point : train_set_data) {
        labeled_point.x.resize(n + 1);
        labeled_point.x.set(n, 1);
    }
    for (auto& labeled_point : test_set.get_data()) {
        labeled_point.x.resize(n + 1);
        labeled_point.x.set(n, 1);
    }

    n += 1;
    int l = train_set_data.size();
    prob->n = n;
    prob->l = l;

    husky::LOG_I << "number of samples: " + std::to_string(l);
    husky::LOG_I << "number of features: " + std::to_string(n);
}

template <bool is_sparse = true>
solution* dcd_svm_no_shrink_l2(problem* prob) {
    clock_t start = clock();

    // Declaration and Initialization
    int l = prob->l;
    int n = prob->n;
    double C = prob->C;

    const auto& labeled_point_vector = prob->train_set->get_data();

    double diag = 0.5 / C;
    double upper_bound = INF;

    double* QD = new double[l];
    int* index = new int[l];

    int iter = 0;
    const int max_iter = prob->max_iter;

    int i, k;

    double G, PG, PGmax_new;
    double diff, obj;

    DenseVector<double> alpha(l, 1.0 / l);
    DenseVector<double> lower_bound(l, 0.0);
    DenseVector<double> beta(l, -1.0);
    DenseVector<double> w(n, 0);

    for (i = 0; i < l; i++) {
        QD[i] = self_dot_product(labeled_point_vector[i].x) + diag;
        index[i] = i;
        w += labeled_point_vector[i].x * labeled_point_vector[i].y * alpha[i];
    }

    iter = 0;
    while (iter < max_iter) {
        PGmax_new = 0.0;

        for (i = 0; i < l; i++) {
            int j = i + std::rand() % (l - i);
            swap(index[i], index[j]);
        }

        bool optimal = true;
        for (k = 0; k < l; k++) {
            i = index[k];
            int yi = labeled_point_vector[i].y;
            auto& xi = labeled_point_vector[i].x;

            G = w.dot(xi) * yi + beta[i] + diag * alpha[i];
            if (EQUAL_VALUE(alpha[i], lower_bound[i])) {
                PG = std::min(0.0, G);
            } else {
                PG = G;
            }

            if (EQUAL_VALUE(PG, 0.0)) {
                continue;
            } else {
                optimal = false;
                double alpha_old = alpha[i];
                alpha[i] = std::min(std::max(alpha[i] - G / QD[i], lower_bound[i]), upper_bound);
                diff = yi * (alpha[i] - alpha_old);
                w += xi * diff;
            }
        }
        if (optimal) {
            break;
        }
        iter++;

        // objective
        obj = 0;
        for (i = 0; i < n; i++) {
            obj += w[i] * w[i];
        }
        for (i = 0; i < l; i++) {
            obj += alpha[i] * (alpha[i] * diag + 2 * beta[i]);
        }
        obj /= 2;
        husky::LOG_I << "[No Shrinking] iteration: " + std::to_string(iter) + ", objective: " + std::to_string(obj);
    }

    delete[] index;
    delete[] QD;

    solution* solu = new solution;
    solu->dual_obj = obj;
    solu->alpha = alpha;
    solu->w = w;

    clock_t end = clock();
    husky::LOG_I << "[No Shringking] time elapsed: " + std::to_string((double) (end - start) / CLOCKS_PER_SEC);

    return solu;
}

template <bool is_sparse = true>
solution* dcd_svm_shrink_l2(problem* prob) {
    clock_t start = clock();
    // Declaration and Initialization
    int l = prob->l;
    int n = prob->n;
    double C = prob->C;

    const auto& labeled_point_vector = prob->train_set->get_data();

    double diag = 0.5 / C;
    double upper_bound = INF;

    double* QD = new double[l];
    int* index = new int[l];

    int iter = 0;
    const int max_iter = prob->max_iter;

    double G, PG, PGmax_new, PGmin_new;
    double PGmax_old = INF;
    double PGmin_old = -INF;

    int i, k;
    int active_size = l;

    double diff, obj;

    DenseVector<double> alpha(l, 1.0 / l);
    DenseVector<double> lower_bound(l, 0.0);
    DenseVector<double> beta(l, -1.0);
    DenseVector<double> w(n, 0);

    for (i = 0; i < l; i++) {
        QD[i] = self_dot_product(labeled_point_vector[i].x) + diag;
        index[i] = i;
        w += labeled_point_vector[i].x * labeled_point_vector[i].y * alpha[i];
    }

    while (iter < max_iter) {
        PGmax_new = 0;
        PGmin_new = 0;

        for (i = 0; i < active_size; i++) {
            int j = i + std::rand() % (active_size - i);
            swap(index[i], index[j]);
        }

        for (k = 0; k < active_size; k++) {
            i = index[k];
            int yi = labeled_point_vector[i].y;
            auto& xi = labeled_point_vector[i].x;

            G = (w.dot(xi)) * yi + beta[i] + diag * alpha[i];

            PG = 0;
            if (EQUAL_VALUE(alpha[i], lower_bound[i])) {
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
                alpha[i] = std::min(std::max(alpha[i] - G / QD[i], lower_bound[i]), upper_bound);
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

        PGmax_old = PGmax_new;
        PGmin_old = PGmin_new;
        if (PGmax_old == 0) {
            PGmax_old = INF;
        }
        if (PGmin_old == 0) {
            PGmin_old = -INF;
        }
        // objective
        obj = 0;
        for (i = 0; i < n; i++) {
            obj += w[i] * w[i];
        }
        // assume that beta is local and alpha is global
        for (i = 0; i < l; i++) {
            obj += alpha[i] * (alpha[i] * diag + 2 * beta[i]);
        }
        obj /= 2;
        husky::LOG_I << "[With Shringking] iteration: " + std::to_string(iter) + ", objective: " + std::to_string(obj);
    }

    delete[] index;
    delete[] QD;

    solution* solu = new solution;
    solu->dual_obj = obj;
    solu->alpha = alpha;
    solu->w = w;

    clock_t end = clock();
    husky::LOG_I << "[With Shringking] time elapsed: " + std::to_string((double) (end - start) / CLOCKS_PER_SEC);

    return solu;
}

void evaluate(problem* prob, solution* solu) {
    const auto& alpha = solu->alpha;
    const auto& w = solu->w;
    const auto& test_set_data = prob->test_set->get_data();

    double error = 0;
    double indicator;
    for (auto& labeled_point : test_set_data) {
        indicator = w.dot(labeled_point.x);
        indicator *= labeled_point.y;
        if (indicator < 0) {
            error += 1;
        }
    }
    husky::LOG_I << "Classification accuracy on testing set with [C = " + std::to_string(prob->C) + "], " +
                        "[max_iter = " + std::to_string(prob->max_iter) + "], " + "[test set size = " +
                        std::to_string(test_set_data.size()) + "]: " +
                        std::to_string(1.0 - static_cast<double>(error / test_set_data.size()));
}

void job_runner() {
    problem* prob = new problem;

    initialize(prob);

    solution* solu_1 = dcd_svm_no_shrink_l2(prob);
    evaluate(prob, solu_1);

    solution* solu_2 = dcd_svm_shrink_l2(prob);
    evaluate(prob, solu_2);

    delete prob;
    delete solu_1;
    delete solu_2;
}

void init() {
    if (husky::Context::get_param("is_sparse") == "true") {
        job_runner();
    } else {
        husky::LOG_I << "Dense data format is not supported";
    }
}

int main(int argc, char** argv) {
    std::vector<std::string> args(
        {"hdfs_namenode", "hdfs_namenode_port", "train", "test", "C", "format", "is_sparse", "max_iter"});
    if (husky::init_with_args(argc, argv, args)) {
        husky::run_job(init);
        return 0;
    }
    return 1;
}
