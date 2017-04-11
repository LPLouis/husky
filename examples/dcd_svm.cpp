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

class problem {
public:
    double C;
    int n;          // number of features (appended 1 included)
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

void initialize(problem* problem_) {
    auto& train_set = husky::ObjListStore::create_objlist<ObjT>("train_set");
    auto& test_set = husky::ObjListStore::create_objlist<ObjT>("test_set");
    problem_->train_set = &train_set;
    problem_->test_set = &test_set;

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
    problem_->C = std::stod(husky::Context::get_param("C"));
    problem_->max_iter = std::stoi(husky::Context::get_param("max_iter"));

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
    problem_->n = n;
    problem_->l = l;

    husky::LOG_I << "number of samples: " + std::to_string(l);
    husky::LOG_I << "number of features: " + std::to_string(n);
}

template <bool is_sparse = true>
solution* dcd_svm_no_shrink_l2(problem* problem_) {
    // Declaration and Initialization
    int l = problem_->l;
    int n = problem_->n;
    double C = problem_->C;

    const auto& labeled_point_vector = problem_->train_set->get_data();

    double diag = 0.5 / C;
    double UB = INF;

    double* QD = new double[l];
    int* index = new int[l];

    int iter = 0;
    const int max_iter = problem_->max_iter;

    int i, k;

    double G, PG, PGmax_new;
    double diff, obj;

    DenseVector<double> alpha(l, 1.0 / l);
    DenseVector<double> LB(l, 0.0);
    DenseVector<double> beta(l, -1.0);
    DenseVector<double> w(n, 0);

    for (i = 0; i < l; i++) {
        QD[i] = self_dot_product(labeled_point_vector[i].x) + diag;
        index[i] = i;
        w += labeled_point_vector[i].x * labeled_point_vector[i].y * alpha[i];
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
            int yi = labeled_point_vector[i].y;
            auto& xi = labeled_point_vector[i].x;

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
        obj = 0;
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

    solution* solution_ = new solution;
    solution_->dual_obj = obj;
    solution_->alpha = alpha;
    solution_->w = w;

    return solution_;
}

template <bool is_sparse = true>
solution* dcd_svm_shrink_l2(problem* problem_) {
    // Declaration and Initialization
    int l = problem_->l;
    int n = problem_->n;
    double C = problem_->C;

    const auto& labeled_point_vector = problem_->train_set->get_data();

    double diag = 0.5 / C;
    double UB = INF;

    double* QD = new double[l];
    int* index = new int[l];

    int iter = 0;
    const int max_iter = problem_->max_iter;

    double G, PG, PGmax_new, PGmin_new;
    double PGmax_old = INF;
    double PGmin_old = -INF;

    int i, k;
    int active_size = l;

    double diff, obj;

    DenseVector<double> alpha(l, 1.0 / l);
    DenseVector<double> LB(l, 0.0);
    DenseVector<double> beta(l, -1.0);
    DenseVector<double> w(n, 0);

    for (i = 0; i < l; i++) {
        QD[i] = self_dot_product(labeled_point_vector[i].x) + diag;
        index[i] = i;
        w += labeled_point_vector[i].x * labeled_point_vector[i].y * alpha[i];
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
            int yi = labeled_point_vector[i].y;
            auto& xi = labeled_point_vector[i].x;

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
        obj = 0;
        for (i = 0; i < n; i++) {
            obj += w[i] * w[i];
        }
        // assume that beta is local and alpha is global
        for (i = 0; i < l; i++) {
            obj += alpha[i] * (alpha[i] * diag + 2 * beta[i]);
        }
        obj /= 2;
        husky::LOG_I << "iteration: " + std::to_string (iter) + ", objective: " + std::to_string(obj);
    }

    delete[] index;
    delete[] QD;

    solution* solution_ = new solution;
    solution_->dual_obj = obj;
    solution_->alpha = alpha;
    solution_->w = w;

    return solution_;
}

void evaluate(problem* problem_, solution* solution_) {
    const auto& alpha = solution_->alpha;
    const auto& w = solution_->w;
    const auto& test_set_data = problem_->test_set->get_data();

    double error = 0;
    double indicator;
    for (auto& labeled_point : test_set_data) {
        indicator = w.dot(labeled_point.x);
        indicator *= labeled_point.y;
        if (indicator < 0) {
            error += 1;
        }
    }
    husky::LOG_I << "Classification accuracy on testing set with [C = " + 
                        std::to_string(problem_->C) + "], " +
                        "[max_iter = " + std::to_string(problem_->max_iter) + "], " +
                        "[test set size = " + std::to_string(test_set_data.size()) + "]: " +
                        std::to_string(1.0 - static_cast<double>(error / test_set_data.size()));  
}

void job_runner() {
    problem* problem_ = new problem;

    initialize(problem_);

    // solution* ret1 = dcd_svm_no_shrink_l2(problem_);
    // evaluate(problem_, ret1);

    solution* ret2 = dcd_svm_shrink_l2(problem_);
    evaluate(problem_, ret2);

    delete problem_;
    // delete ret1;
    delete ret2;
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
