/***
    This is the implementation of the l2 loss Distributed Box-Constrained Quadratic Optimization method  by Lee. et.
    al 2015
    The codes follow the original implementation available in Distributed LibLinear with some modifications

    problem specification:
        f(a) = 0.5* \alpha^TQ\alpha + 0.5/C * \alpha^T\alpha + 1^T\alpha
        LB[i] <= \alpha[i]

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
    max_inn_iter=10

***/

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include "boost/tokenizer.hpp"

#include "core/engine.hpp"
#include "core/utils.hpp"
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

template <typename T, bool is_sparse = true>
T self_dot_product(const husky::lib::Vector<T, is_sparse>& v) {
    T res = 0;
    for (auto it = v.begin_value(); it != v.end_value(); it++) {
        res += (*it) * (*it);
    }
    return res;
}

void get_local_worker_info(int& num_workers, int& tid, int& n, double& C, int& N, int& l, int& index_low,
                           int& index_high, husky::ObjList<ObjT>& train_set, husky::ObjList<ObjT>& test_set,
                           int& max_iter, int& max_inn_iter) {
    // worker info
    num_workers = husky::Context::get_num_workers();
    tid = husky::Context::get_global_tid();

    std::string format_str = husky::Context::get_param("format");
    husky::lib::ml::DataFormat format;
    if (format_str == "libsvm") {
        format = husky::lib::ml::kLIBSVMFormat;
    } else if (format_str == "tsv") {
        format = husky::lib::ml::kTSVFormat;
    }

    // load data
    n = husky::lib::ml::load_data(husky::Context::get_param("train"), train_set, format);
    n = std::max(n, husky::lib::ml::load_data(husky::Context::get_param("test"), test_set, format));

    // get model config parameters
    C = std::stod(husky::Context::get_param("C"));
    max_iter = std::stoi(husky::Context::get_param("max_iter"));
    max_inn_iter = std::stoi(husky::Context::get_param("max_inn_iter"));

    // initialize parameters
    husky::lib::ml::ParameterBucket<double> param_list(n + 1);  // scalar b and vector w

    // get the number of global records
    Aggregator<std::vector<int>> vec_local_samples(
        std::vector<int>(num_workers, 0),
        [](std::vector<int>& a, const std::vector<int>& b) {
            for (int i = 0; i < a.size(); i++)
                a[i] += b[i];
        },
        [num_workers](std::vector<int>& v) { v = std::move(std::vector<int>(num_workers, 0)); });

    vec_local_samples.update_any([&train_set, tid](std::vector<int>& v) { v[tid] = train_set.get_size(); });
    AggregatorFactory::sync();
    vec_local_samples.inactivate();

    auto& num_samples = vec_local_samples.get_value();
    N = 0;
    std::vector<int> sample_distrib_info(num_workers, 0);
    for (int i = 0; i < num_samples.size(); i++) {
        N += num_samples[i];
        sample_distrib_info[i] = N;
    }

    // A worker holds samples [low, high)
    if (tid == 0) {
        index_low = 0;
    } else {
        index_low = sample_distrib_info[tid - 1];
    }
    if (tid == (num_workers - 1)) {
        index_high = N;
    } else {
        index_high = sample_distrib_info[tid];
    }
    l = index_high - index_low;

    if (tid == 0) {
        husky::LOG_I << "Number of samples: " + std::to_string(N);
        husky::LOG_I << "Number of features: " + std::to_string(n + 1);
    }
    return;
}

void bqo_svm() {
    int i, k;

    auto& train_set = husky::ObjListStore::create_objlist<ObjT>("train_set");
    auto& test_set = husky::ObjListStore::create_objlist<ObjT>("test_set");

    // get worker info
    double C;
    int num_workers, tid, n, N, l, index_low, index_high, max_iter, max_inn_iter;
    get_local_worker_info(num_workers, tid, n, C, N, l, index_low, index_high, train_set, test_set, max_iter,
                          max_inn_iter);

    auto& train_set_data = train_set.get_data();
    auto& test_set_data = test_set.get_data();

    // append 1 to the end of every sample
    for (auto& labeled_point : train_set_data) {
        labeled_point.x.resize(n + 1);
        labeled_point.x.set(n, 1);
    }
    for (auto& labeled_point : test_set_data) {
        labeled_point.x.resize(n + 1);
        labeled_point.x.set(n, 1);
    }
    n += 1;

    int active_size, iter_out, inn_iter;
    double diff, primal, gap, opt_eta, grad_alpha_delta_alpha, a_Q_a;
    double G, PG, PGmax_new, PGmin_new;
    double PGmax_old = INF;
    double PGmin_old = -INF;
    double diag = 0.5 / C;
    double old_primal = INF;
    double dual = 0;
    double pdw = 0;
    double initial_gap = C * N;

    double* QD = new double[l];
    int* index = new int[l];

    for (int i = 0; i < l; i++) {
        QD[i] = self_dot_product(train_set_data[i].x) + diag;
        index[i] = i;
    }

    // for now assume we initialize alpha and weight to 0
    DenseVector<double> alpha(l, 0.0);
    DenseVector<double> orig_alpha(l);
    DenseVector<double> delta_alpha(l);
    DenseVector<double> lower_bound(l, 0.0);
    DenseVector<double> grad_alpha(l, -1.0);
    DenseVector<double> weight(n, 0.0);
    DenseVector<double> best_weight(n, 0.0);
    DenseVector<double> orig_weight(n);

    DenseVector<double> orig_param_server_weight(n, 0.0);
    husky::lib::ml::ParameterBucket<double> param_server_weight;
    param_server_weight.init(n, 0.0);

    Aggregator<double> alpha_delta_alpha(0.0, [](double& a, const double& b) { a += b; });
    alpha_delta_alpha.to_reset_each_iter();
    Aggregator<double> et_delta_alpha(0.0, [](double& a, const double& b) { a += b; });
    et_delta_alpha.to_reset_each_iter();
    Aggregator<double> delta_alpha_square(0.0, [](double& a, const double& b) { a += b; });
    delta_alpha_square.to_reset_each_iter();
    Aggregator<double> eta(INF, [](double& a, const double& b) { a = std::min(a, b); }, [](double& a) { a = INF; });
    eta.to_reset_each_iter();

    auto start = std::chrono::steady_clock::now();

    iter_out = 0;
    while (iter_out < max_iter) {
        // get parameters for local svm solver
        orig_weight = weight;
        orig_alpha = alpha;

        // run local svm solver to get local delta alpha
        inn_iter = 0;
        active_size = l;

        while (inn_iter < max_inn_iter) {
            PGmax_new = 0;
            PGmin_new = 0;

            for (i = 0; i < active_size; i++) {
                int j = i + std::rand() % (active_size - i);
                swap(index[i], index[j]);
            }

            for (k = 0; k < active_size; k++) {
                i = index[k];
                int yi = train_set_data[i].y;
                auto& xi = train_set_data[i].x;

                G = (weight.dot(xi)) * yi + -1 + diag * alpha[i];

                PG = 0;
                if (EQUAL_VALUE(alpha[i], 0)) {
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
                    alpha[i] = std::min(std::max(alpha[i] - G / QD[i], 0.0), INF);
                    diff = yi * (alpha[i] - alpha_old);
                    weight += xi * diff;
                }
            }

            inn_iter++;
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
        }

        delta_alpha = alpha - orig_alpha;
        for (i = 0; i < l; i++) {
            et_delta_alpha.update(delta_alpha[i]);
            delta_alpha_square.update(delta_alpha[i] * delta_alpha[i]);
            alpha_delta_alpha.update(delta_alpha[i] * orig_alpha[i]);
            if (delta_alpha[i] < 0) {
                eta.update(-1 * orig_alpha[i] / delta_alpha[i]);
            } else {
                eta.update(INF);
            }
        }
        for (i = 0; i < n; i++) {
            param_server_weight.update(i, weight[i] - orig_weight[i] - orig_param_server_weight[i] / num_workers);
        }
        AggregatorFactory::sync();

        orig_param_server_weight = param_server_weight.get_all_param();
        // get step size
        grad_alpha_delta_alpha = orig_weight.dot(orig_param_server_weight) + 1.0 / C * alpha_delta_alpha.get_value() -
                                 et_delta_alpha.get_value();
        a_Q_a = orig_param_server_weight.dot(orig_param_server_weight) + 1.0 / C * delta_alpha_square.get_value();
        if (grad_alpha_delta_alpha > 0) {
            weight = best_weight;
            break;
        }
        opt_eta = -1 * grad_alpha_delta_alpha / a_Q_a;
        opt_eta = std::min(opt_eta, eta.get_value());

        // update global alpha and global weight
        for (i = 0; i < l; i++) {
            alpha[i] += opt_eta * delta_alpha[i];
        }
        alpha = orig_alpha + opt_eta * delta_alpha;
        weight = orig_weight + opt_eta * orig_param_server_weight;

        // f(w) + f(a) will cancel out the 0.5\alphaQ\alpha term (old value)
        dual += opt_eta * (0.5 * opt_eta * a_Q_a + grad_alpha_delta_alpha);
        pdw += 0.5 * opt_eta * (2 * orig_weight.dot(orig_param_server_weight) +
                                opt_eta * orig_param_server_weight.dot(orig_param_server_weight));

        Aggregator<double> loss(0.0, [](double& a, const double& b) { a += b; });
        loss.to_reset_each_iter();
        for (auto& labeled_point : train_set_data) {
            diff = 1 - labeled_point.y * weight.dot(labeled_point.x);
            loss.update(C * diff * diff);
        }
        AggregatorFactory::sync();

        primal = loss.get_value() + pdw;

        if (primal < old_primal) {
            old_primal = primal;
            best_weight = weight;
        }

        gap = (primal + dual) / initial_gap;

        /*
        if (tid == 0) {
            husky::LOG_I << "[Iter " + std::to_string(iter_out) + "]: primal: " + std::to_string(primal) + ", dual:
        " + std::to_string(dual) + ", gap: " + std::to_string(gap);
        }
        */

        if (gap < EPS) {
            weight = best_weight;
            break;
        }
        iter_out++;
    }

    Aggregator<int> error_agg(0, [](int& a, const int& b) { a += b; });
    Aggregator<int> num_test_agg(0, [](int& a, const int& b) { a += b; });
    auto& ac = AggregatorFactory::get_channel();
    list_execute(test_set, {}, {&ac}, [&](ObjT& labeled_point) {
        double indicator = weight.dot(labeled_point.x);
        indicator *= labeled_point.y;
        if (indicator < 0) {
            error_agg.update(1);
        }
        num_test_agg.update(1);
    });

    if (tid == 0) {
        husky::LOG_I << "Classification accuracy on testing set with C = " + std::to_string(C) + " : " +
                            std::to_string(1.0 - static_cast<double>(error_agg.get_value()) / num_test_agg.get_value());
        auto end = std::chrono::steady_clock::now();
        husky::LOG_I << "Time elapsed: " +
                            std::to_string(
                                std::chrono::duration_cast<std::chrono::duration<float>>(end - start).count());
    }

    delete[] QD;
    delete[] index;
}

void init() {
    if (husky::Context::get_param("is_sparse") == "true") {
        bqo_svm();
    } else {
        husky::LOG_I << "Dense data format is not supported";
    }
}

int main(int argc, char** argv) {
    std::vector<std::string> args({"hdfs_namenode", "hdfs_namenode_port", "train", "test", "C", "format", "is_sparse",
                                   "max_iter", "max_inn_iter"});
    if (husky::init_with_args(argc, argv, args)) {
        husky::run_job(init);
        return 0;
    }
    return 1;
}
