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

class problem {
public:
    double C;
    int n;          // global number of features (appended 1 included)
    int N;          // global number of samples
    int l;
    int W;          // number of workers
    int tid;        // global tid of the worker
    int idx_l;      // index_low
    int idx_h;      // index_high
    int max_iter;
    int max_inn_iter;
    husky::ObjList<ObjT>* train_set;
    husky::ObjList<ObjT>* test_set;
};

class solution {
public:
    double duality_gap;
    DenseVector<double> w;
    DenseVector<double> alpha;
};

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

void initialize(problem* problem_) {
    // worker info
    int W = problem_->W = husky::Context::get_num_workers();
    int tid = problem_->tid = husky::Context::get_global_tid();

    std::string format_str = husky::Context::get_param("format");
    husky::lib::ml::DataFormat format;
    if (format_str == "libsvm") {
        format = husky::lib::ml::kLIBSVMFormat;
    } else if (format_str == "tsv") {
        format = husky::lib::ml::kTSVFormat;
    }
    auto& train_set = husky::ObjListStore::create_objlist<ObjT>("train_set");
    auto& test_set = husky::ObjListStore::create_objlist<ObjT>("test_set");
    problem_->train_set = &train_set;
    problem_->test_set = &test_set;

    // load data
    int n;
    n = husky::lib::ml::load_data(husky::Context::get_param("train"), train_set, format);
    n = std::max(n, husky::lib::ml::load_data(husky::Context::get_param("test"), test_set, format));
    // append 1 to the end of every sample
    for (auto& labeled_point : train_set.get_data()) {
        labeled_point.x.resize(n + 1);
        labeled_point.x.set(n, 1);
    }
    for (auto& labeled_point : test_set.get_data()) {
        labeled_point.x.resize(n + 1);
        labeled_point.x.set(n, 1);
    }
    n += 1;
    problem_->n = n;

    // get model config parameters
    problem_->C = std::stod(husky::Context::get_param("C"));
    problem_->max_iter = std::stoi(husky::Context::get_param("max_iter"));
    problem_->max_inn_iter = std::stoi(husky::Context::get_param("max_inn_iter"));

    // initialize parameters
    husky::lib::ml::ParameterBucket<double> param_list(n + 1);  // scalar b and vector w

    // get the number of global records
    Aggregator<std::vector<int>> local_samples_agg(
        std::vector<int>(W, 0),
        [](std::vector<int>& a, const std::vector<int>& b) {
            for (int i = 0; i < a.size(); i++)
                a[i] += b[i];
        },
        [W](std::vector<int>& v) { v = std::move(std::vector<int>(W, 0)); });

    local_samples_agg.update_any([&train_set, tid](std::vector<int>& v) { v[tid] = train_set.get_size(); });
    AggregatorFactory::sync();
    local_samples_agg.inactivate();

    auto& num_samples = local_samples_agg.get_value();
    int N = 0;
    std::vector<int> sample_distribution_agg(W, 0);
    for (int i = 0; i < num_samples.size(); i++) {
        N += num_samples[i];
        sample_distribution_agg[i] = N;
    }

    int index_low, index_high;
    // A worker holds samples [low, high)
    if (tid == 0) {
        index_low = 0;
    } else {
        index_low = sample_distribution_agg[tid - 1];
    }
    if (tid == (W - 1)) {
        index_high = N;
    } else {
        index_high = sample_distribution_agg[tid];
    }
    int l = index_high - index_low;

    problem_->N = N;
    problem_->l = l;
    problem_->idx_l = index_low;
    problem_->idx_h = index_high;

    if (tid == 0) {
        husky::LOG_I << "Number of samples: " + std::to_string(N);
        husky::LOG_I << "Number of features: " + std::to_string(n);
    }
    return;
}

solution* bqo_svm(problem* problem_) {
    int i, k;

    const auto& train_set = problem_->train_set;
    const auto& test_set = problem_->test_set;
    const auto& train_set_data = train_set->get_data();
    const auto& test_set_data = test_set->get_data();

    const double C = problem_->C;
    const int W = problem_->W;
    const int tid = problem_->tid;
    const int n = problem_->n;
    const int l = problem_->l;
    const int N = problem_->N;
    const int index_low = problem_->idx_l;
    const int index_high = problem_->idx_h;
    const int max_iter = problem_->max_iter;
    const int max_inn_iter = problem_->max_inn_iter;

    int iter_out, inn_iter;

    double diag = 0.5 / C;
    double old_primal, primal, obj, grad_alpha_inc;
    double loss, reg = 0;
    double init_primal = C * N;
    double sum_alpha_inc;
    double w_inc_square;
    double w_dot_w_inc;
    double alpha_inc_square;
    double alpha_inc_dot_alpha;
    double max_step;
    double eta;

    double G;
    double gap;
    // projected gradient
    double PG;

    DenseVector<double> alpha(l, 0.0);
    DenseVector<double> alpha_orig(l);
    DenseVector<double> alpha_inc(l);
    DenseVector<double> w(n, 0.0);
    DenseVector<double> w_orig(n, 0.0);
    DenseVector<double> delta_w(n, 0.0);
    DenseVector<double> best_w(n, 0.0);

    husky::lib::ml::ParameterBucket<double> param_server_w;
    param_server_w.init(n, 0.0);

    Aggregator<double> loss_agg(0.0, [](double& a, const double& b) { a += b; });
    loss_agg.to_reset_each_iter();
    Aggregator<double> sum_alpha_inc_agg(0.0, [](double& a, const double& b) { a += b; });
    sum_alpha_inc_agg.to_reset_each_iter();
    Aggregator<double> alpha_inc_dot_alpha_agg(0.0, [](double& a, const double& b) { a += b; });
    alpha_inc_dot_alpha_agg.to_reset_each_iter();
    Aggregator<double> alpha_inc_square_agg(0.0, [](double& a, const double& b) { a += b; });
    alpha_inc_square_agg.to_reset_each_iter();
    Aggregator<double> eta_agg(INF, [](double& a, const double& b) { a = std::min(a, b); }, [](double& a) { a = INF; });
    eta_agg.to_reset_each_iter();

    double* QD = new double[l];
    int* index = new int[l];

    // cache diagonal Q_ii and index for shrinking
    for (i = 0; i < l; i++) {
        QD[i] = self_dot_product(train_set_data[i].x) + diag;
        index[i] = i;
    }

    old_primal = INF;
    obj = 0;

    /*******************************************************************/
    // comment the following if alpha is 0
    // At first I wonder why don't we set old_primal to 0 here
    // But then I found out that if i did so, the primal value will always be larger than old_primal
    // and as a result best_w is set to w and will be return, which leads to the classifier giving 0
    // as output, which sits right on the decision boundary.
    // this problem is solved if we set alpha to be non-zero though
    for (i = 0; i < l; i++) {
        w += alpha[i] * train_set_data[i].y * train_set_data[i].x;
    }
    param_server_w.init(n, 0.0);
    for (i = 0; i < n; i++) {
        param_server_w.update(i, w[i]);
    }
    AggregatorFactory::sync();
    w = param_server_w.get_all_param();
    // set parameter server back to 0.0
    reg = self_dot_product(w);
    reg *= 0.5;
    i = 0;
    Aggregator<double> et_alpha_agg(0.0, [](double& a, const double& b) { a += b; });
    et_alpha_agg.to_reset_each_iter();
    for (auto& labeled_point : train_set_data) {
        loss = 1 - labeled_point.y * w.dot(labeled_point.x);
        if (loss > 0) {
            // l2 loss
            loss_agg.update(C * loss * loss);
        }
        // this is a minomer, actually this calculate both aTa and eTa
        et_alpha_agg.update(alpha[i] * (alpha[i] * diag - 2));
        i++;
    }
    AggregatorFactory::sync();
    old_primal += reg + loss_agg.get_value();
    obj += 0.5 * et_alpha_agg.get_value() + reg;
    /*******************************************************************/

    iter_out = 0;
    while (iter_out < max_iter) {
        // get parameters for local svm solver
        max_step = INF;
        w_orig = w;
        alpha_orig = alpha;
        sum_alpha_inc = 0;
        w_inc_square = 0;
        w_dot_w_inc = 0;
        alpha_inc_square = 0;
        alpha_inc_dot_alpha = 0;

        for (i = 0; i < l; i++) {
            int j = i + std::rand() % (l - i);
            swap(index[i], index[j]);
        }

        // run local svm solver to get local delta alpha
        inn_iter = 0;
        while (inn_iter < max_inn_iter) {

            for (k = 0; k < l; k++) {
                i = index[k];
                double yi = train_set_data[i].y;
                auto& xi = train_set_data[i].x;

                G = (w.dot(xi)) * yi - 1 + diag * alpha[i];

                PG = 0;
                if (alpha[i] == 0) {
                    if (G < 0) {
                        PG = G;
                    }
                } else if (alpha[i] == INF){
                    if (G > 0) {
                        PG = G;
                    }
                } else {
                    PG = G;
                }

                if (fabs(PG) > 1e-12) {
                    double alpha_old = alpha[i];
                    alpha[i] = std::min(std::max(alpha[i] - G / QD[i], 0.0), INF);
                    loss = yi * (alpha[i] - alpha_old);
                    w += xi * loss;
                }
            }
            inn_iter++;
        }

        alpha_inc = alpha - alpha_orig;
        for (i = 0; i < l; i++) {
            alpha_inc[i] = alpha[i] - alpha_orig[i];
            sum_alpha_inc += alpha_inc[i];
            alpha_inc_square += alpha_inc[i] * alpha_inc[i] * diag;
            alpha_inc_dot_alpha += alpha_inc[i] * alpha_orig[i] * diag;
            if (alpha_inc[i] > 0)
                max_step = std::min(max_step, INF);
            else if (alpha_inc[i] < 0)
                max_step = std::min(max_step, - alpha_orig[i] / alpha_inc[i]);
        }
        sum_alpha_inc_agg.update(sum_alpha_inc);
        alpha_inc_square_agg.update(alpha_inc_square);
        alpha_inc_dot_alpha_agg.update(alpha_inc_dot_alpha);
        eta_agg.update(max_step);

        param_server_w.init(n, 0.0);
        for (i = 0; i < n; i++) {
            param_server_w.update(i, w[i] - w_orig[i]);
        }
        AggregatorFactory::sync();

        sum_alpha_inc = sum_alpha_inc_agg.get_value();
        alpha_inc_square = alpha_inc_square_agg.get_value();
        alpha_inc_dot_alpha = alpha_inc_dot_alpha_agg.get_value();
        max_step = eta_agg.get_value();
        delta_w = param_server_w.get_all_param();

        w_inc_square += self_dot_product(delta_w);
        w_dot_w_inc += w_orig.dot(delta_w);
        param_server_w.init(n, 0.0);
        // get step size
        grad_alpha_inc = w_dot_w_inc + alpha_inc_dot_alpha - sum_alpha_inc;
        if (grad_alpha_inc >= 0) {
            w = best_w;
            break;
        }

        double aQa = alpha_inc_square + w_inc_square;
        eta = std::min(max_step, -grad_alpha_inc /aQa);

        alpha = alpha_orig + eta * alpha_inc;
        w = w_orig + eta * delta_w;

        // f(w) + f(a) will cancel out the 0.5\alphaQ\alpha term (old value)
        obj += eta * (0.5 * eta * aQa + grad_alpha_inc);

        reg += eta * (w_dot_w_inc + 0.5 * eta * w_inc_square);

        primal = 0;

        for (auto& labeled_point : train_set_data) {
            loss = 1 - labeled_point.y * w.dot(labeled_point.x);
            if (loss > 0) {
                loss_agg.update(C * loss * loss);
            }
        }
        AggregatorFactory::sync();

        primal += reg + loss_agg.get_value();


        if (primal < old_primal) {
            old_primal = primal;
            best_w = w;
        }

        gap = (primal + obj) / init_primal;

        if (tid == 0) {
            husky::LOG_I << "primal: " + std::to_string(primal);
            husky::LOG_I << "dual: " + std::to_string(obj);
            husky::LOG_I << "duality_gap: " + std::to_string(gap);
        }

        if (gap < EPS) {
            w = best_w;
            break;
        }
        iter_out++;
    }

    delete[] QD;
    delete[] index;

    solution* solution_ = new solution;
    solution_->duality_gap = gap;
    solution_->alpha = alpha;
    solution_->w = w;

    return solution_;
}

void evaluate(problem* problem_, solution* solution_) {
    const auto& test_set = problem_->test_set;
    const auto& w = solution_->w;

    Aggregator<int> error_agg(0, [](int& a, const int& b) { a += b; });
    Aggregator<int> num_test_agg(0, [](int& a, const int& b) { a += b; });
    auto& ac = AggregatorFactory::get_channel();
    list_execute(*test_set, {}, {&ac}, [&](ObjT& labeled_point) {
        double indicator = w.dot(labeled_point.x);
        indicator *= labeled_point.y;
        if (indicator <= 0) {
            error_agg.update(1);
        }
        num_test_agg.update(1);
    });

    if (problem_->tid == 0) {
        husky::LOG_I << "Classification accuracy on testing set with [C = " + 
                        std::to_string(problem_->C) + "], " +
                        "[max_iter = " + std::to_string(problem_->max_iter) + "], " +
                        "[max_inn_iter = " + std::to_string(problem_->max_inn_iter) + "], " +
                        "[test set size = " + std::to_string(num_test_agg.get_value()) + "]: " +
                        std::to_string(1.0 - static_cast<double>(error_agg.get_value()) / num_test_agg.get_value());  
    }
}

void job_runner() {
    problem* problem_ = new problem;

    initialize(problem_);

    solution* solution_ = bqo_svm(problem_);

    evaluate(problem_, solution_);

    delete problem_;
    delete solution_;
}

void init() {
    if (husky::Context::get_param("is_sparse") == "true") {
        job_runner();
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
