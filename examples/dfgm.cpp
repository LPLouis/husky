/***
  Data point id starts from 1
  Actual feature idx starts from 0
 ***/
#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>
#include <stdio.h>      
#include <assert.h>

#include "customize_data_loader.hpp"

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

class trObj {
public:
    using KeyT = int;
    KeyT key;
    const KeyT& id() const { return key; }
    trObj() = default;
    explicit trObj(const KeyT& k) : key(k) {}
};

class info {
public:
    int num_workers;
    int worker_id;
};

class vd_info {
public:
    husky::ObjList<trObj>* fea_score_dummy_list;
};

template <typename T>
struct my_comparator {
    bool operator()(const struct husky::lib::FeaValPair<T>& lhs, const struct husky::lib::FeaValPair<T>& rhs) {
        return lhs.val < rhs.val;
    }
};

class problem {
public:
    int B;
    double C;
    int n;          // number of features (appended 1 included)
    int N;          // number of samples (global)
    int l;          // number of samples (local)
    int index_low;
    int index_high;
    int max_iter;
    int max_inn_iter;
    DenseVector<double> label;
    husky::ObjList<ObjT>* train_set_sw;
    husky::ObjList<ObjT>* train_set_fw;
    husky::ObjList<ObjT>* test_set;
    vd_info* vd_info_;
};

void show(const DenseVector<double>& vec) {
    std::string ret = "";
    for (int i = 0; i < vec.get_feature_num(); i++) {
        ret += std::to_string(vec[i]) + " "; 
    }
    husky::LOG_I << ret << "\n";
    husky::LOG_I << vec.get_feature_num();
}

void show(const SparseVector<double>& vec) {
    std::string ret = "";
    for (auto it = vec.begin(); it != vec.end(); it++) {
        ret += std::to_string((*it).fea) + ":" + std::to_string((*it).val) + " ";
    }
    husky::LOG_I << ret << "\n";
    husky::LOG_I << vec.get_feature_num();
}

template <typename T>
inline void myswap(T& a, T& b) {
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

template <typename T>
DenseVector<T> elem_wise_dot(const DenseVector<T>& v, const std::vector<double>& mu, const std::vector<SparseVector<T>>& dt) {
    assert(mu.size() == dt.size() && "mu size not equal to dt size");
    DenseVector<T> ret(v.get_feature_num(), 0.0);
    DenseVector<T> temp(v.get_feature_num(), 0.0);
    for (int i = 0; i < mu.size(); i++) {
        temp = temp + (dt[i] * mu[i]);
    }

    for (int i = 0; i < v.get_feature_num(); i++) {
        ret[i] = v[i] * temp[i]; 
    }

    return ret;
}

template <typename T>
SparseVector<T> elem_wise_dot(const SparseVector<T>& v, const std::vector<double>& mu, const std::vector<SparseVector<T>>& dt) {
    assert(mu.size() == dt.size() && "mu size not equal to dt size");
    SparseVector<T> ret(v.get_feature_num());
    DenseVector<T> temp(v.get_feature_num(), 0.0);
    for (int i = 0; i < mu.size(); i++) {
        temp = temp + (dt[i] * mu[i]);
    }

    for (auto it = v.begin(); it != v.end(); it++) {
        ret.set((*it).fea, (*it).val * temp[(*it).fea]);
    }

    return ret;
}

void initialize(info* info_, problem* problem_) {
    // worker info_
    int num_workers, worker_id, n, N, index_low, index_high;
    int sentinel_N;
    info_->num_workers = num_workers = husky::Context::get_num_workers();
    info_->worker_id = worker_id = husky::Context::get_global_tid();

    // initialize and globalize feature socre object list
    problem_->vd_info_ = new vd_info();
    auto& temp_fea_score_dummy_list = husky::ObjListStore::create_objlist<trObj>("fea_score_dummy_list");
    temp_fea_score_dummy_list.add_object(trObj(worker_id));
    globalize(temp_fea_score_dummy_list);
    auto& temp_push_channel = husky::ChannelStore::create_push_channel<std::string>(temp_fea_score_dummy_list, temp_fea_score_dummy_list);
    auto& temp_broadcast_channel = husky::ChannelStore::create_broadcast_channel<int, std::string>(temp_fea_score_dummy_list);
    problem_->vd_info_->fea_score_dummy_list = &temp_fea_score_dummy_list;
    problem_->vd_info_->push_channel = &temp_push_channel;
    problem_->vd_info_->broadcast_channel = &temp_broadcast_channel;

    problem_->train_set_sw = &husky::ObjListStore::create_objlist<ObjT>("train_set_sw");
    problem_->train_set_fw = &husky::ObjListStore::create_objlist<ObjT>("train_set_fw");
    problem_->test_set = &husky::ObjListStore::create_objlist<ObjT>("test_set");

    std::string format_str = husky::Context::get_param("format");
    husky::lib::ml::DataFormat format;
    if (format_str == "libsvm") {
        format = husky::lib::ml::kLIBSVMFormat;
    } else if (format_str == "tsv") {
        format = husky::lib::ml::kTSVFormat;
    }

    // load data
    husky::lib::ml::load_data(husky::Context::get_param("test"), *(problem_->test_set), format);
    n = customize_load_data(husky::Context::get_param("train_sw"), *(problem_->train_set_sw), format);
    sentinel_N = customize_load_data(husky::Context::get_param("train_fw"), *(problem_->train_set_fw), format, false);

    // get model config parameters
    problem_->B = std::stoi(husky::Context::get_param("B"));
    problem_->C = std::stod(husky::Context::get_param("C"));
    problem_->max_iter = std::stoi(husky::Context::get_param("max_iter"));
    problem_->max_inn_iter = std::stoi(husky::Context::get_param("max_inn_iter"));

    auto& train_set_sw_data = problem_->train_set_sw->get_data();
    auto& test_set_data = problem_->test_set->get_data();

    // append 1 to the end of every sample
    for (auto& labeled_point : train_set_sw_data) {
        labeled_point.x.resize(n + 1);
        labeled_point.x.set(n, 1);
    }
    for (auto& labeled_point : test_set_data) {
        labeled_point.x.resize(n + 1);
        labeled_point.x.set(n, 1);
    }
    n += 1;
    problem_->n = n;

    // get the number of global records
    Aggregator<std::vector<int>> vec_local_samples(
        std::vector<int>(num_workers, 0),
        [](std::vector<int>& a, const std::vector<int>& b) {
            for (int i = 0; i < a.size(); i++)
                a[i] += b[i];
        },
        [num_workers](std::vector<int>& v) { v = std::move(std::vector<int>(num_workers, 0)); });

    vec_local_samples.update_any([&](std::vector<int>& v) { v[info_->worker_id] = train_set_sw_data.size(); });
    AggregatorFactory::sync();

    auto& num_samples = vec_local_samples.get_value();
    N = 0;
    std::vector<int> sample_distrib_info_(num_workers, 0);
    for (int i = 0; i < num_samples.size(); i++) {
        N += num_samples[i];
        sample_distrib_info_[i] = N;
    }
    problem_->N = N;
    ASSERT_MSG(N == sentinel_N, "Number of samples in sample-wise and feature-wise partition disagrees");

    // distribute label vector to every worker
    husky::lib::ml::ParameterBucket<double> label(N);
    label.init(N, 0.0);
    for (auto& labeled_point : train_set_sw_data) {
        label.update(labeled_point.id() - 1, labeled_point.y);
    }
    AggregatorFactory::sync();
    problem_->label = label.get_all_param();

    // A worker holds samples [low, high)
    if (worker_id == 0) {
        index_low = 0;
    } else {
        index_low = sample_distrib_info_[worker_id - 1];
    }
    if (worker_id == (num_workers - 1)) {
        index_high = N;
    } else {
        index_high = sample_distrib_info_[worker_id];
    }
    problem_->l = index_high - index_low;
    problem_->index_low = index_low;
    problem_->index_high = index_high;

    if (worker_id == 0) {
        husky::LOG_I << "Number of samples: " + std::to_string(N);
        husky::LOG_I << "Number of features: " + std::to_string(n);
    }
    return;
}

DenseVector<double> bqo_svm(info* info_, problem* problem_) {
    int i, k;

    double C = problem_->C;
    int num_workers = info_->num_workers;
    int worker_id = info_->worker_id;
    int n = problem_->n;
    int N = problem_->N;
    int l = problem_->l;
    int index_low = problem_->index_low;
    int index_high = problem_->index_high;
    int max_iter = problem_->max_iter;
    int max_inn_iter = problem_->max_inn_iter;

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

    auto& train_set_sw_data = problem_->train_set_sw->get_data();

    for (int i = 0; i < l; i++) {
        QD[i] = self_dot_product(train_set_sw_data[i].x) + diag;
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
                myswap(index[i], index[j]);
            }

            for (k = 0; k < active_size; k++) {
                i = index[k];
                int yi = train_set_sw_data[i].y;
                auto& xi = train_set_sw_data[i].x;

                G = (weight.dot(xi)) * yi + -1 + diag * alpha[i];

                PG = 0;
                if (EQUAL_VALUE(alpha[i], 0)) {
                    if (G > PGmax_old) {
                        active_size--;
                        myswap(index[k], index[active_size]);
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
        for (auto& labeled_point : train_set_sw_data) {
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
        if (worker_id == 0) {
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
    list_execute(*(problem_->test_set), {}, {&ac}, [&](ObjT& labeled_point) {
        double indicator = weight.dot(labeled_point.x);
        indicator *= labeled_point.y;
        if (indicator < 0) {
            error_agg.update(1);
        }
        num_test_agg.update(1);
    });

    if (worker_id == 0) {
        husky::LOG_I << "Classification accuracy on testing set with C = " + std::to_string(C) + " : " +
                            std::to_string(1.0 - static_cast<double>(error_agg.get_value()) / num_test_agg.get_value());
        auto end = std::chrono::steady_clock::now();
        husky::LOG_I << "Time elapsed: " +
                            std::to_string(
                                std::chrono::duration_cast<std::chrono::duration<float>>(end - start).count());
    }

    delete[] QD;
    delete[] index;


    husky::lib::ml::ParameterBucket<double> ret;
    ret.init(N, 0.0);
    for (int i = 0; i < train_set_sw_data.size(); i++) {
        ret.update(train_set_sw_data[i].id() - 1, alpha[i]); 
    }
    AggregatorFactory::sync(); 
    return ret.get_all_param();
}

SparseVector<double> find_most_violated(info* info_, problem* problem_, const DenseVector<double>& alpha) {
    int B = problem_->B;
    auto& train_set_fw = problem_->train_set_fw;
    auto& fea_score_dummy_list = problem_->vd_info_->fea_score_dummy_list;
    auto& push_channel = problem_->vd_info_->push_channel;
    auto& broadcast_channel = problem_->vd_info_->broadcast_channel;
    auto& label = problem_->label;
    DenseVector<double> cache(problem_->N);

    //husky::ASSERT_MSG(B <= problem_->n, "B is larger than total number of features");

    for (int i = 0; i < problem_->N; i++) {
        cache[i] = label[i] * alpha[i];
    }

    push_channel->prepare_messages();
    for (auto& labeled_point : train_set_fw->get_data()) {
        std::string push_msg = std::to_string(labeled_point.id()) + ":" + std::to_string(labeled_point.x.dot(cache));
        push_channel->push(push_msg, 0);
    }
    push_channel->flush();

    boost::char_separator<char> sep(":");
    DenseVector<double> fea_score(problem_->n);
    list_execute(*fea_score_dummy_list, [&push_channel, &fea_score, sep](trObj& obj) {
        if (obj.id() == 0) {
            auto& msg = push_channel->get(obj);
            for (std::string msg_elem : msg) {
                boost::tokenizer<boost::char_separator<char>> tok(msg_elem, sep);
                auto it = tok.begin();
                int idx = std::stoi(*it++);
                double val = std::stod(*it); 
                fea_score[idx - 1] = val;
                //husky::LOG_I << "feature [" + std::to_string(idx + 1) + "], score: " + std::to_string(val);
            }
        }
    });
    
    SparseVector<double> control_variable(B);
    std::vector<std::pair<int, double>> temp_vector;
    for (auto it = fea_score.begin_feaval(); it != fea_score.end_feaval(); it++) {
        int fea = (*it).fea;
        double val = (*it).val;
        temp_vector.push_back(std::make_pair(fea, val));
    }
    std::sort(temp_vector.begin(), temp_vector.end(), [](auto& left, auto& right) {
        return left.second > right.second;
    });
    for (int i = 0; i < B; i++) {
        int fea = temp_vector[i].first;
        double val = temp_vector[i].second;
        control_variable.set(fea, val);
    }
    return control_variable;
}

void dfgm() {
    info* info_ = new info;
    problem* problem_ = new problem;

    initialize(info_, problem_);
    auto& fea_score_dummy_list = problem_->vd_info_->fea_score_dummy_list;
    auto& push_channel = problem_->vd_info_->push_channel;
    auto& broadcast_channel = problem_->vd_info_->broadcast_channel;

    DenseVector<double> alpha(problem_->N, 1.0 / problem_->N);
    SparseVector<double> control_variable = find_most_violated(info_, problem_, alpha);

    //DenseVector<double> alpha = bqo_svm(info_, problem_);

    /***
    DenseVector<double> dv_1(10);
    dv_1.set(0, 0);
    dv_1.set(1, 1);
    dv_1.set(2, 2);
    dv_1.set(3, 3);
    dv_1.set(4, 4);

    DenseVector<double> dv_2(10);
    dv_2.set(0, 0);
    dv_2.set(1, 1);
    dv_2.set(2, 4);
    dv_2.set(3, 9);
    dv_2.set(4, 16);

    DenseVector<double> dv_3(10);
    dv_2.set(5, 0);
    dv_2.set(6, 1);
    dv_2.set(7, 4);
    dv_2.set(8, 9);
    dv_2.set(9, 16);

    SparseVector<double> sv_1(5);
    sv_1.set(1,1);
    sv_1.set(3,3);

    SparseVector<double> sv_2(5);
    sv_2.set(1,1);
    sv_2.set(3,9);

    SparseVector<double> dt1(10);
    dt1.set(3,1);
    dt1.set(4,1);

    SparseVector<double> dt2(10);
    dt2.set(1,1);
    dt2.set(4,1);

    SparseVector<double> dt3(10);
    dt3.set(7,1);
    dt3.set(9,1);
    std::vector<double> mu;
    mu.push_back(0.1);
    mu.push_back(0.2);
    mu.push_back(0.7);

    std::vector<SparseVector<double>> dt_vec;
    dt_vec.push_back(dt1);
    dt_vec.push_back(dt2);
    dt_vec.push_back(dt3);

    auto ret_1 = elem_wise_dot(dv_1 + dv_2 + dv_3, mu, dt_vec);
    auto ret_2 = elem_wise_dot(dv_1, mu, dt_vec) + elem_wise_dot(dv_2, mu, dt_vec) + elem_wise_dot(dv_3, mu, dt_vec);

    if (husky::Context::get_global_tid() == 0) {
        husky::LOG_I << "\nshowing ret_1\n";
        show(ret_1);
        husky::LOG_I << "\nshowing ret_2\n";
        show(ret_2);
    }
    ***/
}

void init() {
    if (husky::Context::get_param("is_sparse") == "true") {
        dfgm();
    } else {
        husky::LOG_I << "Dense data format is not supported";
    }
}

int main(int argc, char** argv) {
    std::vector<std::string> args({"hdfs_namenode", "hdfs_namenode_port", "train_sw", "train_fw", "test", "B", "C", "format", "is_sparse",
                                   "max_iter", "max_inn_iter"});
    if (husky::init_with_args(argc, argv, args)) {
        husky::run_job(init);
        return 0;
    }
    return 1;
}
