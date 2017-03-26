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

static int bqo_svm_counter = 0;

class data {
public:
    int n;          // global number of features (appended 1 included)
    int N;          // global number of samples
    int l;
    int W;          // number of workers
    int tid;        // global tid of the worker
    int idx_l;      // index_low
    int idx_h;      // index_high     
    // DenseVector<double> label;
    husky::ObjList<ObjT>* train_set;
    // husky::ObjList<ObjT>* train_set_fw;
    husky::ObjList<ObjT>* test_set;
};

class model {
public:
    int B;
    double C;
    int max_iter;
    int max_inn_iter;
    std::vector<double> mu_set;
    std::vector<SparseVector<double>> dt_set;

    model(){}

    model(const model& m) {
        B = m.B;
        C = m.C;
        max_inn_iter = m.max_inn_iter;
        dt_set = std::vector<SparseVector<double>>();
        for (int i = 0; i < m.dt_set.size(); i++) {
            dt_set.push_back(m.dt_set[i]);
        }
    }
};

class solu {
public:
    double obj;
    DenseVector<double> alpha;
    // note that w is uncontrolled
    DenseVector<double> w;
    // // sparse d + sparse d equals to dense d, so may be don't store \sum_mu_dt may be better?
    // // wt in wt_list is controlled but unweighted
    // std::vector<SparseVector<double>> wt_list;
    // w_controlled is weighted
    DenseVector<double> w_controlled;

    solu() {}

    solu(int l, int n, int T) {
        obj = 0.0;
        alpha = DenseVector<double>(l, 0.0);
        w = DenseVector<double>(n, 0.0);
        // wt_list = std::vector<SparseVector<double>>(T);
        w_controlled = DenseVector<double>(n);
    }
};

class vector_operator {
    public:

    static inline bool double_equals(double a, double b, double epsilon = 1.0e-6) {
        return std::abs(a - b) < epsilon;
    }

    static void show(const std::vector<double>& vec, std::string message_head) {
        std::string ret = message_head;
        for (int i = 0; i < vec.size(); i++) {
            ret += ": (" + std::to_string(i + 1) + ", " + std::to_string(vec[i]) + "),";
        }
        husky::LOG_I << ret;
    }

    static void show(const DenseVector<double>& vec, std::string message_head) {
        std::string ret = message_head + ": ";
        for (int i = 0; i < vec.get_feature_num(); i++) {
            ret += std::to_string(vec[i]) + " "; 
        }
        husky::LOG_I << ret << "\n";
    }

    static void show(const SparseVector<double>& vec, std::string message_head) {
        std::string ret = message_head + ": ";
        for (auto it = vec.begin(); it != vec.end(); it++) {
            ret += std::to_string((*it).fea) + ":" + std::to_string((*it).val) + " ";
        }
        husky::LOG_I << ret << "\n";
    }

    static double rmse(const std::vector<double>& v1, const std::vector<double>& v2) {
        assert(v1.size() == v2.size() && "rmse: size of vectors does not agree");
        double ret = 0.0;
        for (int i = 0; i < v1.size(); i++) {
            double diff = v1[i] - v2[i];
            ret += diff * diff;
        }
        return sqrt(ret / v1.size());
    }

    static double rmse(const DenseVector<double>& v1, const DenseVector<double>& v2) {
        assert(v1.get_feature_num() == v2.get_feature_num() && "rmse: size of dense vectors does not agree");
        double ret = 0.0;
        for (int i = 0; i < v1.get_feature_num(); i++) {
            double diff = v1[i] - v2[i];
            ret += diff * diff;
        }
        return sqrt(ret / v1.get_feature_num());
    }

    template <typename T>
    static inline void swap(T& a, T& b) {
        T t = a;
        a = b;
        b = t;
    }

    static bool sparse_equal(const SparseVector<double>& v1, const SparseVector<double>& v2) {
        assert(v1.get_feature_num() == v2.get_feature_num() && "size of sparse vectors does not agree");
        for (auto it1 = v1.begin(), it2 = v2.begin(); it1 != v1.end() && it2 != v2.end(); it1++, it2++){
            if ((*it1).fea != (*it2).fea) {
                return false;
            }
        }
        return true;
    }

    static bool elem_at(const SparseVector<double>& dt, const std::vector<SparseVector<double>>& dt_set) {
        for (int i = 0; i < dt_set.size(); i++) {
            if (sparse_equal(dt, dt_set[i])) {
                return true;
            }
        }
        return false;
    }

    static DenseVector<double> sum_mu_dt(const std::vector<double>& mu_set, const std::vector<SparseVector<double>>& dt_set) {
        assert(mu_set.size() == dt_set.size() && (dt_set.size() != 0) && "mu_set size not equal to dt size");
        DenseVector<double> ret(dt_set[0].get_feature_num(), 0.0);
        for (int i = 0; i < dt_set.size(); i++) {
            ret = ret + dt_set[i] * mu_set[i];
        }
        return ret;
    }

    // this is used when the dense vector are to be controlled by the controll variable
    static DenseVector<double> elem_wise_dot(const DenseVector<double>& v, const DenseVector<double>& dt) {
        assert(v.get_feature_num() == dt.get_feature_num() && "elem_wise_dot: feature number of dt not equal to feature number of v");
        DenseVector<double> ret(v.get_feature_num(), 0.0);
        for (int i = 0; i < v.get_feature_num(); i++) {
            ret[i] = v[i] * dt[i]; 
        }
        return ret;
    }

    // this is used when the control variable is the summed control variable
    static SparseVector<double> elem_wise_dot(const SparseVector<double>& v, const DenseVector<double>& dt) {
        assert(v.get_feature_num() == dt.get_feature_num() && "elem_wise_dot: feature number of dt not equal to feature number of v");
        SparseVector<double> ret(v.get_feature_num());
        for (auto it = v.begin(); it != v.end(); it++) {
            ret.set((*it).fea, (*it).val * dt[(*it).fea]);
        }
        return ret;
    }

    // this is used when the control variable is the summed control variable, same as above
    static SparseVector<double> elem_wise_dot(const DenseVector<double>& dt, const SparseVector<double>& v) {
        assert(v.get_feature_num() == dt.get_feature_num() && "elem_wise_dot: feature number of dt not equal to feature number of v");
        SparseVector<double> ret(v.get_feature_num());
        for (auto it = v.begin(); it != v.end(); it++) {
            ret.set((*it).fea, (*it).val * dt[(*it).fea]);
        }
        return ret;
    }

    // first argument must be the sparse vector and the second argument must be the control variable
    // also the input argument must be sorted
    static SparseVector<double> elem_wise_dot(const SparseVector<double>& v, const SparseVector<double>& dt) {
        assert(v.get_feature_num() == dt.get_feature_num() && "elem_wise_dot: feature number of dt not equal to feature number of v");
        SparseVector<double> ret(v.get_feature_num());
        auto it1 = v.begin();
        auto it2 = dt.begin();
        while(it1 != v.end() && it2 != dt.end()) {
            int fea1 = (*it1).fea;
            int fea2 = (*it2).fea;
            if (fea1 == fea2) {
                ret.set(fea1, (*it1).val);
                it1++;
                it2++;
            } else if (fea1 > fea2) {
                it2++;
            } else {
                it1++;
            }
        }
        return ret;
    }

    template <typename T, bool is_sparse>
    static T self_dot_product(const husky::lib::Vector<T, is_sparse>& v) {
        T ret = 0;
        for (auto it = v.begin_value(); it != v.end_value(); it++) {
            ret += (*it) * (*it);
        }
        return ret;
    }

    template <typename T>
    static T self_dot_elem_wise_dot(const SparseVector<T>& v, const std::vector<double>& mu_set, const std::vector<SparseVector<double>>& dt_set) {
        assert(dt_set.size() != 0 && "dt_set is of size 0");
        T ret = 0;
        for (int i = 0; i < mu_set.size(); i++) {
            ret += mu_set[i] * self_dot_product(elem_wise_dot(v, dt_set[i]));
        }
        return ret;
    }

    static void my_min(const double *vet, int size, double *min_value, int *min_index) {
        int i;
        double tmp = vet[0];
        min_index[0] = 0;

        for(i=0; i<size; i++){
            if(vet[i] < tmp){
                tmp = vet[i];
                min_index[0] = i;
            }
        }
        min_value[0] = tmp;
    }

    static void my_min(const DenseVector<double>& vet, int size, double *min_value, int *min_index) {
        int i;
        double tmp = vet[0];
        min_index[0] = 0;

        for(i=0; i<size; i++){
            if(vet[i] < tmp){
                tmp = vet[i];
                min_index[0] = i;
            }
        }
        min_value[0] = tmp;
    }

    template<typename T>
    static int find_max_index(const std::vector<T>& v) {
        assert(v.size() != 0 && "find_max_index: size of vector can not be 0");
        int index = 0;
        T val = v[0];
        for (int i = 1; i < v.size(); i++) {
            if (val < v[i]) {
                index = i;
                val = v[i];
            }
        }
        return index;
    }

    static double find_max_step(const std::vector<double>& mu_set, const DenseVector<double>& desc) {
        assert(mu_set.size() != 0 && mu_set.size() == desc.get_feature_num() && "arg_min_dm_over_Dm: error");
        int i;
        int flag = 1;
        double step_max = 0;
        for (i = 0; i < mu_set.size(); i++) {
            if (desc[i] < 0) {
                // if step_max uninitialized
                if (flag == 1) {
                    step_max = -mu_set[i] / desc[i];
                    flag = 0;
                // if step_max initialized
                } else {
                    double tmp = -mu_set[i] / desc[i];
                    if (tmp < step_max) {
                        step_max = tmp;
                    }
                }
            }
        }
        // if no entry satisfy, return 0
        return step_max;
    }

    static bool sum_equal_to(const std::vector<double>& v, double val) {
        if (v.size() == 0) {
            return false;
        }
        double sum = 0.0;
        for (int i = 0; i < v.size(); i++) {
            sum += v[i];
        }

        if (double_equals(sum, val)) {
            return true;
        }
        husky::LOG_I << "sum_equal_to: " << std::to_string(sum);
        return false;
    }

    static bool sum_equal_to(const DenseVector<double>& v, double val) {
        if (v.get_feature_num() == 0) {
            return false;
        }
        double sum = 0.0;
        for (int i = 0; i < v.get_feature_num(); i++) {
            sum += v[i];
        }

        if (double_equals(sum, val)) {
            return true;
        }
        husky::LOG_I << "sum_equal_to: " << std::to_string(sum);
        return false;
    }

    static void normalize(DenseVector<double>& v) {
        double sum = 0.0;
        for (int i = 0; i < v.get_feature_num(); i++) {
            sum += v[i];
        }
        for (int i = 0; i < v.get_feature_num(); i++) {
            v[i] /= sum;
        }
    }
};

void initialize(data* data_, model* model_) {
    // worker info
    int W = data_->W = husky::Context::get_num_workers();
    int tid = data_->tid = husky::Context::get_global_tid();

    std::string format_str = husky::Context::get_param("format");
    husky::lib::ml::DataFormat format;
    if (format_str == "libsvm") {
        format = husky::lib::ml::kLIBSVMFormat;
    } else if (format_str == "tsv") {
        format = husky::lib::ml::kTSVFormat;
    }
    auto& train_set = husky::ObjListStore::create_objlist<ObjT>("train_set");
    auto& test_set = husky::ObjListStore::create_objlist<ObjT>("test_set");
    data_->train_set = &train_set;
    data_->test_set = &test_set;

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
    data_->n = n;

    // get model config parameters
    model_->B = std::stoi(husky::Context::get_param("B"));
    model_->C = std::stod(husky::Context::get_param("C"));
    model_->max_iter = std::stoi(husky::Context::get_param("max_iter"));
    model_->max_inn_iter = std::stoi(husky::Context::get_param("max_inn_iter"));

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

    data_->N = N;
    data_->l = l;
    data_->idx_l = index_low;
    data_->idx_h = index_high;

    if (l != 0) {
        husky::LOG_I << "Worker " + std::to_string(data_->tid) + " holds sample [" + std::to_string(index_low) + ", " + std::to_string(index_high) + ")";
    }

    if (tid == 0) {
        husky::LOG_I << "Number of samples: " + std::to_string(N);
        husky::LOG_I << "Number of features: " + std::to_string(n);
    }
    return;
}

SparseVector<double> find_most_violated(data* data_, model* model_, solu* solu_ = NULL) {
    int B = model_->B;
    auto& train_set = data_->train_set;
    DenseVector<double> alpha;
    DenseVector<double> w;
    if (solu_ == NULL) {
        // set alpha to 1 because we do not know how many data others have
        alpha = DenseVector<double>(data_->l, 1.0);
        w = DenseVector<double>(data_->n, 0.0);
    } else {
        w = solu_->w;
    }
    // DenseVector<double> cache(data_->l);
    // DenseVector<double> label = data_->label;
    std::vector<std::pair<int, double>> fea_score;


    // for (int i = 0; i < data_->l; i++) {
    //     cache[i] = label[i] * alpha[i];
    // }
    // for (auto& labeled_point : train_set_fw->get_data()) {
    //     fea_score.push_back(std::make_pair(labeled_point.id() - 1, labeled_point.x.dot(cache)));
    // }
    // fea_score.push_back(std::make_pair(data_->n - 1, cache.dot(DenseVector<double>(data_->l, 1.0))));
    int i;
    if (solu_ == NULL) {
        husky::lib::ml::ParameterBucket<double> param_server_w;
        param_server_w.init(data_->n, 0.0);
        for (auto& labeled_point : train_set->get_data()) {
            w += labeled_point.x * labeled_point.y * alpha[i];
        }
        for (i = 0; i < data_->n; i++) {
            param_server_w.update(i, w[i]);
        }
        AggregatorFactory::sync();
        w = param_server_w.get_all_param();
    }

    for (i = 0; i < data_->n; i++) {
        fea_score.push_back(std::make_pair(i, w[i] * w[i]));
    }

    std::sort(fea_score.begin(), fea_score.end(), [](auto& left, auto& right) {
        return left.second > right.second;
    });
    SparseVector<double> control_variable(data_->n);
    for (i = 0; i < B; i++) {
        int fea = fea_score[i].first;
        double val = fea_score[i].second;
        control_variable.set(fea, 1.0);
    }
    control_variable.sort_asc();
    return control_variable;
}

void bqo_svm(data* data_, model* model_, solu* output_solu_, solu* input_solu_ = NULL, double* QD = NULL, int* index = NULL, bool cache = false) {
    bqo_svm_counter++;
    husky::LOG_I << "enter BQO_SVM, counter: " + std::to_string(bqo_svm_counter);
    int i, k;

    const auto& train_set = data_->train_set;
    const auto& train_set_data = train_set->get_data();
    const auto& mu_set = model_->mu_set;
    const auto& dt_set = model_->dt_set;

    // combine all control variables to a single control variable (Assumption: coef of dt sum to 1)
    const DenseVector<double> control_variable = vector_operator::sum_mu_dt(mu_set, dt_set);

    const double C = model_->C;
    const int W = data_->W;
    const int tid = data_->tid;
    const int n = data_->n;
    const int l = data_->l;
    const int N = data_->N;
    const int index_low = data_->idx_l;
    const int index_high = data_->idx_h;
    const int max_iter = model_->max_iter;
    const int max_inn_iter = model_->max_inn_iter;

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
    double best_loss = C * N;

    DenseVector<double> alpha;
    if (input_solu_ == NULL) {
        alpha = DenseVector<double>(l, 0.0);
    } else {
        alpha = input_solu_->alpha;
    }
    DenseVector<double> orig_alpha(l);
    DenseVector<double> delta_alpha(l);
    DenseVector<double> lower_bound(l, 0.0);
    DenseVector<double> grad_alpha(l, -1.0);
    DenseVector<double> w(n, 0.0);
    DenseVector<double> w_controlled(n, 0.0); // newly added
    DenseVector<double> best_w(n, 0.0);
    DenseVector<double> orig_w(n);
    DenseVector<double> orig_w_controlled(n); // newly added
    DenseVector<double> orig_param_server_w_controlled(n); // newly added

    DenseVector<double> orig_param_server_w(n, 0.0);
    husky::lib::ml::ParameterBucket<double> param_server_w;
    param_server_w.init(n, 0.0);

    Aggregator<double> loss(0.0, [](double& a, const double& b) { a += b; });
    loss.to_reset_each_iter();
    Aggregator<double> alpha_delta_alpha(0.0, [](double& a, const double& b) { a += b; });
    alpha_delta_alpha.to_reset_each_iter();
    Aggregator<double> et_delta_alpha(0.0, [](double& a, const double& b) { a += b; });
    et_delta_alpha.to_reset_each_iter();
    Aggregator<double> delta_alpha_square(0.0, [](double& a, const double& b) { a += b; });
    delta_alpha_square.to_reset_each_iter();
    Aggregator<double> eta(INF, [](double& a, const double& b) { a = std::min(a, b); }, [](double& a) { a = INF; });
    eta.to_reset_each_iter();

    if (!cache) {
        QD = new double[l];
        index = new int[l];
        for (i = 0; i < l; i++) {
            QD[i] = vector_operator::self_dot_elem_wise_dot(train_set_data[i].x, mu_set, dt_set) + diag;
            index[i] = i;
        }
    }

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
    for (i = 0; i < n; i++) {
        param_server_w.update(i, w[i]);
    }
    AggregatorFactory::sync();
    husky::LOG_I << "check point1, counter: " + std::to_string(bqo_svm_counter);
    w = param_server_w.get_all_param();
    w_controlled = vector_operator::elem_wise_dot(w, control_variable);
    // double reg = self_dot_product(w);
    double reg = w.dot(w_controlled);
    i = 0;
    Aggregator<double> et_alpha_agg(0.0, [](double& a, const double& b) { a += b; });
    et_alpha_agg.to_reset_each_iter();
    for (auto& labeled_point : train_set_data) {
        // diff = 1 - labeled_point.y * w.dot(labeled_point.x);
        diff = 1 - labeled_point.y * w_controlled.dot(labeled_point.x);
        if (diff > 0.0) {
            loss.update(C * diff * diff);
        }
        // this is a minomer, actually this calculate both aTa and eTa
        et_alpha_agg.update(alpha[i] * (alpha[i] / C - 2));
        i++;
    }
    AggregatorFactory::sync();
    husky::LOG_I << "check point2, counter: " + std::to_string(bqo_svm_counter);
    old_primal += 0.5 * reg + loss.get_value();
    dual += reg + et_alpha_agg.get_value();
    dual /= 2;
    best_w = w;
    // set param_server_w back to 0.0
    param_server_w.init(n, 0.0);
    /*******************************************************************/

    auto start = std::chrono::steady_clock::now();

    iter_out = 0;
    while (iter_out < max_iter) {
        // husky::LOG_I << "[BQO_SVM][outer iteration]: " + std::to_string(iter_out);
        // get parameters for local svm solver
        orig_w = w;
        orig_w_controlled = w_controlled;
        orig_alpha = alpha;

        // run local svm solver to get local delta alpha
        inn_iter = 0;
        active_size = l;

        while (inn_iter < max_inn_iter) {
            PGmax_new = 0;
            PGmin_new = 0;

            for (i = 0; i < active_size; i++) {
                int j = i + std::rand() % (active_size - i);
                vector_operator::swap(index[i], index[j]);
            }

            for (k = 0; k < active_size; k++) {
                i = index[k];
                int yi = train_set_data[i].y;
                auto& xi = train_set_data[i].x;

                // G = (w.dot(xi)) * yi + -1 + diag * alpha[i];
                G = (w_controlled.dot(xi)) * yi + -1 + diag * alpha[i];
                PG = 0;
                if (EQUAL_VALUE(alpha[i], 0)) {
                    if (G > PGmax_old) {
                        active_size--;
                        vector_operator::swap(index[k], index[active_size]);
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
                    w += xi * diff;
                    w_controlled = vector_operator::elem_wise_dot(w, control_variable);
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
            // since parameter server won't be reset, need to cancel out the original values
            param_server_w.update(i, w[i] - orig_w[i] - orig_param_server_w[i] / W);
        }
        AggregatorFactory::sync();
        husky::LOG_I << "check point3, counter: " + std::to_string(bqo_svm_counter);

        orig_param_server_w = param_server_w.get_all_param();
        orig_param_server_w_controlled = vector_operator::elem_wise_dot(orig_param_server_w, control_variable);
        // get step size
        // grad_alpha_delta_alpha = orig_w.dot(orig_param_server_w) + 1.0 / C * alpha_delta_alpha.get_value() -
        //                          et_delta_alpha.get_value();
        grad_alpha_delta_alpha = orig_w.dot(orig_param_server_w_controlled) + 1.0 / C * alpha_delta_alpha.get_value() -
                                 et_delta_alpha.get_value();
        // a_Q_a = orig_param_server_w.dot(orig_param_server_w) + 1.0 / C * delta_alpha_square.get_value();
        a_Q_a = orig_param_server_w.dot(orig_param_server_w_controlled) + 1.0 / C * delta_alpha_square.get_value();                         
        if (grad_alpha_delta_alpha > 0) {
            w = best_w;
            break;
        }
        opt_eta = -1 * grad_alpha_delta_alpha / a_Q_a;
        opt_eta = std::min(opt_eta, eta.get_value());

        // update global alpha and global w
        for (i = 0; i < l; i++) {
            alpha[i] += opt_eta * delta_alpha[i];
        }
        alpha = orig_alpha + opt_eta * delta_alpha;
        w = orig_w + opt_eta * orig_param_server_w;
        w_controlled = orig_w_controlled + opt_eta * orig_param_server_w_controlled;

        // f(w) + f(a) will cancel out the 0.5\alphaQ\alpha term (old value)
        dual += opt_eta * (0.5 * opt_eta * a_Q_a + grad_alpha_delta_alpha);
        // pdw += 0.5 * opt_eta * (2 * orig_w.dot(orig_param_server_w) +
        //                         opt_eta * orig_param_server_w.dot(orig_param_server_w));
        pdw += 0.5 * opt_eta * (2 * orig_w.dot(orig_param_server_w_controlled) +
                                opt_eta * orig_param_server_w.dot(orig_param_server_w_controlled));

        for (auto& labeled_point : train_set_data) {
            // diff = 1 - labeled_point.y * w.dot(labeled_point.x);
            diff = 1 - labeled_point.y * w_controlled.dot(labeled_point.x);
            if (diff < 0) {
                loss.update(C * diff * diff);
            }
        }
        if (l == 0) {
            loss.update(0.0);
        }
        AggregatorFactory::sync();
        husky::LOG_I << "check point4, counter: " + std::to_string(bqo_svm_counter);

        primal = loss.get_value() + pdw;
        if (primal < old_primal) {
            old_primal = primal;
            best_w = w;
            best_loss = loss.get_value();
        }

        gap = (primal + dual) / initial_gap;
        // if (tid == 0) {
        //     husky::LOG_I << "[BQO_SVM][primal]: " + std::to_string(primal);
        //     husky::LOG_I << "[BQO_SVM][dual]: " + std::to_string(dual);
        //     husky::LOG_I << "[BQO_SVM][duality gap]: " + std::to_string(gap);
        // }
        if (gap < EPS) {
            w = best_w;
            break;
        }
        iter_out++;
    }

    delete[] QD;
    delete[] index;

    output_solu_->w = w;
    output_solu_->w_controlled = vector_operator::elem_wise_dot(w, control_variable);
    output_solu_->obj = 0.5 * w.dot(output_solu_->w_controlled) + best_loss;
    output_solu_->alpha = alpha;

    husky::LOG_I << "leaving BQO_SVM, counter: " + std::to_string(bqo_svm_counter);
}

void simpleMKL(data* data_, model* model_, solu* solu_) {
    assert(model_->mu_set.size() == model_->dt_set.size()  && "size of mu_set and dt_set do not agree");
    
    int i,j;
    int nloop, loop, maxloop;
    nloop = 1;
    loop = 1;
    maxloop = 12;

    const int l = data_->l;   
    const int n = data_->n;
    const int T = model_->mu_set.size();
    const double gold_ratio = (sqrt(double(5)) + 1) / 2;

    auto& mu_set = model_->mu_set;
    auto& dt_set = model_->dt_set;

    // initialize mu_set
    double init = 1.0 / T;
    for (int i = 0; i < T; i++) {
        mu_set[i] = init;
    }

    // cache QD and index
    const auto& labeled_point_vector = data_->train_set->get_data();
    DenseVector<double> control_variable = vector_operator::sum_mu_dt(mu_set, dt_set);
    double diag = 0.5 / model_->C;
    double* QD = new double[l];
    int* index = new int[l];
    for (i = 0; i < l; i++) {
        QD[i] = vector_operator::self_dot_elem_wise_dot(labeled_point_vector[i].x, mu_set, dt_set) + diag;
        index[i] = i;
    }

    bqo_svm(data_, model_, solu_, NULL, QD, index, true);
    double obj = solu_->obj;
    // compute gradient
    DenseVector<double> grad(T);
    std::vector<SparseVector<double>> wt_list(T);
    for (i = 0; i < T; i++) {
        wt_list[i] = vector_operator::elem_wise_dot(solu_->w, dt_set[i]);
    }
    for (i = 0; i < T; i++) {
        grad[i] = -0.5 * solu_->w.dot(wt_list[i]);
    }

    model* new_model = new model(*model_);
    new_model->mu_set = std::vector<double>(T);
    auto& new_mu_set = new_model->mu_set;

    model* tmp_model = new model(*model_);
    tmp_model->mu_set = std::vector<double>(T);
    auto& tmp_mu_set = tmp_model->mu_set;

    model* tmp_ls_model_1 = new model(*model_);
    tmp_ls_model_1->mu_set = std::vector<double>(T);
    auto& tmp_ls_mu_set_1 = tmp_ls_model_1->mu_set;

    model* tmp_ls_model_2 = new model(*model_);
    tmp_ls_model_2->mu_set = std::vector<double>(T);
    auto& tmp_ls_mu_set_2 = tmp_ls_model_2->mu_set;

    solu* tmp_solu = new solu(l, n, T);
    solu* tmp_ls_solu_1 = new solu(l, n, T);
    solu* tmp_ls_solu_2 = new solu(l, n, T);

    while (loop == 1 && maxloop > 0 && T > 1) {
        nloop++;

        double old_obj = obj;
        // if (data_->tid == 0) {
            husky::LOG_I << "[SimpleMKL][outer loop][old_obj]: " + std::to_string(old_obj);
        // }

        // initialize a new model
        new_model->mu_set = mu_set;

        // normalize gradient
        double sum_grad = grad.dot(grad);
        grad /= sqrt(sum_grad);

        // compute descent direction
        int max_index = vector_operator::find_max_index(mu_set);
        double grad_tmp = grad[max_index];
        for (i = 0; i < T; i++) {
            grad[i] -= grad_tmp;
        }

        DenseVector<double> desc(T, 0.0);
        double sum_desc = 0;
        for (i = 0; i < T; i++) {
            if (mu_set[i] > 0 || grad[i] < 0) {
                desc[i] = -grad[i];
            }
            sum_desc += desc[i];
        }
        desc[max_index] = -sum_desc;

        double step_min = 0;
        double cost_min = old_obj;
        double cost_max = 0;
        double step_max = 0;
        // note here we use new_sigma
        step_max = vector_operator::find_max_step(new_mu_set, desc);

        double delta_max = step_max;

        int flag = 1;
        if (step_max == 0) {
            flag = 0;
        }

        if (flag == 1) {
            if (step_max > 0.1) {
                step_max = 0.1;
                delta_max = step_max;
            }

            while (cost_max < cost_min) {
                // if (data_->tid == 0) {
                    husky::LOG_I << "[SimpleMKL][inner loop][cost_max]: " + std::to_string(cost_max);
                // }
                for (i = 0; i < T; i++) {
                    tmp_mu_set[i] = new_mu_set[i] + step_max * desc[i];
                }
                // use descent direction to compute new objective
                // consider modifying input solution to speed up
                bqo_svm(data_, tmp_model, tmp_solu, solu_, QD, index, true);
                cost_max = tmp_solu->obj;
                if (cost_max < cost_min) {
                    cost_min = cost_max;

                    new_mu_set = tmp_mu_set;
                    mu_set = tmp_mu_set;

                    sum_desc = 0;
                    int fflag = 1;
                    for (i = 0; i < T; i++) {
                        if (new_mu_set[i] > 1e-12 || desc[i] > 0) {
                            ;
                        } else {
                            desc[i] = 0;
                        }

                        if (i != max_index) {
                            sum_desc += desc[i];
                        }
                        // as long as one of them has descent direction negative, we will go on
                        if (desc[i] < 0) {
                            fflag = 0;
                        }
                    }

                    desc[max_index] = -sum_desc;
                    // only copy solu_->alpha, compute the others later
                    solu_->alpha = tmp_solu->alpha;

                    if (fflag) {
                        step_max = 0;
                        delta_max = 0;
                    } else {
                        step_max = vector_operator::find_max_step(new_mu_set, desc);
                        delta_max = step_max;
                        cost_max = 0;
                    } // if (fflag)
                } // if (cost_max < cost_min)
            } // while (cost_max < cost_min) 

            // conduct line search
            double* step = new double[4];
            step[0] = step_min;
            step[1] = 0;
            step[2] = 0;
            step[3] = step_max;

            double* cost = new double[4];
            step[0] = cost_min;
            step[1] = 0;
            step[2] = 0;
            step[3] = cost_max;

            double min_val;
            int min_idx;

            if (cost_max < cost_min) {
                min_val = cost_max;
                min_idx = 3;
            } else {
                min_val = cost_min;
                min_idx = 0;
            }

            int step_loop = 0;
            while ((step_max - step_min) > 1e-1 * fabs(delta_max) && step_max > 1e-12) {
                // if (data_->tid == 0) {
                    husky::LOG_I << "[SimpleMKL][line_search] iteration: " + std::to_string(step_loop);
                // }
                step_loop += 1;
                if (step_loop > 8) {
                    break;
                }
                double step_medr = step_min + (step_max - step_min) / gold_ratio;
                double step_medl = step_min + (step_medr - step_min) / gold_ratio;

                // half
                for (i = 0; i < T; i++) {
                    tmp_ls_mu_set_1[i] = new_mu_set[i] + step_medr * desc[i];
                }
                bqo_svm(data_, tmp_ls_model_1, tmp_ls_solu_1, solu_, QD, index, true);

                // half half
                for (i = 0; i < T; i++) {
                    tmp_ls_mu_set_2[i] = new_mu_set[i] + step_medl * desc[i];
                }
                bqo_svm(data_, tmp_ls_model_2, tmp_ls_solu_2, solu_, QD, index, true);

                step[0] = step_min;
                step[1] = step_medl;
                step[2] = step_medr;
                step[3] = step_max;

                cost[0] = cost_min;
                cost[1] = tmp_ls_solu_2->obj;
                cost[2] = tmp_ls_solu_1->obj;
                cost[3] = cost_max;  

                vector_operator::my_min(cost, 4, &min_val, &min_idx);

                switch(min_idx) {
                    case 0:
                        step_max = step_medl;
                        cost_max = cost[1];
                        solu_->obj = tmp_ls_solu_2->obj;
                        solu_->w = tmp_ls_solu_2->w;
                        solu_->alpha = tmp_ls_solu_2->alpha;
                    break;

                    case 1:
                        step_max = step_medr;
                        cost_max = cost[2];
                        solu_->obj = tmp_ls_solu_1->obj; 
                        solu_->w = tmp_ls_solu_1->w; 
                        solu_->alpha = tmp_ls_solu_1->alpha;
                    break;

                    case 2:
                        step_min = step_medl;
                        cost_min = cost[1];
                        solu_->obj = tmp_ls_solu_2->obj;
                        solu_->w = tmp_ls_solu_2->w;
                        solu_->alpha = tmp_ls_solu_2->alpha;                        
                    break;

                    case 3:
                        step_min = step_medr;
                        cost_min = cost[2];
                        solu_->obj = tmp_ls_solu_1->obj;
                        solu_->w = tmp_ls_solu_1->w; 
                        solu_->alpha = tmp_ls_solu_1->alpha;                      
                    break;
                }// switch(min_idx);      
            } // while ((step_max - step_min) > 1e-1 * fabs(delta_max) && step_max > 1e-12)

            // assignment
            double step_size = step[min_idx];
            if (solu_->obj < old_obj) {
                for (i = 0; i < T; i++) {
                    new_mu_set[i] += step_size * desc[i];
                }
                mu_set = new_mu_set;
            }

            delete cost;
            delete step;
        }// if(flag)

        // test convergence
        int mu_max_idx;
        mu_max_idx = vector_operator::find_max_index(mu_set);
        double mu_max = mu_set[mu_max_idx];
        // normalize mu_max
        if (mu_max > 1e-12) {
            double mu_sum = 0;
            for (i = 0; i < T; i++) {
                if (mu_set[i] < 1e-12) {
                    mu_set[i] = 0;
                }
                mu_sum += mu_set[i];
            }
            for (i = 0; i < T; i++) {
                mu_set[i] /= mu_sum;
            }
        }

        // recover w_controlled and wt_list
        control_variable = vector_operator::sum_mu_dt(mu_set, dt_set);
        solu_->w_controlled = vector_operator::elem_wise_dot(solu_->w, control_variable);
        for (i = 0; i < T; i++) {
            wt_list[i] = vector_operator::elem_wise_dot(solu_->w, dt_set[i]);
            grad[i] = -0.5 * solu_->w.dot(wt_list[i]);
        }
        double min_grad = 0;
        double max_grad = 0;
        int ffflag = 1;
        for (i = 0; i < T; i++) {
            if (mu_set[i] > 1e-8) {
                if (ffflag) {
                    min_grad = grad[i];
                    max_grad = grad[i];
                    ffflag = 0;
                } else {
                    if (grad[i] < min_grad) {
                        min_grad = grad[i];
                    }
                    if (grad[i] > max_grad) {
                        max_grad = grad[i];
                    }
                }
            }
        }

        double KKTconstraint = fabs(min_grad - max_grad) / fabs(min_grad);
        // note we find min idx in grad, corresponding to max idx in -grad
        double min_tmp;
        int min_tmp_idx;
        vector_operator::my_min(grad, T, &min_tmp, &min_tmp_idx);
        double tmp_sum = 0;
        for (i = 0; i < l; i++) {
            tmp_sum += solu_->alpha[i];
        }
        double rhs = 0.5 * solu_->w.dot(solu_->w_controlled);
        double dual_gap = (rhs + min_tmp) / rhs;
        // if (data_->tid == 0) {
        //     husky::LOG_I << "[SimpleMKL][outer loop][dual_gap]: " + std::to_string(fabs(dual_gap));
        //     husky::LOG_I << "[SimpleMKL][outer loop][KKTconstraint]: " + std::to_string(KKTconstraint);
        // }
        if (KKTconstraint < 0.05 || fabs(dual_gap) < 0.01) {
            loop = 0;
        }
        if (nloop > maxloop) {
            loop = 0;
            break;
        }
    }
    delete tmp_ls_solu_1;
    delete tmp_ls_solu_2;
    delete tmp_ls_model_1;
    delete tmp_ls_model_2;
    delete tmp_model;
    delete tmp_solu;
}

void evaluate(data* data_, model* model_, solu* solu_) {
    const auto& test_set = data_->test_set;
    const auto& w = solu_->w_controlled;

    Aggregator<int> error_agg(0, [](int& a, const int& b) { a += b; });
    Aggregator<int> num_test_agg(0, [](int& a, const int& b) { a += b; });
    auto& ac = AggregatorFactory::get_channel();
    list_execute(*test_set, {}, {&ac}, [&](ObjT& labeled_point) {
        double indicator = w.dot(labeled_point.x);
        indicator *= labeled_point.y;
        if (indicator < 0) {
            error_agg.update(1);
        }
        num_test_agg.update(1);
    });

    if (data_->tid == 0) {
        husky::LOG_I << "Classification accuracy on testing set with [C = " + 
                        std::to_string(model_->C) + "], " +
                        "[max_iter = " + std::to_string(model_->max_iter) + "], " +
                        "[max_inn_iter = " + std::to_string(model_->max_inn_iter) + "], " +
                        "[test set size = " + std::to_string(num_test_agg.get_value()) + "]: " +
                        std::to_string(1.0 - static_cast<double>(error_agg.get_value()) / num_test_agg.get_value());  
    }
}

void job_runner() {
    data* data_ = new data;
    model* model_ = new model;
    solu* solu_ = new solu;
    initialize(data_, model_);

    int iter = 0;
    int max_iter = 10;
    double mkl_obj = INF;
    double last_obj = INF;
    double obj_diff = 0.0;
    SparseVector<double> dt(data_->n);
    while(iter < max_iter) {
        last_obj = mkl_obj;
        if (iter == 0) {
            dt = find_most_violated(data_, model_);
        } else {
            dt = find_most_violated(data_, model_, solu_);
        }
        if (vector_operator::elem_at(dt, model_->dt_set)) {
            if (data_->tid == 0) {
                husky::LOG_I << "[job_runner]FGM converged";
            }
            break;
        }
        model_->dt_set.push_back(dt);
        model_->mu_set.push_back(1.0);
        simpleMKL(data_, model_, solu_);
        mkl_obj = solu_->obj;
        obj_diff = fabs(mkl_obj - last_obj);
        // if (data_->tid == 0) {
            husky::LOG_I << "[job_runner][iteration " + std::to_string(iter) + "][mkl_obj " + std::to_string(mkl_obj) + "][obj_diff " + std::to_string(obj_diff) + "]";
        //     vector_operator::show(model_->mu_set, "mu_set");
        // }
        if (obj_diff < 0.001 * abs(last_obj)) {
            break;
        }
        if (model_->mu_set[iter] < 0.0001) {
            break;
        }
        iter++;
    }
    evaluate(data_, model_, solu_);



    // data* data_ = new data;
    // model* model_ = new model;
    // solu* solu_ = new solu;
    // initialize(data_, model_);
    // SparseVector<double> dt(data_->n);
    // dt = find_most_violated(data_, model_);
    // model_->dt_set.push_back(dt);
    // model_->mu_set.push_back(1.0);
    // bqo_svm(data_, model_, solu_);
    // if (data_->tid == 0) {
    //     vector_operator::show(model_->mu_set, "mu_set");
    // }
    // evaluate(data_, model_, solu_);

    // delete data_;
    // delete model_;
    // delete solu_;
}

void init() {
    if (husky::Context::get_param("is_sparse") == "true") {
        job_runner();
    } else {
        husky::LOG_I << "Dense data format is not supported";
    }
}

int main(int argc, char** argv) {
    std::vector<std::string> args({"hdfs_namenode", "hdfs_namenode_port", "train", "test", "B", "C", "format", "is_sparse",
                                   "max_iter", "max_inn_iter"});
    if (husky::init_with_args(argc, argv, args)) {
        husky::run_job(init);
        return 0;
    }
    return 1;
}
