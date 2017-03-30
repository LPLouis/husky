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
#define EPS 1.0e-3

static int bqo_svm_xsp_counter = 0;

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
    // cache of train_set extracted by d
    std::vector<std::vector<SparseVector<double>>> xsp;
    double* normalized_fea;
};

class model {
public:
    int B;
    double C;
    int max_out_iter;
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

class solu_xsp {
public:
    double obj;
    DenseVector<double> alpha;
    // note that w is uncontrolled
    DenseVector<double> w;
    // sparse d + sparse d equals to dense d, so may be don't store \sum_mu_dt may be better?
    // wt in wt_list is controlled but unweighted
    std::vector<DenseVector<double>> wt_list;
    solu_xsp() {}

    solu_xsp(int l, int n, int T) {
        obj = 0.0;
        alpha = DenseVector<double>(l, 0.0);
        wt_list = std::vector<DenseVector<double>>(T);
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
    static void my_max(const DenseVector<double>& vet, int size, double *max_value, int *max_index)
    {
        int i;
        double tmp = vet[0];
        max_index[0] = 0;

        for(i=0; i<size; i++){
            if(vet[i] > tmp){
                tmp = vet[i];
                max_index[0] = i;
            }
        }
        max_value[0] = tmp;
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
    model_->max_out_iter = std::stoi(husky::Context::get_param("max_out_iter"));

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

SparseVector<double> find_most_violated_xsp(data* data_, model* model_, solu_xsp* solu_xsp_ = NULL) {
    int i, j;
    int B = model_->B;
    auto& train_set = data_->train_set;
    DenseVector<double> alpha;
    DenseVector<double> w(data_->n);
    if (solu_xsp_ == NULL) {
        // set alpha to 1 because we do not know how much data others have
        alpha = DenseVector<double>(data_->l, 1.0);
        w = DenseVector<double>(data_->n, 0.0);
    } else {
        for (i = 0; i < model_->mu_set.size(); i++) {
            const auto& tmp_wt = solu_xsp_->wt_list[i];
            if (i == 0) {
                for (j = 0; j < data_->n; j++) {
                    w[j] = tmp_wt[j];
                }
            } else {
                w[j] += tmp_wt[j];
            }
        }
        // w = solu_xsp_->w;
    }

    std::vector<std::pair<int, double>> fea_score;

    if (solu_xsp_ == NULL) {
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

void cache_xsp(data* data_, const SparseVector<double>& dt) {
    auto& train_set_data = data_->train_set->get_data();

    auto& xsp = data_->xsp;
    int size = xsp.size();
    xsp.push_back(std::vector<SparseVector<double>>());
    for (auto& labeled_point : train_set_data) {
        auto& xi = labeled_point.x;

        auto xi_sp = vector_operator::elem_wise_dot(xi, dt);
        xsp[size].push_back(std::move(xi_sp));
    }
}

void bqo_svm_xsp(data* data_, model* model_, solu_xsp* output_solu_xsp_, solu_xsp* input_solu_xsp_ = NULL, double* QD = NULL, int* index = NULL, bool cache = false) {
    int i, j, k;

    const auto& train_set = data_->train_set;
    const auto& train_set_data = train_set->get_data();
    const auto& xsp = data_->xsp;
    const auto& mu_set = model_->mu_set;
    const auto& dt_set = model_->dt_set;

    const double C = model_->C;
    const int W = data_->W;
    const int tid = data_->tid;
    const int n = data_->n;
    const int l = data_->l;
    const int N = data_->N;
    const int T = mu_set.size();
    const int index_low = data_->idx_l;
    const int index_high = data_->idx_h;
    const int max_iter = model_->max_iter;
    const int max_inn_iter = model_->max_inn_iter;

    int iter_out, inn_iter;

    double diag = 0.5 / C;
    double old_primal, primal, obj, grad_alpha_inc;
    double loss, reg = 0;
    double init_primal = C * N;
    double sum_alpha_inc;
    double w_inc_square;
    // std::vector<double>w_inc_square(T);
    double w_dot_w_inc;
    // std::vector<double>w_dot_w_inc(T);
    double alpha_inc_square;
    double alpha_inc_dot_alpha;
    double max_step;
    double eta;

    double G;
    double gap;
    // projected gradient
    double PG;

    DenseVector<double> alpha(l);
    double* alpha_orig = new double[l];
    double* alpha_inc = new double[l];
    // DenseVector<double> alpha_orig(l);
    // DenseVector<double> alpha_inc(l);
    // DenseVector<double> w(n, 0.0);
    // DenseVector<double> w_orig(n, 0.0);
    // DenseVector<double> delta_w(n, 0.0);
    // DenseVector<double> best_w(n, 0.0);
    // husky::lib::ml::ParameterBucket<double> param_server_w;
    // param_server_w.init(n, 0.0);
    std::vector<DenseVector<double>>wt_list(T);
    std::vector<DenseVector<double>>wt_list_orig(T);
    std::vector<DenseVector<double>>delta_wt_list(T);
    std::vector<DenseVector<double>>best_wt_list(T);
    std::vector<husky::lib::ml::ParameterBucket<double>> param_server_wt_list(T);
    for (i = 0; i < T; i++) {
        wt_list_orig[i] = DenseVector<double>(n);
        delta_wt_list[i] = DenseVector<double>(n);
        best_wt_list[i] = DenseVector<double>(n);
        param_server_wt_list[i].init(n, 0.0);
    }

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

    // if input_solu_xsp_ == NULL => alpha = 0 => wt_list is 0, no need to initialize
    if (input_solu_xsp_ == NULL) {
        for (i = 0; i < l; i++) {
            alpha[i] = 0;
        }
        for (i = 0; i < T; i++) {
            wt_list[i] = DenseVector<double>(n, 0.0);
        }
    } else {
        const auto& input_alpha = input_solu_xsp_->alpha;
        const auto& input_wt_list = input_solu_xsp_->wt_list;
        for (i = 0; i < l; i++) {
            alpha[i] = input_alpha[i];
        }
        for (i = 0; i < T; i++) {
            wt_list[i] = input_wt_list[i];
        }
    }
    if (!cache) {
        QD = new double[l];
        index = new int[l];
        for (i = 0; i < l; i++) {
            QD[i] = 0;
            for (k = 0; k < T; k++) {
                QD[i] += vector_operator::self_dot_product(xsp[k][i]) * mu_set[k];
            }
            QD[i] += diag;
            index[i] = i;
        }
    }

    old_primal = INF;
    obj = 0;

    /*******************************************************************/
    // if input_solu_xsp_ is NULL, then we will initialize alpha with 0 => w will be 0 => primal_obj = C * N, obj = 0
    if (input_solu_xsp_ != NULL) {
        // wt_list should have already been initialize with the input wt_list
        for (i = 0; i < T; i++) {
            reg += vector_operator::self_dot_product(wt_list[i]) * mu_set[i];
        }
        // set parameter server back to 0.0
        // reg = self_dot_product(w);
        reg *= 0.5;
        Aggregator<double> et_alpha_agg(0.0, [](double& a, const double& b) { a += b; });
        et_alpha_agg.to_reset_each_iter();
        for (i = 0; i < l; i++) {
            const auto& labeled_point = train_set_data[i];
            loss = 0;
            for (j = 0; j < T; j++) {
                loss += mu_set[j] * wt_list[j].dot(xsp[j][i]);
            }
            loss *= labeled_point.y * -1;
            loss += 1;
            // loss = 1 - labeled_point.y * w.dot(labeled_point.x);
            if (loss > 0) {
                // l2 loss
                loss_agg.update(C * loss * loss);
            }
            // this is a minomer, actually this calculate both aTa and eTa
            et_alpha_agg.update(alpha[i] * (alpha[i] * diag - 2));
        }
        AggregatorFactory::sync();
        old_primal += reg + loss_agg.get_value();
        obj += 0.5 * et_alpha_agg.get_value() + reg;
    }
    /*******************************************************************/

    iter_out = 0;
    while (iter_out < max_iter) {
        // get parameters for local svm solver
        max_step = INF;
        // w_orig = w;
        for (i = 0; i < T; i++) {
            for (j = 0; j < n; j++) {
                wt_list_orig[i][j] = wt_list[i][j];
            }
        }
        for (i = 0; i < l; i++) {
            alpha_orig[i] = alpha[i];
        }
        // alpha_orig = alpha;
        sum_alpha_inc = 0;
        w_inc_square = 0;
        w_dot_w_inc = 0;
        alpha_inc_square = 0;
        alpha_inc_dot_alpha = 0;

        for (i = 0; i < l; i++) {
            int j = i + std::rand() % (l - i);
            vector_operator::swap(index[i], index[j]);
        }

        // run local svm solver to get local delta alpha
        inn_iter = 0;
        while (inn_iter < max_inn_iter) {

            for (k = 0; k < l; k++) {
                i = index[k];
                double yi = train_set_data[i].y;
                // auto& xi = train_set_data[i].x;

                // G = (w.dot(xi)) * yi - 1 + diag * alpha[i];
                G = 0;
                for (j = 0; j < T; j++) {
                    G += mu_set[j] * wt_list[j].dot(xsp[j][i]);
                }
                G = G * yi - 1 + diag * alpha[i];

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
                    // w += xi * loss;
                    for (j = 0; j < T; j++) {
                        wt_list[j] += xsp[j][i] * loss;
                    }
                }
            }
            inn_iter++;
        }
        for (i = 0; i < l; i++) {
            alpha_inc[i] = alpha[i] - alpha_orig[i];
        }
        // alpha_inc = alpha - alpha_orig;
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

        // this is very inefficient as most of the entries will be 0
        for (i = 0; i < T; i++) {
            param_server_wt_list[i].init(n, 0.0);
            for (j = 0; j < n; j++) {
                param_server_wt_list[i].update(j, wt_list[i][j] - wt_list_orig[i][j]);
            }
        }
        // for (i = 0; i < n; i++) {
        //     param_server_w.update(i, w[i] - w_orig[i]);
        // }
        AggregatorFactory::sync();

        sum_alpha_inc = sum_alpha_inc_agg.get_value();
        alpha_inc_square = alpha_inc_square_agg.get_value();
        alpha_inc_dot_alpha = alpha_inc_dot_alpha_agg.get_value();
        max_step = eta_agg.get_value();
        for (i = 0; i < T; i++) {
            const auto& tmp_wt = param_server_wt_list[i].get_all_param();
            for (j = 0; j < n; j++) {
                delta_wt_list[i][j] = tmp_wt[j];
            }
            // delta_wt_list[i] = param_server_wt_list[i].get_all_param();
        }
        // delta_w = param_server_w.get_all_param();
        for (i = 0; i < T; i++) {
            w_inc_square += mu_set[i] * vector_operator::self_dot_product(delta_wt_list[i]);
            w_dot_w_inc += mu_set[i] * wt_list_orig[i].dot(delta_wt_list[i]);
        }
        // w_inc_square += self_dot_product(delta_w);
        // w_dot_w_inc += w_orig.dot(delta_w);
        // get step size
        grad_alpha_inc = w_dot_w_inc + alpha_inc_dot_alpha - sum_alpha_inc;
        if (grad_alpha_inc >= 0) {
            for (i = 0; i < T; i++) {
                for (j = 0; j < n; j++) {
                    wt_list[i][j] = best_wt_list[i][j];
                }
            }
            // w = best_w;
            break;
        }

        double aQa = alpha_inc_square + w_inc_square;
        eta = std::min(max_step, -grad_alpha_inc /aQa);

        for (i = 0; i < l; i++) {
            alpha[i] = alpha_orig[i] + eta * alpha_inc[i];
        }
        // alpha = alpha_orig + eta * alpha_inc;
        for (i = 0; i < T; i++) {
            for (j = 0; j < n; j++) {
                wt_list[i][j] = wt_list_orig[i][j] + eta * delta_wt_list[i][j];
            }
        }
        // w = w_orig + eta * delta_w;

        // f(w) + f(a) will cancel out the 0.5\alphaQ\alpha term (old value)
        obj += eta * (0.5 * eta * aQa + grad_alpha_inc);

        reg += eta * (w_dot_w_inc + 0.5 * eta * w_inc_square);

        primal = 0;

        for (i = 0; i < l; i++) {
            double yi = train_set_data[i].y;
            loss = 0;
            for (j = 0; j < T; j++) {
                loss += wt_list[j].dot(xsp[j][i]);
            }
            loss *= yi * -1;
            loss += 1;
            // loss = 1 - labeled_point.y * w.dot(labeled_point.x);
            if (loss > 0) {
                // l2 loss
                loss_agg.update(C * loss * loss);
            }
            // this is a minomer, actually this calculate both aTa and eTa
        }
        AggregatorFactory::sync();

        primal += reg + loss_agg.get_value();


        if (primal < old_primal) {
            old_primal = primal;
            for (i = 0; i < T; i++) {
                for(j = 0; j < n; j++) {
                    best_wt_list[i][j] = wt_list[i][j];
                }
            }
            // best_w = w;
        }

        gap = (primal + obj) / init_primal;

        if (tid == 0) {
            husky::LOG_I << "primal: " + std::to_string(primal);
            husky::LOG_I << "dual: " + std::to_string(obj);
            husky::LOG_I << "duality_gap: " + std::to_string(gap);
        }

        if (gap < EPS) {
            for (i = 0; i < T; i++) {
                for (j = 0; j < n; j++) {
                    wt_list[i][j] = best_wt_list[i][j];
                }
            }
            // w = best_w;
            break;
        }
        iter_out++;
    }

    output_solu_xsp_->obj = obj;
    output_solu_xsp_->alpha = alpha;
    output_solu_xsp_->wt_list = wt_list;

    if (!cache) {
        delete[] QD;
        delete[] index;
    }
    delete[] alpha_inc;
    delete[] alpha_orig;
}

void evaluate(data* data_, model* model_, solu_xsp* solu_xsp_) {
    const auto& test_set = data_->test_set;
    const auto& wt_list = solu_xsp_->wt_list;
    const auto& mu_set = model_->mu_set;
    DenseVector<double> w_controlled(data_->n, 0.0);
    for (int i = 0; i < mu_set.size(); i++) {
        w_controlled += mu_set[i] * wt_list[i];
    }

    double indicator;
    Aggregator<int> error_agg(0, [](int& a, const int& b) { a += b; });
    Aggregator<int> num_test_agg(0, [](int& a, const int& b) { a += b; });
    auto& ac = AggregatorFactory::get_channel();
    list_execute(*test_set, {}, {&ac}, [&](ObjT& labeled_point) {
        double indicator = w_controlled.dot(labeled_point.x);
        indicator *= labeled_point.y;
        if (indicator <= 0) {
            error_agg.update(1);
        }
        num_test_agg.update(1);
    });

    if (data_->tid == 0) {
        husky::LOG_I << "Classification accuracy on testing set with [B = " + std::to_string(model_->B) + "][C = " + 
                        std::to_string(model_->C) + "], " +
                        "[max_out_iter = " + std::to_string(model_->max_out_iter) + "], " +
                        "[max_iter = " + std::to_string(model_->max_iter) + "], " +
                        "[max_inn_iter = " + std::to_string(model_->max_inn_iter) + "], " +
                        "[test set size = " + std::to_string(num_test_agg.get_value()) + "]: " +
                        std::to_string(1.0 - static_cast<double>(error_agg.get_value()) / num_test_agg.get_value());  
    }
}

void run_bqo_svm() {
    data* data_ = new data;
    model* model_ = new model;
    solu_xsp* solu_xsp_ = new solu_xsp;
    initialize(data_, model_);
    SparseVector<double> dt1(data_->n);
    SparseVector<double> dt2(data_->n);
    SparseVector<double> dt3(data_->n);
    dt1.set(7 ,1.0);
    dt1.set(22 ,1.0);
    dt1.set(28 ,1.0);
    dt1.set(31 ,1.0);
    dt1.set(74 ,1.0);

    dt2.set(7, 1.0);
    dt2.set(8, 1.0);
    dt2.set(45 ,1.0);
    dt2.set(76 ,1.0);
    dt2.set(78 ,1.0);

    dt3.set(12 ,1.0);
    dt3.set(59 ,1.0);
    dt3.set(60 ,1.0);
    dt3.set(83 ,1.0);
    dt3.set(90 ,1.0);

    model_->dt_set.push_back(dt1);
    model_->dt_set.push_back(dt2);
    model_->dt_set.push_back(dt3);
    model_->mu_set.push_back(1.0);
    model_->mu_set.push_back(1.0);
    model_->mu_set.push_back(1.0);
    cache_xsp(data_, dt1);
    cache_xsp(data_, dt2);
    cache_xsp(data_, dt3);
    for (int i = 0; i < 3; i++) {
        model_->mu_set[i] /= 3.0;
    }
    bqo_svm_xsp(data_, model_, solu_xsp_);
    vector_operator::show(model_->mu_set, "mu_set");
    husky::LOG_I << "trainning objective: " + std::to_string(solu_xsp_->obj);
    evaluate(data_, model_, solu_xsp_);
}

void simpleMKL(data* data_, model* model_, solu_xsp* solu_) {
    assert(model_->mu_set.size() == model_->dt_set.size() && data_->xsp.size() == model_->mu_set.size() && "size of mu_set, dt_set and xsp do not agree");
    
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
    const auto& train_set_data = data_->train_set->get_data();
    const auto& xsp = data_->xsp;
    double diag = 0.5 / model_->C;
    double* QD = new double[l];
    int* index = new int[l];
    for (i = 0; i < l; i++) {
        QD[i] = 0;
        for (j = 0; j < T; j++) {
            QD[i] += vector_operator::self_dot_product(xsp[j][i]) * mu_set[j];
        }
        QD[i] += diag;
        index[i] = i;
    }

    bqo_svm_xsp(data_, model_, solu_, NULL, QD, index, true);
    double obj = solu_->obj;
    auto& alpha = solu_->alpha;
    auto& wt_list = solu_->wt_list;
    // compute gradient
    DenseVector<double> grad(T);
    for (i = 0; i < T; i++) {
        // grad[i] = -0.5 * vector_operator::self_dot_product(solu_->wt_list[i]);
        grad[i] = -0.5 * vector_operator::self_dot_product(wt_list[i]);
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

    solu_xsp* tmp_solu = new solu_xsp(l, n, T);
    solu_xsp* tmp_ls_solu_1 = new solu_xsp(l, n, T);
    solu_xsp* tmp_ls_solu_2 = new solu_xsp(l, n, T);

    while (loop == 1 && maxloop > 0 && T > 1) {
        nloop++;

        double old_obj = obj;
        husky::LOG_I << "[outer loop: " + std::to_string(nloop) + "][old_obj]: " + std::to_string(old_obj);

        // initialize a new model
        for (i = 0; i < T; i++) {
            new_mu_set[i] = mu_set[i];
        }
        // new_model->mu_set = mu_set;

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
                husky::LOG_I << "[inner loop][cost_max]: " + std::to_string(cost_max);
                for (i = 0; i < T; i++) {
                    tmp_mu_set[i] = new_mu_set[i] + step_max * desc[i];
                }
                // use descent direction to compute new objective
                // consider modifying input solution to speed up
                bqo_svm_xsp(data_, tmp_model, tmp_solu, solu_, QD, index, true);
                cost_max = tmp_solu->obj;
                if (cost_max < cost_min) {
                    cost_min = cost_max;

                    for (i = 0; i < T; i++) {
                        new_mu_set[i] = tmp_mu_set[i];
                        mu_set[i] = tmp_mu_set[i];
                    }

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
                    for (i = 0; i < l; i++) {
                        alpha[i] = tmp_solu->alpha[i];
                    }
                    for (i = 0; i < T; i++) {
                        const auto& tmp_wt = tmp_solu->wt_list[i];
                        for (j = 0; j < n; j++) {
                            wt_list[i][j] = tmp_wt[j];
                        }
                        // solu_->wt_list[i] = tmp_solu->wt_list[i];
                    }

                    if (fflag) {
                        step_max = 0;
                        delta_max = 0;
                    } else {
                        // we can still descend, loop again
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
                husky::LOG_I << "[line_search] iteration: " + std::to_string(step_loop);
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
                bqo_svm_xsp(data_, tmp_ls_model_1, tmp_ls_solu_1, solu_, QD, index, true);

                // half half
                for (i = 0; i < T; i++) {
                    tmp_ls_mu_set_2[i] = new_mu_set[i] + step_medl * desc[i];
                }
                bqo_svm_xsp(data_, tmp_ls_model_2, tmp_ls_solu_2, solu_, QD, index, true);

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
                        for (i = 0; i < T; i++) {
                            const auto& tmp_wt = tmp_ls_solu_2->wt_list[i];
                            for (j = 0; j < n; j++) {
                                wt_list[i][j] = tmp_wt[j];
                            }
                            // solu_->wt_list[i] = tmp_ls_solu_2->wt_list[i];
                        }
                        for (i = 0; i < l; i++) {
                            solu_->alpha[i] = tmp_ls_solu_2->alpha[i];
                        }
                    break;

                    case 1:
                        step_max = step_medr;
                        cost_max = cost[2];
                        solu_->obj = tmp_ls_solu_1->obj; 
                        for (i = 0; i < T; i++) {
                            const auto& tmp_wt = tmp_ls_solu_1->wt_list[i];
                            for (j = 0; j < n; j++) {
                                wt_list[i][j] = tmp_wt[j];
                            }
                            // solu_->wt_list[i] = tmp_ls_solu_2->wt_list[i];
                        }
                        for (i = 0; i < l; i++) {
                            solu_->alpha[i] = tmp_ls_solu_1->alpha[i];
                        }
                    break;

                    case 2:
                        step_min = step_medl;
                        cost_min = cost[1];
                        solu_->obj = tmp_ls_solu_2->obj;
                        for (i = 0; i < T; i++) {
                            const auto& tmp_wt = tmp_ls_solu_2->wt_list[i];
                            for (j = 0; j < n; j++) {
                                wt_list[i][j] = tmp_wt[j];
                            }
                            // solu_->wt_list[i] = tmp_ls_solu_2->wt_list[i];
                        }
                        for (i = 0; i < l; i++) {
                            solu_->alpha[i] = tmp_ls_solu_2->alpha[i];
                        }                      
                    break;

                    case 3:
                        step_min = step_medr;
                        cost_min = cost[2];
                        solu_->obj = tmp_ls_solu_1->obj;
                        for (i = 0; i < T; i++) {
                            const auto& tmp_wt = tmp_ls_solu_1->wt_list[i];
                            for (j = 0; j < n; j++) {
                                wt_list[i][j] = tmp_wt[j];
                            }
                            // solu_->wt_list[i] = tmp_ls_solu_2->wt_list[i];
                        }
                        for (i = 0; i < l; i++) {
                            solu_->alpha[i] = tmp_ls_solu_1->alpha[i];
                        }                    
                    break;
                }// switch(min_idx);      
            } // while ((step_max - step_min) > 1e-1 * fabs(delta_max) && step_max > 1e-12)

            // assignment
            double step_size = step[min_idx];
            if (solu_->obj < old_obj) {
                for (i = 0; i < T; i++) {
                    new_mu_set[i] += step_size * desc[i];
                    mu_set[i] = new_mu_set[i];
                }
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


        for (i = 0; i < T; i++) {
            grad[i] = -0.5 * vector_operator::self_dot_product(wt_list[i]);
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

        husky::LOG_I << "min_grad: " + std::to_string(min_grad) + ", max_grad: " + std::to_string(max_grad);
        double KKTconstraint = fabs(min_grad - max_grad) / fabs(min_grad);
        // note we find min idx in grad, corresponding to max idx in -grad

        DenseVector<double> tmp_grad(T);
        for (i = 0; i < T; i++) {
            tmp_grad[i] = -1 * grad[i];
        }
        double max_tmp;
        int max_tmp_idx;
        vector_operator::my_max(tmp_grad, T, &max_tmp, &max_tmp_idx);
        double tmp_sum = 0;
        for (i = 0; i < l; i++) {
            tmp_sum += solu_->alpha[i];
        }

        double dual_gap = (solu_->obj + max_tmp - tmp_sum) / solu_->obj;
        husky::LOG_I << "[outer loop][dual_gap]: " + std::to_string(fabs(dual_gap));
        husky::LOG_I << "[outer loop][KKTconstraint]: " + std::to_string(KKTconstraint);
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

void run_simple_mkl() {
    data* data_ = new data;
    model* model_ = new model;
    solu_xsp* solu_ = new solu_xsp;
    initialize(data_, model_);
    SparseVector<double> dt1(data_->n);
    SparseVector<double> dt2(data_->n);
    SparseVector<double> dt3(data_->n);
    dt1.set(7 ,1.0);
    dt1.set(22 ,1.0);
    dt1.set(28 ,1.0);
    dt1.set(31 ,1.0);
    dt1.set(74 ,1.0);

    dt2.set(7, 1.0);
    dt2.set(8, 1.0);
    dt2.set(45 ,1.0);
    dt2.set(76 ,1.0);
    dt2.set(78 ,1.0);

    dt3.set(12 ,1.0);
    dt3.set(59 ,1.0);
    dt3.set(60 ,1.0);
    dt3.set(83 ,1.0);
    dt3.set(90 ,1.0);

    model_->dt_set.push_back(dt1);
    model_->dt_set.push_back(dt2);
    model_->dt_set.push_back(dt3);
    model_->mu_set.push_back(1.0);
    model_->mu_set.push_back(1.0);
    model_->mu_set.push_back(1.0);
    cache_xsp(data_, dt1);
    cache_xsp(data_, dt2);
    cache_xsp(data_, dt3);
    for (int i = 0; i < 3; i++) {
        model_->mu_set[i] /= 3.0;
    }

    // simpleMKL_old(data_, model_, solu_);
    simpleMKL(data_, model_, solu_);
    if (data_->tid == 0) {
        vector_operator::show(model_->mu_set, "mu_set");
        husky::LOG_I << "trainning objective: " + std::to_string(solu_->obj);
    }
    evaluate(data_, model_, solu_);    
}

void job_runner() {
    data* data_ = new data;
    model* model_ = new model;
    solu_xsp* solu_xsp_ = new solu_xsp;
    initialize(data_, model_);

    int iter = 0;
    int max_out_iter = model_->max_out_iter;
    double mkl_obj = INF;
    double last_obj = INF;
    double obj_diff = 0.0;
    SparseVector<double> dt(data_->n);
    while(iter < max_out_iter) {
        last_obj = mkl_obj;
        if (iter == 0) {
            dt = find_most_violated_xsp(data_, model_);
        } else {
            dt = find_most_violated_xsp(data_, model_, solu_xsp_);
        }
        // if (vector_operator::elem_at(dt, model_->dt_set)) {
        //     if (data_->tid == 0) {
        //         husky::LOG_I << "FGM converged";
        //     }
        //     break;
        // }
        model_->dt_set.push_back(dt);
        model_->mu_set.push_back(1.0);
        cache_xsp(data_, dt);
        husky::LOG_I << "cache completed! number of kernel: " + std::to_string(data_->xsp.size());
        simpleMKL(data_, model_, solu_xsp_);
        mkl_obj = solu_xsp_->obj;
        obj_diff = fabs(mkl_obj - last_obj);
        if (data_->tid == 0) {
            husky::LOG_I << "[iteration " + std::to_string(iter) + "][mkl_obj " + std::to_string(mkl_obj) + "][obj_diff " + std::to_string(obj_diff) + "]";
            vector_operator::show(model_->mu_set, "mu_set");
        }
        if (obj_diff < 0.001 * abs(last_obj)) {
            if (data_->tid == 0) {
                husky::LOG_I << "FGM converged";
            }
            break;
        }
        if (model_->mu_set[iter] < 0.0001) {
            if (data_->tid == 0) { 
                husky::LOG_I << "FGM converged";
            }
            break;
        }
        iter++;
        evaluate(data_, model_, solu_xsp_);
    }
    if (data_->tid == 0) {
        vector_operator::show(model_->mu_set, "mu_set");
    }
    evaluate(data_, model_, solu_xsp_);
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
                                   "max_out_iter", "max_iter", "max_inn_iter"});
    if (husky::init_with_args(argc, argv, args)) {
        husky::run_job(init);
        return 0;
    }
    return 1;
}
