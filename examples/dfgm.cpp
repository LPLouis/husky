/***
    1. Use one parameter server to compute all the aggregations?
***/
#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>
#include <set>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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
#define EPS 1.0e-2
#define MAX_NUM_KERNEL 20


struct feature_node
{
    int index;  // index of the feature inside wsp, inside the range of [0, B)
    double value;
};

class data 
{
public:
    int n;          // global number of features (appended 1 included)
    int N;          // global number of samples
    int l;
    int W;          // number of workers
    int tid;        // global tid of the worker
    int idx_l;      // index_low
    int idx_h;      // index_high     
    husky::ObjList<ObjT>* train_set;
    husky::ObjList<ObjT>* test_set;

    feature_node ***xsp;
    double* norm;
    // allocate memory in advance
    data()
    {
        xsp = new feature_node**[MAX_NUM_KERNEL];
    }

    ~data()
    {
        if (xsp != NULL)
        {
            delete [] xsp;
        }
        if (norm != NULL)
        {
            delete [] norm;
        }
    }
};

class model 
{
public:
    int B;
    double C;
    int n_kernel;
    int max_out_iter;
    int max_iter;
    int max_inn_iter;
    double *mu_set;
    int** dt_set;

    model()
    {
        n_kernel = 0;
        mu_set = new double[MAX_NUM_KERNEL];
        dt_set = new int*[MAX_NUM_KERNEL];
    }

    model(model* model_)
    {
        B = model_->B;
        C = model_->C;
        n_kernel = model_->n_kernel;
        max_out_iter = model_->max_out_iter;
        max_iter = model_->max_iter;
        max_inn_iter = model_->max_inn_iter;
        mu_set = new double[MAX_NUM_KERNEL];
        dt_set = new int*[MAX_NUM_KERNEL];

        for (int i = 0; i < n_kernel; i++)
        {
            mu_set[i] = model_->mu_set[i];
        }
        for (int i = 0; i < n_kernel; i++)
        {
            for (int j = 0; j < B; j++)
            {
                dt_set[i][j] = model_->dt_set[i][j];
            }
        }
    }

    void add_dt(int *dt, int B)
    {
        assert(n_kernel <= MAX_NUM_KERNEL && "n_kernel exceeds MAX_NUM_KERNEL");
        dt_set[n_kernel] = dt;
        n_kernel += 1;
        std::fill_n(mu_set, n_kernel, 1.0 / n_kernel);
    }

    ~model()
    {
        if (mu_set != NULL)
        {
            delete [] mu_set;
        }
        if (dt_set != NULL)
        {
            for (int i = 0; i < n_kernel; i++)
            {
                delete [] dt_set[i];
            }
            delete [] dt_set;
        }
    }
};

class solu 
{
public:
    double obj;
    double *alpha;
    double *wsp;

    solu() 
    {
        alpha = NULL;
        wsp = NULL;
    }

    ~solu()
    {
        if (alpha != NULL)
        {
            delete [] alpha;
        }
        if (wsp != NULL)
        {
            delete [] wsp;
        }
    }
};

class vector_operator 
{
public:

    static bool sparse_equal(int* dt, int* v, int B)
    {
        bool flag = true;
        for (int i = 0; i < B; i++)
        {
            if (dt[i] != v[i])
            {
                flag = false;
                break;
            }
        }
        return flag;
    }

    static bool element_at(int* dt, int **dt_set, int n_kernel, int B)
    {
        bool flag = false;
        for (int i = 0; i < n_kernel; i++)
        {
            if (sparse_equal(dt, dt_set[i], B))
            {
                flag = true;
                break;
            }
        }
        return flag;
    }

    static inline bool double_equals(double a, double b, double epsilon = 1.0e-6) {
        return std::abs(a - b) < epsilon;
    }

    static void show(const std::vector<double>& vec, std::string message_head) 
    {
        std::string ret = message_head;
        for (int i = 0; i < vec.size(); i++) {
            ret += ": (" + std::to_string(i + 1) + ", " + std::to_string(vec[i]) + "),";
        }
        husky::LOG_I << ret;
    }

    static void show(const DenseVector<double>& vec, std::string message_head) 
    {
        std::string ret = message_head + ": ";
        for (int i = 0; i < vec.get_feature_num(); i++) {
            ret += std::to_string(vec[i]) + " "; 
        }
        husky::LOG_I << ret;
    }

    static void show(const SparseVector<double>& vec, std::string message_head) 
    {
        std::string ret = message_head + ": ";
        for (auto it = vec.begin(); it != vec.end(); it++) {
            ret += std::to_string((*it).fea) + ":" + std::to_string((*it).val) + " ";
        }
        husky::LOG_I << ret;
    }

    static void show(int *dt, int B, std::string message_head)
    {
        std::string ret = message_head + ": ";
        for (int i = 0; i < B; i++)
        {
            ret += std::to_string(i) + ":" + std::to_string(dt[i]) + " ";
        }
        husky::LOG_I << ret;
    }

    static void show(double *dt, int B, std::string message_head)
    {
        std::string ret = message_head + ": ";
        for (int i = 0; i < B; i++)
        {
            ret += std::to_string(i) + ":" + std::to_string(dt[i]) + " ";
        }
        husky::LOG_I << ret;
    }

    template <typename T>
    static inline void swap(T& a, T& b) {
        T t = a;
        a = b;
        b = t;
    }

    static void my_min(const double *vet, const int size, double *min_value, int *min_index) 
    {
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

    static void my_soft_min(const double* mu_set, const double* desc, const int n_kernel, double* step_max)
    {
        int i;
        int flag = 1;
        for(i = 0; i < n_kernel; i++)
        {
            if(desc[i] < 0)
            {
                if(flag == 1)
                {
                    step_max[0] = -mu_set[i] / desc[i];
                    flag = 0;
                }
                else
                {
                    double tmp = -mu_set[i] / desc[i];
                    if (tmp < step_max[0])
                    {
                        step_max[0] = tmp;
                    }
                }
            }
        }
    }

    static void my_max(const double* vet, const int size, double *max_value, int *max_index)
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

    static double dot_product(const double* w, const SparseVector<double>& v, const int n)
    {
        assert(v.get_feature_num() == n && "dot_product: error\n");
        double ret = 0;
        auto it = v.begin();
        while(it != v.end())
        {
            ret += w[(*it).fea] * (*it).val;
            it++;
        }
        return ret;
    }
};

void initialize(data* data_, model* model_) 
{
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
    // uncomment the following if bias is needed (note we will train the bias instead of obtaining it after w is solved for sake of simplicity)
    // for (auto& labeled_point : train_set.get_data()) {
    //     labeled_point.x.resize(n + 1);
    //     labeled_point.x.set(n, 1);
    // }
    // for (auto& labeled_point : test_set.get_data()) {
    //     labeled_point.x.resize(n + 1);
    //     labeled_point.x.set(n, 1);
    // }
    // n += 1;
    data_->n = n;

    // get model config parameters
    model_->B = std::stoi(husky::Context::get_param("B"));
    model_->C = std::stod(husky::Context::get_param("C"));
    model_->max_iter = std::stoi(husky::Context::get_param("max_iter"));
    model_->max_inn_iter = std::stoi(husky::Context::get_param("max_inn_iter"));
    model_->max_out_iter = std::stoi(husky::Context::get_param("max_out_iter"));

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

    double *norm = new double[n];
    std::fill_n(norm, n, 0);
    husky::lib::ml::ParameterBucket<double> param_server(data_->n);
    for (auto& labeled_point : train_set.get_data())
    {
        auto it = labeled_point.x.begin();
        while (it != labeled_point.x.end())
        {
            param_server.update((*it).fea, (*it).val * (*it).val);
            it++;
        }
    }
    AggregatorFactory::sync();
    const auto& tmp_param_server = param_server.get_all_param();
    for (int i = 0; i < n; i++)
    {
        norm[i] = sqrt(tmp_param_server[i]);
    }
    data_->norm = norm;

    if (l != 0) 
    {
        husky::LOG_I << "Worker " + std::to_string(data_->tid) + " holds sample [" + std::to_string(index_low) + ", " + std::to_string(index_high) + ")";
        husky::LOG_I << "Worker " + std::to_string(data_->tid) + " holds " + std::to_string(l) + " sample ";
    }

    if (tid == 0) {
        husky::LOG_I << "Number of samples: " + std::to_string(N);
        husky::LOG_I << "Number of features: " + std::to_string(n);
    }
    return;
}

void destroy(data* data_, model* model_, solu* solu_)
{
    int i, j;
    int l = data_->l;
    int n_kernel = model_->n_kernel;

    for (i = 0; i < n_kernel; i++)
    {
        for (j = 0; j < l; j++)
        {
            delete [] data_->xsp[i][j];
        }
        delete [] data_->xsp[i];
    }
}

int* most_violated(data* data_, model* model_, solu* solu_ = NULL) 
{
    int i, j;
    auto& train_set_data = data_->train_set->get_data();
    double* alpha;
    double* norm = data_->norm;

    if (solu_ == NULL) 
    {
        // set alpha to 1 because we do not know how much data others have
        alpha = new double[data_->l];
        std::fill_n(alpha, data_->l, 1.0);
    } 
    else 
    {
        alpha = solu_->alpha;
    }

    std::vector<std::pair<int, double>> fea_score;
    husky::lib::ml::ParameterBucket<double> param_server(data_->n);

    double* w = new double[data_->n];
    std::fill_n(w, data_->n, 0);
    for (i = 0; i < data_->l; i++)
    {
        double alphay = train_set_data[i].y * alpha[i];
        auto it = train_set_data[i].x.begin();
        while (it != train_set_data[i].x.end())
        {
            w[(*it).fea] += alphay * (*it).val;
            it++;
        }
        // vector_operator::add_sparse(w, train_set_data[i].x, train_set_data[i].y * alpha[i], data_->n);
    }

    for (i = 0; i < data_->n; i++) 
    {
        param_server.update(i, w[i]);
    }
    AggregatorFactory::sync();
    const auto& tmp_w = param_server.get_all_param();
    for (i = 0; i < data_->n; i++) 
    {
        if (norm[i] == 0)
        {
            fea_score.push_back(std::make_pair(i, 0));
        }
        else
        {
            fea_score.push_back(std::make_pair(i, fabs(tmp_w[i]) / norm[i]));
        }
    }
    std::sort(fea_score.begin(), fea_score.end(), [](auto& left, auto& right) {
        return left.second > right.second;
    });

    int *dt = new int[model_->B];
    for (i = 0; i < model_->B; i++) {
        int fea = fea_score[i].first;
        dt[i] = fea;
    }

    delete [] w;
    if (solu_ == NULL)
    {
        delete [] alpha;
    }
    
    std::sort(dt, dt + model_->B);
    return dt;
}

// this function caches the new kernel corresponding to the given dt
// this function modifies n_kernel, mu_set and dt_set inside model
void cache_xsp(data* data_, model* model_, int *dt, int B) 
{
    int i, j;
    int l = data_->l;
    int n_kernel = model_->n_kernel;
    auto& train_set_data = data_->train_set->get_data();

    // cache new kernel
    data_->xsp[n_kernel] = new feature_node*[l];
    for (i = 0; i < l; i++)
    {
        auto xi = train_set_data[i].x;
        // at most B + 1 elements;
        feature_node* tmp = new feature_node[B + 1];
        feature_node* tmp_orig = tmp;
        auto it = xi.begin();
        j = 0;
        while (it != xi.end() && j != B)
        {
            int fea = (*it).fea;
            int dt_idx = dt[j];
            if (dt_idx == fea)
            {
                tmp->index = j;
                tmp->value = (*it).val;
                j++;
                it++;
                tmp++;
            }
            else if (dt_idx < fea)
            {
                j++;
            }
            else
            {
                it++;
            }
        }
        tmp->index = -1;
        data_->xsp[n_kernel][i] = tmp_orig;
    }
    // modify model_
    model_->add_dt(dt, B);
}

void bqo_svm(data* data_, model* model_, solu* output_solu_, solu* input_solu_ = NULL, double* QD = NULL, int* index = NULL, bool cache = false) 
{
    int i, j, k;
    double tmp, tmp2;

    const auto& train_set_data = data_->train_set->get_data();
    double* mu_set = model_->mu_set;
    int** dt_set = model_->dt_set;
    feature_node*** xsp = data_->xsp;

    const double C = model_->C;
    const int B = model_->B;
    const int n_kernel = model_->n_kernel;
    const int W = data_->W;
    const int tid = data_->tid;
    const int n = data_->n;
    const int l = data_->l;
    const int N = data_->N;
    const int index_low = data_->idx_l;
    const int index_high = data_->idx_h;
    const int max_iter = model_->max_iter;
    const int max_inn_iter = model_->max_inn_iter;

    int iter_out, inn_iter;

    double diag = 0.5 / C;
    double old_primal, primal, obj, grad_alpha_inc;
    double loss, reg = 0;
    double init_primal = C * N;
    double w_inc_square;
    double w_dot_w_inc;

    double sum_alpha_inc;
    double alpha_inc_square;
    double alpha_inc_dot_alpha;
    double sum_alpha_inc_org;
    double alpha_inc_square_org;
    double alpha_inc_dot_alpha_org;

    double max_step;
    double eta;

    double G;
    double gap;
    // projected gradient
    double PG;

    double* alpha = new double[l];
    double* alpha_orig = new double[l];
    double* alpha_inc = new double[l];
    double* wsp = new double[n_kernel * B];
    double* wsp_orig = new double[n_kernel * B];
    double* delta_wsp = new double[n_kernel * B];
    double* best_wsp = new double[n_kernel * B];
    // 3 for sum_alpha_inc, alpha_inc_square and alpha_inc_dot_alpha respectively
    husky::lib::ml::ParameterBucket<double> param_list(n_kernel * B + 3);
    Aggregator<double> loss_agg(0.0, [](double& a, const double& b) { a += b; });
    loss_agg.to_reset_each_iter();
    Aggregator<double> eta_agg(INF, [](double& a, const double& b) { a = std::min(a, b); }, [](double& a) { a = INF; });
    eta_agg.to_reset_each_iter();

    // if input_solu_ == NULL => alpha = 0 => wt_list is 0, no need to initialize
    if (input_solu_ == NULL) 
    {
        std::fill_n(alpha, l, 0);
        std::fill_n(wsp, n_kernel * B, 0);
    } 
    else 
    {
        for (i = 0; i < l; i++) 
        {
            alpha[i] = input_solu_->alpha[i];
        }
        for (i = 0; i < n_kernel * B; i++) 
        {
            wsp[i] = input_solu_->wsp[i];
        }
    }
    if (!cache) 
    {
        QD = new double[l];
        index = new int[l];
        for (i = 0; i < l; i++) {
            QD[i] = 0;
            for (k = 0; k < n_kernel; k++) {
                tmp = 0;
                feature_node* x = xsp[k][i];
                while (x->index != -1)
                {
                    tmp += x->value * x->value;
                    x++;
                }
                QD[i] += mu_set[k] * tmp;
                // QD[i] += mu_set[k] * vector_operator::self_dot_product(xsp[k][i], B);
            }
            QD[i] += diag;
            index[i] = i;
        }
    }

    old_primal = INF;
    obj = 0;

    /*******************************************************************/
    // if input_solu_ is NULL, then we will initialize alpha with 0 => w will be 0 => primal_obj = C * N, obj = 0
    if (input_solu_ != NULL) 
    {
        for (i = 0; i < n_kernel; i++) 
        {
            tmp = 0;
            double* tmp_wsp = &wsp[i * B];
            for (j = 0; j < B; j++)
            {
                // tmp += wsp[i * B + j] * wsp[i * B + j];
                tmp += tmp_wsp[j] * tmp_wsp[j];
            }
            reg += mu_set[i] * tmp;
        }

        // reg = self_dot_product(w);
        reg *= 0.5;
        Aggregator<double> et_alpha_agg(0.0, [](double& a, const double& b) { a += b; });
        et_alpha_agg.to_reset_each_iter();
        for (i = 0; i < l; i++) {
            const auto& labeled_point = train_set_data[i];
            loss = 0;
            // loss = 1 - labeled_point.y * w.dot(labeled_point.x);
            for (j = 0; j < n_kernel; j++) 
            {
                tmp = 0;
                double* tmp_wsp = &wsp[j * B];
                feature_node* x = xsp[j][i];
                while (x->index != -1)
                {
                    tmp += tmp_wsp[x->index] * x->value;
                    x++;
                }
                loss += mu_set[j] * tmp;
                // loss += mu_set[j] * vector_operator::dot_product(&wsp[j * B], xsp[j][i], B);
            }
            loss *= labeled_point.y * -1;
            loss += 1;
            if (loss > 0)
            {
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
    std::fill_n(delta_wsp, n_kernel * B, 0);
    iter_out = 0;
    while (iter_out < max_iter) {
        // get parameters for local svm solver
        max_step = INF;
        // w_orig = w;
        for (i = 0; i < n_kernel * B; i++) 
        {
            wsp_orig[i] = wsp[i];
        }
        for (i = 0; i < l; i++) 
        {
            alpha_orig[i] = alpha[i];
        }
        // alpha_orig = alpha;
        w_inc_square = 0;
        w_dot_w_inc = 0;

        if (iter_out == 0)
        {
            sum_alpha_inc_org = 0;
            alpha_inc_square_org = 0;
            alpha_inc_dot_alpha_org = 0;
        }
        else
        {
            sum_alpha_inc_org = sum_alpha_inc;
            alpha_inc_square_org = alpha_inc_square;
            alpha_inc_dot_alpha_org = alpha_inc_dot_alpha;
        }
        sum_alpha_inc = 0;
        alpha_inc_square = 0;
        alpha_inc_dot_alpha = 0;

        for (i = 0; i < l; i++) 
        {
            int j = i + std::rand() % (l - i);
            vector_operator::swap(index[i], index[j]);
        }

        // run local svm solver to get local delta alpha
        inn_iter = 0;
        while (inn_iter < max_inn_iter) 
        {
            for (k = 0; k < l; k++) 
            {
                i = index[k];
                double yi = train_set_data[i].y;

                G = 0;
                for (j = 0; j < n_kernel; j++) 
                {
                    tmp = 0;
                    double* tmp_wsp = &wsp[j * B];
                    feature_node* x = xsp[j][i];
                    while (x->index != -1)
                    {
                        tmp += tmp_wsp[x->index] * x->value;
                        x++;
                    }
                    G += mu_set[j] * tmp;
                    // G += mu_set[j] * vector_operator::dot_product(&wsp[j * B], xsp[j][i], B);
                }
                G = G * yi - 1 + diag * alpha[i];

                PG = 0;
                if (alpha[i] == 0) 
                {
                    if (G < 0) 
                    {
                        PG = G;
                    }
                } 
                else if (alpha[i] == INF)
                {
                    if (G > 0) 
                    {
                        PG = G;
                    }
                } 
                else 
                {
                    PG = G;
                }

                if (fabs(PG) > 1e-12) 
                {
                    double alpha_old = alpha[i];
                    alpha[i] = std::min(std::max(alpha[i] - G / QD[i], 0.0), INF);
                    loss = yi * (alpha[i] - alpha_old);
                    for (j = 0; j < n_kernel; j++) 
                    {
                        double* tmp_wsp = &wsp[j * B];
                        feature_node* x = xsp[j][i];
                        while (x->index != -1)
                        {
                            tmp_wsp[x->index] += loss * x->value;
                            x++;
                        }
                        // vector_operator::add_sparse(&wsp[j * B], xsp[j][i], loss, B);
                    }
                }
            }
            inn_iter++;
        }

        for (i = 0; i < l; i++) 
        {
            alpha_inc[i] = alpha[i] - alpha_orig[i];
            sum_alpha_inc += alpha_inc[i];
            alpha_inc_square += alpha_inc[i] * alpha_inc[i] * diag;
            alpha_inc_dot_alpha += alpha_inc[i] * alpha_orig[i] * diag;
            if (alpha_inc[i] > 0)
            {
                max_step = std::min(max_step, INF);
            }
            else if (alpha_inc[i] < 0)
            {
                max_step = std::min(max_step, - alpha_orig[i] / alpha_inc[i]);
            }
        }
        eta_agg.update(max_step);
        int wsp_size = n_kernel * B;
        for (i = 0; i < n_kernel * B; i++) 
        {
            param_list.update(i, wsp[i] - wsp_orig[i] - delta_wsp[i] / W);
        }
        param_list.update(wsp_size, sum_alpha_inc - sum_alpha_inc_org / W);
        param_list.update(wsp_size + 1, alpha_inc_square - alpha_inc_square_org / W);
        param_list.update(wsp_size + 2, alpha_inc_dot_alpha - alpha_inc_dot_alpha_org / W);

        AggregatorFactory::sync();

        max_step = eta_agg.get_value();
        const auto& tmp_wsp = param_list.get_all_param();
        for (i = 0; i < n_kernel * B; i++)
        {
            delta_wsp[i] = tmp_wsp[i];
        }
        sum_alpha_inc = tmp_wsp[wsp_size];
        alpha_inc_square = tmp_wsp[wsp_size + 1];
        alpha_inc_dot_alpha = tmp_wsp[wsp_size + 2];
        husky::LOG_I << "sum_alpha_inc: " + std::to_string(sum_alpha_inc) + ", alpha_inc_square: " + std::to_string(alpha_inc_square) + ", alpha_inc_dot_alpha" + std::to_string(alpha_inc_dot_alpha);

        for (i = 0; i < n_kernel; i++) 
        {
            tmp = tmp2 = 0;
            double* tmp_delta_wsp = &delta_wsp[i * B];
            double* tmp_wsp_orig = &wsp_orig[i * B];
            for (j = 0; j < B; j++)
            {
                tmp += tmp_delta_wsp[j] * tmp_delta_wsp[j];
                tmp2 += tmp_delta_wsp[j] * tmp_wsp_orig[j];
            }
            w_inc_square += mu_set[i] * tmp;
            w_dot_w_inc += mu_set[i] * tmp2;
            // w_inc_square += mu_set[i] * vector_operator::self_dot_product(&delta_wsp[i * B], B);
            // w_dot_w_inc += mu_set[i] * vector_operator::dot_product(&wsp_orig[i * B], &delta_wsp[i * B], B);
        }
        husky::LOG_I << "w_inc_square: " + std::to_string(w_inc_square) + ", w_dot_w_inc: " + std::to_string(w_dot_w_inc);

        grad_alpha_inc = w_dot_w_inc + alpha_inc_dot_alpha - sum_alpha_inc;
        if (grad_alpha_inc >= 0) 
        {
            for (i = 0; i < n_kernel * B; i++) 
            {
                wsp[i] = best_wsp[i];
            }
            break;
        }

        double aQa = alpha_inc_square + w_inc_square;
        eta = std::min(max_step, -grad_alpha_inc /aQa);

        // alpha = alpha_orig + eta * alpha_inc;
        for (i = 0; i < l; i++) 
        {
            alpha[i] = alpha_orig[i] + eta * alpha_inc[i];
        }
        // w = w_orig + eta * delta_w;
        for (i = 0; i < n_kernel * B; i++) 
        {
            wsp[i] = wsp_orig[i] + eta * delta_wsp[i];
        }

        // f(w) + f(a) will cancel out the 0.5\alphaQ\alpha term (old value)
        obj += eta * (0.5 * eta * aQa + grad_alpha_inc);

        reg += eta * (w_dot_w_inc + 0.5 * eta * w_inc_square);

        primal = 0;

        for (i = 0; i < l; i++) 
        {
            double yi = train_set_data[i].y;
            loss = 0;
            for (j = 0; j < n_kernel; j++) 
            {
                tmp = 0;
                double* tmp_wsp = &wsp[j * B];
                feature_node* x = xsp[j][i];
                while (x->index != -1)
                {
                    tmp += tmp_wsp[x->index] * x->value;
                    x++;
                }
                loss += mu_set[j] * tmp;
                // loss += mu_set[j] * vector_operator::dot_product(&wsp[j * B], xsp[j][i], B);
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


        if (primal < old_primal) 
        {
            old_primal = primal;
            for (i = 0; i < n_kernel * B; i++) 
            {
                best_wsp[i] = wsp[i];
            }
        }

        gap = (primal + obj) / init_primal;

        // if (tid == 0) 
        // {
        //     husky::LOG_I << "primal: " + std::to_string(primal);
        //     husky::LOG_I << "dual: " + std::to_string(obj);
        //     husky::LOG_I << "duality_gap: " + std::to_string(gap);
        // }

        if (gap < EPS) 
        {
            for (i = 0; i < n_kernel * B; i++) 
            {
                wsp[i] = best_wsp[i];
            }
            // w = best_w;
            break;
        }
        iter_out++;
    }

    output_solu_->obj = old_primal;
    output_solu_->wsp = wsp;
    output_solu_->alpha = alpha;
    delete [] wsp_orig;
    delete [] delta_wsp;
    delete [] best_wsp;
    delete [] alpha_inc;
    delete [] alpha_orig;

    if (!cache) 
    {
        delete [] QD;
        delete [] index;
    }
}

// mu_set, wsp, alpha and obj inside model actually will not be used
// this function assumes mu_set, wsp and alpha are newed
void fast_bqo_svm(data* data_, model* model_, double* mu_set, double* wsp, double* alpha_, double* obj_, double* QD = NULL, int* index = NULL, bool cache = false)
{
    clock_t start = clock();
    int i, j, k;
    double tmp, tmp2;

    const auto& train_set_data = data_->train_set->get_data();
    int** dt_set = model_->dt_set;
    feature_node*** xsp = data_->xsp;

    const double C = model_->C;
    const int B = model_->B;
    const int n_kernel = model_->n_kernel;
    const int W = data_->W;
    const int tid = data_->tid;
    const int n = data_->n;
    const int l = data_->l;
    const int N = data_->N;
    const int index_low = data_->idx_l;
    const int index_high = data_->idx_h;
    const int max_iter = model_->max_iter;
    const int max_inn_iter = model_->max_inn_iter;

    int iter_out, inn_iter;

    double diag = 0.5 / C;
    double old_primal, primal, obj, grad_alpha_inc;
    double loss, reg = 0;
    double init_primal = C * N;
    double w_inc_square;
    double w_dot_w_inc;

    double sum_alpha_inc;
    double alpha_inc_square;
    double alpha_inc_dot_alpha;
    double sum_alpha_inc_org;
    double alpha_inc_square_org;
    double alpha_inc_dot_alpha_org;

    double max_step;
    double eta;

    double G;
    double gap;
    // projected gradient
    double PG;

    double* alpha = new double[l];
    double* alpha_orig = new double[l];
    double* alpha_inc = new double[l];
    double* wsp_orig = new double[n_kernel * B];
    double* delta_wsp = new double[n_kernel * B];
    double* best_wsp = new double[n_kernel * B];
    // 3 for sum_alpha_inc, alpha_inc_square and alpha_inc_dot_alpha respectively
    husky::lib::ml::ParameterBucket<double> param_list(n_kernel * B + 3);
    Aggregator<double> loss_agg(0.0, [](double& a, const double& b) { a += b; });
    loss_agg.to_reset_each_iter();
    Aggregator<double> eta_agg(INF, [](double& a, const double& b) { a = std::min(a, b); }, [](double& a) { a = INF; });
    eta_agg.to_reset_each_iter();

    std::fill_n(alpha, l, 0);
    std::fill_n(wsp, n_kernel * B, 0);
    if (!cache) 
    {
        QD = new double[l];
        index = new int[l];
        for (i = 0; i < l; i++) {
            QD[i] = 0;
            for (k = 0; k < n_kernel; k++) {
                tmp = 0;
                feature_node* x = xsp[k][i];
                while (x->index != -1)
                {
                    tmp += x->value * x->value;
                    x++;
                }
                QD[i] += mu_set[k] * tmp;
                // QD[i] += mu_set[k] * vector_operator::self_dot_product(xsp[k][i], B);
            }
            QD[i] += diag;
            index[i] = i;
        }
    }

    old_primal = INF;
    obj = 0;

    std::fill_n(delta_wsp, n_kernel * B, 0);
    iter_out = 0;
    while (iter_out < max_iter) {
        // get parameters for local svm solver
        max_step = INF;
        // w_orig = w;
        for (i = 0; i < n_kernel * B; i++) 
        {
            wsp_orig[i] = wsp[i];
        }
        for (i = 0; i < l; i++) 
        {
            alpha_orig[i] = alpha[i];
        }
        // alpha_orig = alpha;
        w_inc_square = 0;
        w_dot_w_inc = 0;
        if (iter_out == 0)
        {
            sum_alpha_inc_org = 0;
            alpha_inc_square_org = 0;
            alpha_inc_dot_alpha_org = 0;
        }
        else
        {
            sum_alpha_inc_org = sum_alpha_inc;
            alpha_inc_square_org = alpha_inc_square;
            alpha_inc_dot_alpha_org = alpha_inc_dot_alpha;
        }
        sum_alpha_inc = 0;
        alpha_inc_square = 0;
        alpha_inc_dot_alpha = 0;

        for (i = 0; i < l; i++) 
        {
            int j = i + std::rand() % (l - i);
            vector_operator::swap(index[i], index[j]);
        }

        // run local svm solver to get local delta alpha
        inn_iter = 0;
        while (inn_iter < max_inn_iter) 
        {
            for (k = 0; k < l; k++) {
                i = index[k];
                double yi = train_set_data[i].y;

                G = 0;
                for (j = 0; j < n_kernel; j++) 
                {
                    tmp = 0;
                    double* tmp_wsp = &wsp[j * B];
                    feature_node* x = xsp[j][i];
                    while (x->index != -1)
                    {
                        tmp += tmp_wsp[x->index] * x->value;
                        x++;
                    }
                    G += mu_set[j] * tmp;
                    // G += mu_set[j] * vector_operator::dot_product(&wsp[j * B], xsp[j][i], B);
                }
                G = G * yi - 1 + diag * alpha[i];

                PG = 0;
                if (alpha[i] == 0) 
                {
                    if (G < 0) 
                    {
                        PG = G;
                    }
                } 
                else if (alpha[i] == INF)
                {
                    if (G > 0) 
                    {
                        PG = G;
                    }
                } 
                else 
                {
                    PG = G;
                }

                if (fabs(PG) > 1e-12) 
                {
                    double alpha_old = alpha[i];
                    alpha[i] = std::min(std::max(alpha[i] - G / QD[i], 0.0), INF);
                    loss = yi * (alpha[i] - alpha_old);
                    for (j = 0; j < n_kernel; j++) 
                    {
                        double* tmp_wsp = &wsp[j * B];
                        feature_node* x = xsp[j][i];
                        while (x->index != -1)
                        {
                            tmp_wsp[x->index] += loss * x->value;
                            x++;
                        }
                        // vector_operator::add_sparse(&wsp[j * B], xsp[j][i], loss, B);
                    }
                }
            }
            inn_iter++;
        }

        for (i = 0; i < l; i++) 
        {
            alpha_inc[i] = alpha[i] - alpha_orig[i];
            sum_alpha_inc += alpha_inc[i];
            alpha_inc_square += alpha_inc[i] * alpha_inc[i] * diag;
            alpha_inc_dot_alpha += alpha_inc[i] * alpha_orig[i] * diag;
            if (alpha_inc[i] > 0)
            {
                max_step = std::min(max_step, INF);
            }
            else if (alpha_inc[i] < 0)
            {
                max_step = std::min(max_step, - alpha_orig[i] / alpha_inc[i]);
            }
        }
        eta_agg.update(max_step);
        int wsp_size = n_kernel * B;
        for (i = 0; i < n_kernel * B; i++) 
        {
            param_list.update(i, wsp[i] - wsp_orig[i] - delta_wsp[i] / W);
        }
        param_list.update(wsp_size, sum_alpha_inc - sum_alpha_inc_org / W);
        param_list.update(wsp_size + 1, alpha_inc_square - alpha_inc_square_org / W);
        param_list.update(wsp_size + 2, alpha_inc_dot_alpha - alpha_inc_dot_alpha_org / W);
        AggregatorFactory::sync();
        max_step = eta_agg.get_value();
        const auto& tmp_wsp = param_list.get_all_param();
        for (i = 0; i < n_kernel * B; i++)
        {
            delta_wsp[i] = tmp_wsp[i];
        }
        sum_alpha_inc = tmp_wsp[wsp_size];
        alpha_inc_square = tmp_wsp[wsp_size + 1];
        alpha_inc_dot_alpha = tmp_wsp[wsp_size + 2];

        for (i = 0; i < n_kernel; i++) 
        {
            tmp = tmp2 = 0;
            double* tmp_delta_wsp = &delta_wsp[i * B];
            double* tmp_wsp_orig = &wsp_orig[i * B];
            for (j = 0; j < B; j++)
            {
                tmp += tmp_delta_wsp[j] * tmp_delta_wsp[j];
                tmp2 += tmp_delta_wsp[j] * tmp_wsp_orig[j];
            }
            w_inc_square += mu_set[i] * tmp;
            w_dot_w_inc += mu_set[i] * tmp2;
            // w_inc_square += mu_set[i] * vector_operator::self_dot_product(&delta_wsp[i * B], B);
            // w_dot_w_inc += mu_set[i] * vector_operator::dot_product(&wsp_orig[i * B], &delta_wsp[i * B], B);
        }

        grad_alpha_inc = w_dot_w_inc + alpha_inc_dot_alpha - sum_alpha_inc;
        if (grad_alpha_inc >= 0) 
        {
            for (i = 0; i < n_kernel * B; i++) 
            {
                wsp[i] = best_wsp[i];
            }
            break;
        }

        double aQa = alpha_inc_square + w_inc_square;
        eta = std::min(max_step, -grad_alpha_inc / aQa);

        // alpha = alpha_orig + eta * alpha_inc;
        for (i = 0; i < l; i++) 
        {
            alpha[i] = alpha_orig[i] + eta * alpha_inc[i];
        }
        // w = w_orig + eta * delta_w;
        for (i = 0; i < n_kernel * B; i++) 
        {
            wsp[i] = wsp_orig[i] + eta * delta_wsp[i];
        }

        // f(w) + f(a) will cancel out the 0.5\alphaQ\alpha term (old value)
        obj += eta * (0.5 * eta * aQa + grad_alpha_inc);

        reg += eta * (w_dot_w_inc + 0.5 * eta * w_inc_square);

        primal = 0;

        for (i = 0; i < l; i++) 
        {
            double yi = train_set_data[i].y;
            loss = 0;
            for (j = 0; j < n_kernel; j++) 
            {
                tmp = 0;
                double* tmp_wsp = &wsp[j * B];
                feature_node* x = xsp[j][i];
                while (x->index != -1)
                {
                    tmp += tmp_wsp[x->index] * x->value;
                    x++;
                }
                loss += mu_set[j] * tmp;
                // loss += mu_set[j] * vector_operator::dot_product(&wsp[j * B], xsp[j][i], B);
            }
            loss *= yi * -1;
            loss += 1;
            // loss = 1 - labeled_point.y * w.dot(labeled_point.x);
            if (loss > 0) 
            {
                // l2 loss
                loss_agg.update(C * loss * loss);
            }
            // this is a minomer, actually this calculate both aTa and eTa
        }
        AggregatorFactory::sync();

        primal += reg + loss_agg.get_value();


        if (primal < old_primal) 
        {
            old_primal = primal;
            for (i = 0; i < n_kernel * B; i++) 
            {
                best_wsp[i] = wsp[i];
            }
            for (i = 0; i < l; i++)
            {
                alpha_[i] = alpha[i];
            }
            *obj_ = primal;
        }

        gap = (primal + obj) / init_primal;

        // if (tid == 0) 
        // {
        //     husky::LOG_I << "primal: " + std::to_string(primal);
        //     husky::LOG_I << "dual: " + std::to_string(obj);
        //     husky::LOG_I << "duality_gap: " + std::to_string(gap);
        // }

        if (gap < EPS) 
        {
            for (i = 0; i < n_kernel * B; i++) 
            {
                wsp[i] = best_wsp[i];
            }
            // w = best_w;
            break;
        }
        iter_out++;
    }

    delete [] alpha;
    delete [] wsp_orig;
    delete [] delta_wsp;
    delete [] best_wsp;
    delete [] alpha_inc;
    delete [] alpha_orig;

    if (!cache) 
    {
        delete [] QD;
        delete [] index;
    }
    clock_t end = clock();
    if (data_->tid == 0)
    {
        husky::LOG_I << "BQO_SVM, time elapsed: " + std::to_string((double)(end - start) / CLOCKS_PER_SEC);
    }
}

void fast_bqo_svm_cache(data* data_, model* model_, double* mu_set, double* wsp, double* alpha_, double* obj_, double* input_alpha = NULL, double* input_wsp = NULL, double* QD = NULL, int* index = NULL, bool cache = false)
{
    clock_t start = clock();
    int i, j, k;
    double tmp, tmp2;

    const auto& train_set_data = data_->train_set->get_data();
    int** dt_set = model_->dt_set;
    feature_node*** xsp = data_->xsp;

    const double C = model_->C;
    const int B = model_->B;
    const int n_kernel = model_->n_kernel;
    const int W = data_->W;
    const int tid = data_->tid;
    const int n = data_->n;
    const int l = data_->l;
    const int N = data_->N;
    const int index_low = data_->idx_l;
    const int index_high = data_->idx_h;
    const int max_iter = model_->max_iter;
    const int max_inn_iter = model_->max_inn_iter;

    int iter_out, inn_iter;

    double diag = 0.5 / C;
    double old_primal, primal, obj, grad_alpha_inc;
    double loss, reg = 0;
    double init_primal = C * N;
    double w_inc_square;
    double w_dot_w_inc;

    double sum_alpha_inc;
    double alpha_inc_square;
    double alpha_inc_dot_alpha;
    double sum_alpha_inc_org;
    double alpha_inc_square_org;
    double alpha_inc_dot_alpha_org;

    double max_step;
    double eta;

    double G;
    double gap;
    // projected gradient
    double PG;

    double* alpha = new double[l];
    double* alpha_orig = new double[l];
    double* alpha_inc = new double[l];
    double* wsp_orig = new double[n_kernel * B];
    double* delta_wsp = new double[n_kernel * B];
    double* best_wsp = new double[n_kernel * B];
    // 3 for sum_alpha_inc, alpha_inc_square and alpha_inc_dot_alpha respectively
    husky::lib::ml::ParameterBucket<double> param_list(n_kernel * B + 3);
    Aggregator<double> loss_agg(0.0, [](double& a, const double& b) { a += b; });
    loss_agg.to_reset_each_iter();
    Aggregator<double> eta_agg(INF, [](double& a, const double& b) { a = std::min(a, b); }, [](double& a) { a = INF; });
    eta_agg.to_reset_each_iter();

    // if input_solu_ == NULL => alpha = 0 => wt_list is 0, no need to initialize
    if (input_alpha == NULL) 
    {
        std::fill_n(alpha, l, 0);
        std::fill_n(wsp, n_kernel * B, 0);
    } 
    else 
    {
        for (i = 0; i < l; i++) 
        {
            alpha[i] = input_alpha[i];
        }
        for (i = 0; i < n_kernel * B; i++) 
        {
            wsp[i] = input_wsp[i];
        }
    }
    if (!cache) 
    {
        QD = new double[l];
        index = new int[l];
        for (i = 0; i < l; i++) {
            QD[i] = 0;
            for (k = 0; k < n_kernel; k++) {
                tmp = 0;
                feature_node* x = xsp[k][i];
                while (x->index != -1)
                {
                    tmp += x->value * x->value;
                    x++;
                }
                QD[i] += mu_set[k] * tmp;
                // QD[i] += mu_set[k] * vector_operator::self_dot_product(xsp[k][i], B);
            }
            QD[i] += diag;
            index[i] = i;
        }
    }

    old_primal = INF;
    obj = 0;

    /*******************************************************************/
    // if input_solu_ is NULL, then we will initialize alpha with 0 => w will be 0 => primal_obj = C * N, obj = 0
    if (input_alpha != NULL) 
    {
        for (i = 0; i < n_kernel; i++) 
        {
            tmp = 0;
            double* tmp_wsp = &wsp[i * B];
            for (j = 0; j < B; j++)
            {
                // tmp += wsp[i * B + j] * wsp[i * B + j];
                tmp += tmp_wsp[j] * tmp_wsp[j];
            }
            reg += mu_set[i] * tmp;
        }

        // reg = self_dot_product(w);
        reg *= 0.5;
        Aggregator<double> et_alpha_agg(0.0, [](double& a, const double& b) { a += b; });
        et_alpha_agg.to_reset_each_iter();
        for (i = 0; i < l; i++) {
            const auto& labeled_point = train_set_data[i];
            loss = 0;
            // loss = 1 - labeled_point.y * w.dot(labeled_point.x);
            for (j = 0; j < n_kernel; j++) 
            {
                tmp = 0;
                double* tmp_wsp = &wsp[j * B];
                feature_node* x = xsp[j][i];
                while (x->index != -1)
                {
                    tmp += tmp_wsp[x->index] * x->value;
                    x++;
                }
                loss += mu_set[j] * tmp;
                // loss += mu_set[j] * vector_operator::dot_product(&wsp[j * B], xsp[j][i], B);
            }
            loss *= labeled_point.y * -1;
            loss += 1;
            if (loss > 0)
            {
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

    std::fill_n(delta_wsp, n_kernel * B, 0);
    iter_out = 0;
    while (iter_out < max_iter) {
        // get parameters for local svm solver
        max_step = INF;
        // w_orig = w;
        for (i = 0; i < n_kernel * B; i++) 
        {
            wsp_orig[i] = wsp[i];
        }
        for (i = 0; i < l; i++) 
        {
            alpha_orig[i] = alpha[i];
        }
        // alpha_orig = alpha;
        w_inc_square = 0;
        w_dot_w_inc = 0;
        if (iter_out == 0)
        {
            sum_alpha_inc_org = 0;
            alpha_inc_square_org = 0;
            alpha_inc_dot_alpha_org = 0;
        }
        else
        {
            sum_alpha_inc_org = sum_alpha_inc;
            alpha_inc_square_org = alpha_inc_square;
            alpha_inc_dot_alpha_org = alpha_inc_dot_alpha;
        }
        sum_alpha_inc = 0;
        alpha_inc_square = 0;
        alpha_inc_dot_alpha = 0;

        for (i = 0; i < l; i++) 
        {
            int j = i + std::rand() % (l - i);
            vector_operator::swap(index[i], index[j]);
        }

        // run local svm solver to get local delta alpha
        inn_iter = 0;
        while (inn_iter < max_inn_iter) 
        {
            for (k = 0; k < l; k++) {
                i = index[k];
                double yi = train_set_data[i].y;

                G = 0;
                for (j = 0; j < n_kernel; j++) 
                {
                    tmp = 0;
                    double* tmp_wsp = &wsp[j * B];
                    feature_node* x = xsp[j][i];
                    while (x->index != -1)
                    {
                        tmp += tmp_wsp[x->index] * x->value;
                        x++;
                    }
                    G += mu_set[j] * tmp;
                    // G += mu_set[j] * vector_operator::dot_product(&wsp[j * B], xsp[j][i], B);
                }
                G = G * yi - 1 + diag * alpha[i];

                PG = 0;
                if (alpha[i] == 0) 
                {
                    if (G < 0) 
                    {
                        PG = G;
                    }
                } 
                else if (alpha[i] == INF)
                {
                    if (G > 0) 
                    {
                        PG = G;
                    }
                } 
                else 
                {
                    PG = G;
                }

                if (fabs(PG) > 1e-12) 
                {
                    double alpha_old = alpha[i];
                    alpha[i] = std::min(std::max(alpha[i] - G / QD[i], 0.0), INF);
                    loss = yi * (alpha[i] - alpha_old);
                    for (j = 0; j < n_kernel; j++) 
                    {
                        double* tmp_wsp = &wsp[j * B];
                        feature_node* x = xsp[j][i];
                        while (x->index != -1)
                        {
                            tmp_wsp[x->index] += loss * x->value;
                            x++;
                        }
                        // vector_operator::add_sparse(&wsp[j * B], xsp[j][i], loss, B);
                    }
                }
            }
            inn_iter++;
        }

        for (i = 0; i < l; i++) 
        {
            alpha_inc[i] = alpha[i] - alpha_orig[i];
            sum_alpha_inc += alpha_inc[i];
            alpha_inc_square += alpha_inc[i] * alpha_inc[i] * diag;
            alpha_inc_dot_alpha += alpha_inc[i] * alpha_orig[i] * diag;
            if (alpha_inc[i] > 0)
            {
                max_step = std::min(max_step, INF);
            }
            else if (alpha_inc[i] < 0)
            {
                max_step = std::min(max_step, - alpha_orig[i] / alpha_inc[i]);
            }
        }
        eta_agg.update(max_step);
        int wsp_size = n_kernel * B;
        for (i = 0; i < n_kernel * B; i++) 
        {
            param_list.update(i, wsp[i] - wsp_orig[i] - delta_wsp[i] / W);
        }
        param_list.update(wsp_size, sum_alpha_inc - sum_alpha_inc_org / W);
        param_list.update(wsp_size + 1, alpha_inc_square - alpha_inc_square_org / W);
        param_list.update(wsp_size + 2, alpha_inc_dot_alpha - alpha_inc_dot_alpha_org / W);
        AggregatorFactory::sync();
        max_step = eta_agg.get_value();
        const auto& tmp_wsp = param_list.get_all_param();
        for (i = 0; i < n_kernel * B; i++)
        {
            delta_wsp[i] = tmp_wsp[i];
        }
        sum_alpha_inc = tmp_wsp[wsp_size];
        alpha_inc_square = tmp_wsp[wsp_size + 1];
        alpha_inc_dot_alpha = tmp_wsp[wsp_size + 2];

        for (i = 0; i < n_kernel; i++) 
        {
            tmp = tmp2 = 0;
            double* tmp_delta_wsp = &delta_wsp[i * B];
            double* tmp_wsp_orig = &wsp_orig[i * B];
            for (j = 0; j < B; j++)
            {
                tmp += tmp_delta_wsp[j] * tmp_delta_wsp[j];
                tmp2 += tmp_delta_wsp[j] * tmp_wsp_orig[j];
            }
            w_inc_square += mu_set[i] * tmp;
            w_dot_w_inc += mu_set[i] * tmp2;
            // w_inc_square += mu_set[i] * vector_operator::self_dot_product(&delta_wsp[i * B], B);
            // w_dot_w_inc += mu_set[i] * vector_operator::dot_product(&wsp_orig[i * B], &delta_wsp[i * B], B);
        }

        grad_alpha_inc = w_dot_w_inc + alpha_inc_dot_alpha - sum_alpha_inc;
        if (grad_alpha_inc >= 0) 
        {
            for (i = 0; i < n_kernel * B; i++) 
            {
                wsp[i] = best_wsp[i];
            }
            break;
        }

        double aQa = alpha_inc_square + w_inc_square;
        eta = std::min(max_step, -grad_alpha_inc / aQa);

        // alpha = alpha_orig + eta * alpha_inc;
        for (i = 0; i < l; i++) 
        {
            alpha[i] = alpha_orig[i] + eta * alpha_inc[i];
        }
        // w = w_orig + eta * delta_w;
        for (i = 0; i < n_kernel * B; i++) 
        {
            wsp[i] = wsp_orig[i] + eta * delta_wsp[i];
        }

        // f(w) + f(a) will cancel out the 0.5\alphaQ\alpha term (old value)
        obj += eta * (0.5 * eta * aQa + grad_alpha_inc);

        reg += eta * (w_dot_w_inc + 0.5 * eta * w_inc_square);

        primal = 0;

        for (i = 0; i < l; i++) 
        {
            double yi = train_set_data[i].y;
            loss = 0;
            for (j = 0; j < n_kernel; j++) 
            {
                tmp = 0;
                double* tmp_wsp = &wsp[j * B];
                feature_node* x = xsp[j][i];
                while (x->index != -1)
                {
                    tmp += tmp_wsp[x->index] * x->value;
                    x++;
                }
                loss += mu_set[j] * tmp;
                // loss += mu_set[j] * vector_operator::dot_product(&wsp[j * B], xsp[j][i], B);
            }
            loss *= yi * -1;
            loss += 1;
            // loss = 1 - labeled_point.y * w.dot(labeled_point.x);
            if (loss > 0) 
            {
                // l2 loss
                loss_agg.update(C * loss * loss);
            }
            // this is a minomer, actually this calculate both aTa and eTa
        }
        AggregatorFactory::sync();

        primal += reg + loss_agg.get_value();


        if (primal < old_primal) 
        {
            old_primal = primal;
            for (i = 0; i < n_kernel * B; i++) 
            {
                best_wsp[i] = wsp[i];
            }
            for (i = 0; i < l; i++)
            {
                alpha_[i] = alpha[i];
            }
            *obj_ = primal;
        }

        gap = (primal + obj) / init_primal;

        // if (tid == 0) 
        // {
        //     husky::LOG_I << "primal: " + std::to_string(primal);
        //     husky::LOG_I << "dual: " + std::to_string(obj);
        //     husky::LOG_I << "duality_gap: " + std::to_string(gap);
        // }

        if (gap < EPS) 
        {
            for (i = 0; i < n_kernel * B; i++) 
            {
                wsp[i] = best_wsp[i];
            }
            // w = best_w;
            break;
        }
        iter_out++;
    }

    delete [] alpha;
    delete [] wsp_orig;
    delete [] delta_wsp;
    delete [] best_wsp;
    delete [] alpha_inc;
    delete [] alpha_orig;

    if (!cache) 
    {
        delete [] QD;
        delete [] index;
    }
    clock_t end = clock();
    if (data_->tid == 0)
    {
        husky::LOG_I << "BQO_SVM, time elapsed: " + std::to_string((double)(end - start) / CLOCKS_PER_SEC);
    }
}

void simpleMKL(data* data_, model* model_, solu* solu_) 
{
    clock_t start = clock();
    int i, j;
    int nloop, loop, maxloop;
    nloop = 1;
    loop = 1;
    maxloop = 4;
    double tmp;

    const int l = data_->l;   
    const int n = data_->n;
    const int n_kernel = model_->n_kernel;
    const int B = model_->B;
    const double gold_ratio = (sqrt(double(5)) + 1) / 2;

    double* mu_set = model_->mu_set;
    int** dt_set = model_->dt_set;

    // initialize mu_set
    std::fill_n(mu_set, 1.0 / n_kernel, n_kernel);

    // cache QD and index
    const auto& train_set_data = data_->train_set->get_data();
    feature_node ***xsp = data_->xsp;
    double diag = 0.5 / model_->C;
    double* QD = new double[l];
    int* index = new int[l];
    for (i = 0; i < l; i++) 
    {
        QD[i] = 0;
        for (j = 0; j < n_kernel; j++) 
        {
            tmp = 0;
            feature_node *x = xsp[j][i];
            while (x->index != -1)
            {
                tmp += x->value * x->value;
                x++;
            }
            QD[i] += mu_set[j] * tmp;
            // QD[i] += mu_set[j] * vector_operator::self_dot_product(xsp[j][i], B);
        }
        QD[i] += diag;
        index[i] = i;
    }

    fast_bqo_svm(data_, model_, mu_set, solu_->wsp, solu_->alpha, &solu_->obj, QD, index, true);
    // fast_bqo_svm_cache(data_, model_, mu_set, solu_->wsp, solu_->alpha, &solu_->obj, NULL, NULL, QD, index, true);

    double obj = solu_->obj;
    double* alpha = solu_->alpha;
    double* wsp = solu_->wsp;
    // compute gradient
    double* grad = new double[n_kernel];
    for (i = 0; i < n_kernel; i++) 
    {
        tmp = 0;
        double *wsp_tmp_ptr = &wsp[i * B];
        for (j = 0; j < B; j++)
        {
            tmp += wsp_tmp_ptr[j] * wsp_tmp_ptr[j];
        }
        grad[i] = -0.5 * tmp;    
    }

    while (loop == 1 && maxloop > 0 && n_kernel > 1) {
        nloop++;

        double old_obj = obj;

        double* new_mu_set = new double[n_kernel];
        for (i = 0; i < n_kernel; i++)
        {
            new_mu_set[i] = mu_set[i];
        }

        // normalize gradient
        double sum_grad = 0;
        for(i = 0; i < n_kernel; i++)
        {
            sum_grad += grad[i] * grad[i];
        }
        double sqrt_grad = sqrt(sum_grad);
        for(i = 0; i < n_kernel; i++)
        {
            grad[i] /= sqrt_grad;
        }

        // compute descent direction
        double max_mu = 0;
        int max_index = 0;

        vector_operator::my_max(mu_set, n_kernel, &max_mu, &max_index);
        double grad_tmp = grad[max_index];
        for (i = 0; i < n_kernel; i++) 
        {
            grad[i] -= grad_tmp;
        }

        double* desc = new double[n_kernel];
        double sum_desc = 0;
        for(i = 0; i < n_kernel; i++)
        {
            if( mu_set[i] > 0 || grad[i] < 0)
            {
                desc[i] = -grad[i];
            }
            else
            {
                desc[i] = 0;
            }
            sum_desc += desc[i];
        }

        desc[max_index] = -sum_desc;

        double step_min = 0;
        double cost_min = old_obj;
        double cost_max = 0;
        double step_max = 0;

        // note here we use new_mu_set
        vector_operator::my_soft_min(new_mu_set, desc, n_kernel, &step_max);

        double delta_max = step_max;

        int flag = 1;
        if (step_max == 0) 
        {
            flag = 0;
        }

        if (flag == 1) 
        {
            if (step_max > 0.1) 
            {
                step_max = 0.1;
                delta_max = step_max;
            }

            // model* tmp_model = new model(model_);
            // solu* tmp_solu = new solu(l, n, B, n_kernel);
            double *tmp_mu_set = new double[n_kernel];
            double *tmp_wsp = new double[n_kernel * B];
            double *tmp_alpha = new double[l];
            double tmp_obj;

            while (cost_max < cost_min) 
            {
                if (data_->tid == 0)
                {
                    husky::LOG_I << "descent direction search";
                }
                for(i = 0; i < n_kernel; i++)
                {
                    tmp_mu_set[i] = new_mu_set[i] + step_max * desc[i];
                }
                // use descent direction to compute new objective
                fast_bqo_svm(data_, model_, tmp_mu_set, tmp_wsp, tmp_alpha, &tmp_obj, QD, index, true);
                // fast_bqo_svm_cache(data_, model_, tmp_mu_set, tmp_wsp, tmp_alpha, &solu_->obj, solu_->alpha, solu_->wsp, QD, index, true);
                cost_max = tmp_obj;
                if (cost_max < cost_min) 
                {
                    cost_min = cost_max;

                    for (i = 0; i < n_kernel; i++) 
                    {
                        new_mu_set[i] = tmp_mu_set[i];
                        mu_set[i] = tmp_mu_set[i];
                    }

                    sum_desc = 0;
                    int fflag = 1;
                    for (i = 0; i < n_kernel; i++) 
                    {
                        if (new_mu_set[i] > 1e-12 || desc[i] > 0) 
                        {
                            ;
                        } 
                        else 
                        {
                            desc[i] = 0;
                        }

                        if (i != max_index) 
                        {
                            sum_desc += desc[i];
                        }
                        // as long as one of them has descent direction negative, we will go on
                        if (desc[i] < 0) 
                        {
                            fflag = 0;
                        }
                    }

                    desc[max_index] = -sum_desc;
                    solu_->obj = tmp_obj;
                    for (i = 0; i < l; i++) 
                    {
                        alpha[i] = tmp_alpha[i];
                    }
                    for (i = 0; i < n_kernel * B; i++)
                    {
                        wsp[i] = tmp_wsp[i];
                    }

                    if (fflag) 
                    {
                        step_max = 0;
                        delta_max = 0;
                    } 
                    else 
                    {
                        // we can still descend, loop again
                        vector_operator::my_soft_min(new_mu_set, desc, n_kernel, &step_max);
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

            if (cost_max < cost_min) 
            {
                min_val = cost_max;
                min_idx = 3;
            } 
            else 
            {
                min_val = cost_min;
                min_idx = 0;
            }

            // model* tmp_ls_model_1 = new model(model_);
            // model* tmp_ls_model_2 = new model(model_);
            // solu* tmp_ls_solu_1 = new solu(l, n, B, n_kernel);
            // solu* tmp_ls_solu_2 = new solu(l, n, B, n_kernel);

            double* tmp_mu_set_1 = new double[n_kernel];
            double* tmp_wsp_1 = new double[n_kernel * B];
            double* tmp_alpha_1 = new double[l];
            double tmp_obj_1;

            double* tmp_mu_set_2 = new double[n_kernel];
            double* tmp_wsp_2 = new double[n_kernel * B];
            double* tmp_alpha_2 = new double[l];
            double tmp_obj_2;

            int step_loop = 0;
            while ((step_max - step_min) > 1e-1 * fabs(delta_max) && step_max > 1e-12) 
            {
                if (data_->tid == 0)
                {
                    husky::LOG_I << "line search";
                }
                step_loop += 1;
                if (step_loop > 8) 
                {
                    break;
                }
                double step_medr = step_min + (step_max - step_min) / gold_ratio;
                double step_medl = step_min + (step_medr - step_min) / gold_ratio;

                // half
                for (i = 0; i < n_kernel; i++) 
                {
                    tmp_mu_set_1[i] = new_mu_set[i] + step_medr * desc[i];
                }
                fast_bqo_svm(data_, model_, tmp_mu_set_1, tmp_wsp_1, tmp_alpha_1, &tmp_obj_1, QD, index, true);
                // fast_bqo_svm_cache(data_, model_, tmp_mu_set_1, tmp_wsp_1, tmp_alpha_1, &tmp_obj_1, tmp_alpha, tmp_wsp, QD, index, true);

                // half half
                for (i = 0; i < n_kernel; i++) 
                {
                    tmp_mu_set_2[i] = new_mu_set[i] + step_medl * desc[i];
                }
                fast_bqo_svm(data_, model_, tmp_mu_set_2, tmp_wsp_2, tmp_alpha_2, &tmp_obj_2, QD, index, true);
                // fast_bqo_svm_cache(data_, model_, tmp_mu_set_2, tmp_wsp_2, tmp_alpha_2, &tmp_obj_2, tmp_alpha, tmp_wsp, QD, index, true);

                step[0] = step_min;
                step[1] = step_medl;
                step[2] = step_medr;
                step[3] = step_max;

                cost[0] = cost_min;
                cost[1] = tmp_obj_2;
                cost[2] = tmp_obj_1;
                cost[3] = cost_max;

                vector_operator::my_min(cost, 4, &min_val, &min_idx);

                switch(min_idx) 
                {
                    case 0:
                        step_max = step_medl;
                        cost_max = cost[1];
                        solu_->obj = tmp_obj_2;
                        for (i = 0; i < n_kernel * B; i++) 
                        {
                            wsp[i] = tmp_wsp_2[i];
                        }
                        for (i = 0; i < l; i++) 
                        {
                            alpha[i] = tmp_alpha_2[i];
                        }
                    break;

                    case 1:
                        step_max = step_medr;
                        cost_max = cost[2];
                        solu_->obj = tmp_obj_1;
                        for (i = 0; i < n_kernel * B; i++) 
                        {
                            wsp[i] = tmp_wsp_1[i];
                        }
                        for (i = 0; i < l; i++) 
                        {
                            alpha[i] = tmp_alpha_1[i];
                        }
                    break;

                    case 2:
                        step_min = step_medl;
                        cost_min = cost[1];
                        solu_->obj = tmp_obj_2;
                        for (i = 0; i < n_kernel * B; i++) 
                        {
                            wsp[i] = tmp_wsp_2[i];
                        }
                        for (i = 0; i < l; i++) 
                        {
                            alpha[i] = tmp_alpha_2[i];
                        }                    
                    break;

                    case 3:
                        step_min = step_medr;
                        cost_min = cost[2];
                        solu_->obj = tmp_obj_1;
                        for (i = 0; i < n_kernel * B; i++) 
                        {
                            wsp[i] = tmp_wsp_1[i];
                        }
                        for (i = 0; i < l; i++) 
                        {
                            alpha[i] = tmp_alpha_1[i];
                        }                 
                    break;
                }// switch(min_idx);      
            } // while ((step_max - step_min) > 1e-1 * fabs(delta_max) && step_max > 1e-12)

            // assignment
            double step_size = step[min_idx];
            if (solu_->obj < old_obj) 
            {
                for (i = 0; i < n_kernel; i++) 
                {
                    new_mu_set[i] += step_size * desc[i];
                    mu_set[i] = new_mu_set[i];
                }
            }
            // delete tmp_ls_model_1;
            // delete tmp_ls_model_2;
            // delete tmp_ls_solu_1;
            // delete tmp_ls_solu_2;
            // delete tmp_solu;
            // delete tmp_model;
            delete [] tmp_alpha_2;
            delete [] tmp_wsp_2;
            delete [] tmp_mu_set_2;

            delete [] tmp_alpha_1;
            delete [] tmp_wsp_1;
            delete [] tmp_mu_set_1;

            delete [] tmp_alpha;
            delete [] tmp_wsp;
            delete [] tmp_mu_set;

            delete cost;
            delete step;
        }// if(flag)

        // test convergence
        double mu_max;
        int mu_max_idx;

        vector_operator::my_max(mu_set, n_kernel, &mu_max, &mu_max_idx);
        // normalize mu_max
        if (mu_max > 1e-12) 
        {
            double mu_sum = 0;
            for (i = 0; i < n_kernel; i++) 
            {
                if (mu_set[i] < 1e-12) 
                {
                    mu_set[i] = 0;
                }
                mu_sum += mu_set[i];
            }
            for (i = 0; i < n_kernel; i++) 
            {
                mu_set[i] /= mu_sum;
            }
        }


        for (i = 0; i < n_kernel; i++) 
        {
            tmp = 0;
            double *wsp_tmp_ptr = &wsp[i * B];
            for (j = 0; j < B; j++)
            {
                tmp += wsp_tmp_ptr[j] * wsp_tmp_ptr[j];
            }
            grad[i] = -0.5 * tmp;
            // grad[i] = -0.5 * vector_operator::self_dot_product(&wsp[i * B], B);
        }
        double min_grad = 0;
        double max_grad = 0;
        int ffflag = 1;
        for (i = 0; i < n_kernel; i++) 
        {
            if (mu_set[i] > 1e-8) 
            {
                if (ffflag) 
                {
                    min_grad = grad[i];
                    max_grad = grad[i];
                    ffflag = 0;
                } 
                else 
                {
                    if (grad[i] < min_grad) 
                    {
                        min_grad = grad[i];
                    }
                    if (grad[i] > max_grad) 
                    {
                        max_grad = grad[i];
                    }
                }
            }
        }

        // double KKTconstraint = fabs(min_grad - max_grad) / fabs(min_grad);
        // note we find min idx in grad, corresponding to max idx in -grad

        double* tmp_grad = new double[n_kernel];
        for (i = 0; i < n_kernel; i++) 
        {
            tmp_grad[i] = -1 * grad[i];
        }
        double max_tmp;
        int max_tmp_idx;
        vector_operator::my_max(tmp_grad, n_kernel, &max_tmp, &max_tmp_idx);

        // Aggregator<double> et_alpha_agg(0.0, [](double& a, const double& b) { a += b; });
        // et_alpha_agg.to_reset_each_iter();
        // for (i = 0; i < n_kernel; i++) 
        // {
        //     // et_alpha_agg.update(alpha[i]);
        //     et_alpha_agg.update(alpha[i] * (1 - alpha[i] * diag));
        // }
        // AggregatorFactory::sync();
        // double tmp_sum = et_alpha_agg.get_value();

        // double dual_gap = (solu_->obj + max_tmp - tmp_sum) / solu_->obj;
        double mkl_grad = 0;
        for (i = 0; i < n_kernel; i++)
        {
            tmp = 0;
            double* tmp_wsp = &wsp[i * B];
            for (j = 0; j < B; j++)
            {
                tmp += tmp_wsp[j] * tmp_wsp[j];
            }
            mkl_grad += mu_set[i] * tmp;
        }
        mkl_grad *= 0.5;
        double dual_gap = (mkl_grad - max_tmp) / mkl_grad;
        if (data_->tid == 0)
        {
            husky::LOG_I << "[outer loop][dual_gap]: " + std::to_string(fabs(dual_gap));
            // husky::LOG_I << "[outer loop][KKTconstraint]: " + std::to_string(KKTconstraint);
        }
        // if (KKTconstraint < 0.05 || fabs(dual_gap) < 0.01) 
        if (fabs(dual_gap) < 0.01)
        {
            loop = 0;
        }
        if (nloop > maxloop) 
        {
            loop = 0;
            break;
        }

        delete [] tmp_grad;
        delete [] desc;
        delete [] new_mu_set;
    }
    delete [] grad;
    clock_t end = clock();
    if (data_->tid == 0)
    {
        husky::LOG_I << "time elapsed " + std::to_string((double)(end - start) / CLOCKS_PER_SEC);
    }
}

double evaluate(data* data_, model* model_, solu* solu_) 
{
    int i, j;
    int n = data_->n;
    int B = model_->B;
    int n_kernel = model_->n_kernel;
    int** dt_set = model_->dt_set;
    double* mu_set = model_->mu_set;
    double* wsp = solu_->wsp;
    double* w = new double[n];
    double tmp;
    // recover w from wsp
    std::fill_n(w, n, 0);
    for (i = 0; i < n_kernel; i++)
    {
        double* wsp_tmp_ptr = &wsp[i * B];
        int* dt_set_tmp = dt_set[i];
        for (j = 0; j < B; j++)
        {
            w[dt_set_tmp[j]] += mu_set[i] * wsp_tmp_ptr[j];
            // w[dt_set[i][j]] += mu_set[i] * wsp[i * B + j];
        }
    }
    const auto& test_set = data_->test_set;

    double indicator;
    Aggregator<int> error_agg(0, [](int& a, const int& b) { a += b; });
    Aggregator<int> num_test_agg(0, [](int& a, const int& b) { a += b; });
    auto& ac = AggregatorFactory::get_channel();
    list_execute(*test_set, {}, {&ac}, [&](ObjT& labeled_point) 
    {
        tmp = 0;
        auto it = labeled_point.x.begin();
        while (it != labeled_point.x.end())
        {
            tmp += w[(*it).fea] * (*it).val;
            it++;
        }
        indicator = labeled_point.y * tmp;
        // double indicator = labeled_point.y * vector_operator::dot_product(w, labeled_point.x, data_->n);
        if (indicator <= 0) 
        {
            error_agg.update(1);
        }
        num_test_agg.update(1);
    });

    if (data_->tid == 0) 
    {
        husky::LOG_I << "Classification accuracy on testing set with [B = " + 
                        std::to_string(model_->B) + "][C = " + 
                        std::to_string(model_->C) + "], " +
                        "[max_out_iter = " + std::to_string(model_->max_out_iter) + "], " +
                        "[max_iter = " + std::to_string(model_->max_iter) + "], " +
                        "[max_inn_iter = " + std::to_string(model_->max_inn_iter) + "], " +
                        "[test set size = " + std::to_string(num_test_agg.get_value()) + "]: " +
                        std::to_string(1.0 - static_cast<double>(error_agg.get_value()) / num_test_agg.get_value());  
    }
    delete [] w;
    return 1.0 - static_cast<double>(error_agg.get_value()) / num_test_agg.get_value();
}

void run_bqo_svm() 
{
    data* data_ = new data;
    model* model_ = new model;
    solu* solu_ = new solu;
    initialize(data_, model_);
    auto start = std::chrono::steady_clock::now();
    // int dt1[] = {7, 22, 28, 31, 74};
    // int dt2[] = {7, 8, 45, 76, 78};
    // int dt3[] = {12, 59, 60, 83, 90};
    // cache_xsp(data_, model_, dt1, 5);
    // cache_xsp(data_, model_, dt2, 5);
    // cache_xsp(data_, model_, dt3, 5);
    int *dt = most_violated(data_, model_);
    vector_operator::show(dt, model_->B, "dt");
    cache_xsp(data_, model_, dt, model_->B); // cache xsp, add dt to model and increment number of kernels
    husky::LOG_I << "cache completed! number of kernel: " + std::to_string(model_->n_kernel);
    bqo_svm(data_, model_, solu_);
    vector_operator::show(model_->mu_set, model_->n_kernel, "mu_set");
    husky::LOG_I << "trainning objective: " + std::to_string(solu_->obj);
    evaluate(data_, model_, solu_);
    destroy(data_, model_, solu_);
    auto end = std::chrono::steady_clock::now();
    husky::LOG_I << "Time elapsed: "
                    << std::chrono::duration_cast<std::chrono::duration<float>>(end - start).count();
}

void job_runner() 
{
    data* data_ = new data;
    model* model_ = new model;
    solu* solu_ = new solu;

    initialize(data_, model_);

    // double *kernel_acc = new double[MAX_NUM_KERNEL];
    // double *kernel_time = new double[MAX_NUM_KERNEL];

    double *kernel_acc = new double[MAX_NUM_KERNEL];
    double *kernel_violated_time = new double[MAX_NUM_KERNEL];
    double *kernel_cache_time = new double[MAX_NUM_KERNEL];
    double *kernel_train_time = new double[MAX_NUM_KERNEL];

    int iter = 0;
    int max_out_iter = model_->max_out_iter;
    double mkl_obj = INF;
    double last_obj = INF;
    double obj_diff = 0.0;
    int *dt;
    clock_t start, end;
    while(iter < max_out_iter)
    {
        start = clock();
        last_obj = mkl_obj;
        if (iter == 0) 
        {
            dt = most_violated(data_, model_);
        } 
        else 
        {
            dt = most_violated(data_, model_, solu_);
        }
        end = clock();
        kernel_violated_time[iter] = (double)(end - start) / CLOCKS_PER_SEC;

        // if (vector_operator::element_at(dt, model_->dt_set, model_->n_kernel, model_->B))
        // {
        //     husky::LOG_I << "element_at: FGM converged";
        //     delete [] dt;
        //     break;
        // }

        if (data_->tid == 0)
        {
            vector_operator::show(dt, model_->B, "dt");
        }

        start = clock();
        cache_xsp(data_, model_, dt, model_->B);
        end = clock();
        kernel_cache_time[iter] = (double)(end - start) / CLOCKS_PER_SEC;

        if (data_->tid == 0)
        {
            husky::LOG_I << "cache completed! number of kernel: " + std::to_string(model_->n_kernel);
        }
        if (solu_->wsp == NULL && solu_->alpha == NULL)
        {
            solu_->wsp = new double[model_->n_kernel * model_->B];
            solu_->alpha = new double[data_->l];
        }
        else 
        {
            delete [] solu_->wsp;
            delete [] solu_->alpha;
            solu_->wsp = new double[model_->n_kernel * model_->B];
            solu_->alpha = new double[data_->l];            
        }

        start = clock();
        simpleMKL(data_, model_, solu_);
        end = clock();
        kernel_train_time[iter] = (double)(end - start) / CLOCKS_PER_SEC;
        kernel_acc[iter] = evaluate(data_, model_, solu_);

        mkl_obj = solu_->obj;
        obj_diff = fabs(mkl_obj - last_obj);
        if (data_->tid == 0) 
        {
            husky::LOG_I << "[iteration " + std::to_string(iter + 1) + "][mkl_obj " + std::to_string(mkl_obj) + "][obj_diff " + std::to_string(obj_diff) + "]";
            vector_operator::show(model_->mu_set, model_->n_kernel, "mu_set");
        }
        // if (mkl_obj > last_obj)
        // {
        //     if (data_->tid == 0) 
        //     {
        //         husky::LOG_I << "mkl_obj > last_obj, FGM converged";
        //     }
        //     break;
        // }
        // if (obj_diff < 0.001 * abs(last_obj)) 
        // {
        //     if (data_->tid == 0) 
        //     {
        //         husky::LOG_I << "obj_diff < 0.001 * abs(last_obj), FGM converged";
        //     }
        //     break;
        // }
        // if (model_->mu_set[iter] < 0.0001) 
        // {
        //     if (data_->tid == 0) 
        //     { 
        //         husky::LOG_I << "mu_set_[iter] < 0.0001, FGM converged";
        //     }
        //     break;
        // }
        iter++;
    }

    if (iter == max_out_iter)
    {
        iter = max_out_iter - 1;
    }
    if (data_->tid == 0) 
    {
        vector_operator::show(model_->mu_set, model_->n_kernel, "mu_set");
    }
    if (data_->tid == 0)
    {
        int** dt_set = model_->dt_set;
        FILE* dout = fopen("dfgm_url_dt", "w");
        for (int i = 0; i <= iter; i++)
        {
            for (int j = 0; j < model_->B; j++)
            {
                fprintf(dout, "%d ", dt_set[i][j]);
            }
            fprintf(dout, "\n");
        }

        // FILE* fout = fopen("fgm_syn_large_plot.csv", "w");
        FILE* fout = fopen("dfgm_url_plot.csv", "w");
        fprintf(fout, "n_kernel.Vs.accuracy ");
        for (int i = 0; i <= iter; i++)
        {
            fprintf(fout, "%f ", kernel_acc[i]);
        }
        fprintf(fout, "\n");
        fprintf(fout, "n_kernel.Vs.violated_time ");
        for (int i = 0; i <= iter; i++)
        {
            fprintf(fout, "%f ", kernel_violated_time[i]);
        }
        fprintf(fout, "\n");
        fprintf(fout, "n_kernel.Vs.cache_time ");
        for (int i = 0; i <= iter; i++)
        {
            fprintf(fout, "%f ", kernel_cache_time[i]);
        }
        fprintf(fout, "\n");
        fprintf(fout, "n_kernel.Vs.train_time ");
        for (int i = 0; i <= iter; i++)
        {
            fprintf(fout, "%f ", kernel_train_time[i]);
        }
        fprintf(fout, "\n");
        fprintf(fout, "mu_set ");
        for (int i = 0; i <= iter; i++)
        {
            fprintf(fout, "%f ", model_->mu_set[i]);
        }
        fprintf(fout, "\n");

        delete [] kernel_acc;
        delete [] kernel_violated_time;
        delete [] kernel_cache_time;
        delete [] kernel_train_time;
    }
    evaluate(data_, model_, solu_);
    destroy(data_, model_, solu_);
}

void init() 
{
    if (husky::Context::get_param("is_sparse") == "true") 
    {
        job_runner();
    } 
    else 
    {
        husky::LOG_I << "Dense data format is not supported";
    }
}

int main(int argc, char** argv) 
{
    std::vector<std::string> args({"hdfs_namenode", "hdfs_namenode_port", "train", "test", "B", "C", "format", "is_sparse",
                                   "max_out_iter", "max_iter", "max_inn_iter"});
    if (husky::init_with_args(argc, argv, args)) 
    {
        husky::run_job(init);
        return 0;
    }
    return 1;
}
