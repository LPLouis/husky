/***
    Performance discussion:
    1. Underlying implementation of the DenseVector class and SparseVector class.
    2. Representation of the problem , solution w and alpha
        => Right now I am using DenseVector for both w and alpha, which is not going to work well if the problem is really of high dimensional
        => Alternatively I can also use SparseVector for w, but this creates a problem when we try to implement it in a distributed manner because the ParameterServer in Husky accepts a DenseVector, though this problem can be resolved by mapping SparseIndex to DenseIndex. But this seems too much of a burden on the programmer's side
        => In the original FGM implementation, they actually cache the kernel X_tX_t^T for every dt! This idea seems implausible and crazy to me at first and I am supprised that they actually take this approach for faster speed
    3. Analysis of the parameter B
        => When I try the FGM algorithm (the one i implemented) on the a9 dataset available on LIBSVM website, I foudn that if we set B to be 5, 10, 20, or anything less than 100, then the FGM algorithm will actually converge after the first iteration, i.e., after solving the problem with one violated constraint, it still generates the same constraint in the next iteration. I think this may be a good sign that this algorithm is not suitable for this dataset in the sense that although some features may have high violated coefficient (maybe because its features are categorical?), they may not have enough discriminative power. Interestingly, I try using about 20 features (124 in total), which gives accuracies of only 12%. If you do the math, if all features are equally important, then having 20 features should give you 16% accuracy. So is this a good sign that most violated constraint is not a good criterion for selecting features, though intuitively it looks good (coeff of w)? Or is it all just because the features of this dataset are cetegorical?
    4. Design decison 
        => What should SparseVector + SparseVector equal to ? DenseVector or SparseVector, is differentiating these only during run time a good idea?
    5. Implementation
        => Actually self_dot_elem_wise_dot can be switched back to the original version using combined control variable
        => in the actual FGM implementation by Tan Mingkui, they actually do not use the convergence criterion d_k+1 \in C_k. Instead, they use the difference of objective (whether it is smaller than epsion) and the values of the newly add mu_set (whether it is less than 1e-3).
        => Also i realized that we actually do not need to split the data featurewise since we need to compute global w anyway
***/


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

#include "customize_data_loader.hpp"

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
#define EPS 1.0e-3
#define MAX_NUM_KERNEL 10

class data 
{
public:
    int n;          // number of features (appended 1 included)
    int l;         
    husky::ObjList<ObjT>* train_set;
    husky::ObjList<ObjT>* test_set;
    double ***xsp;

    data()
    {
        xsp = new double**[MAX_NUM_KERNEL];
    }

    ~data()
    {
        delete [] xsp;
    }
};

class model 
{
public:
    int B;
    double C;
    int n_kernel;
    int max_out_iter;
    int max_inn_iter;
    double *mu_set;
    int** dt_set;

    model()
    {
        n_kernel = 0;
        mu_set = new double[MAX_NUM_KERNEL];
        dt_set = new int*[MAX_NUM_KERNEL];
    }

    model(const model* m)
    {
        int i, j;
        B = m->B;
        C = m->C;
        max_out_iter = m->max_out_iter;
        max_inn_iter = m->max_inn_iter;
        n_kernel = m->n_kernel;
        mu_set = new double[MAX_NUM_KERNEL];
        for (i = 0; i < n_kernel; i++)
        {
            mu_set[i] = m->mu_set[i];
        }
        dt_set = new int*[MAX_NUM_KERNEL];
        for (i = 0; i < n_kernel; i++)
        {
            dt_set[i] = new int[B];
            for (j = 0; j < B; j++)
            {
                dt_set[i][j] = m->dt_set[i][j];
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
    int l, n, B, n_kernel;
    double obj;
    double *alpha;
    double *wsp;

    solu() 
    {
        alpha = NULL;
        wsp = NULL;
    }

    solu(int l, int n, int B, int n_kernel)
    {
        obj = 0.0;
        this->l = l;
        this->n = n;
        this->B = B;
        this->n_kernel = n_kernel;
        alpha = new double[l];
        wsp = new double[n_kernel * B];
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

class solu_xsp 
{
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
        w = DenseVector<double>(n, 0.0);
        wt_list = std::vector<DenseVector<double>>(T);
    }
};

class vector_operator 
{
public:

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

    // this function is used in cache_xsp
    static void void_elem_wise_dot(double* dst, const SparseVector<double>& v, const int* dt, const int B)
    {
        int i = 0;
        auto it = v.begin();
        while (it != v.end() && i != B)
        {
            int fea = (*it).fea;
            int idx = dt[i];
            if (fea < idx)
            {
                it++;
            }
            else if (fea > idx)
            {
                dst[i] = 0;
                i++;
            }
            else
            {
                // fea == idx
                dst[i] = (*it).val;
                i++;
                it++;
            }
        }
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
    static T self_dot_product(const T* v, const int n)
    {
        T ret = 0;
        for (int i = 0; i < n; i++)
        {
            ret += v[i] * v[i];
        }
        return ret;
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

    static void add_sparse(double* w, const SparseVector<double>& v, const double coef, const int n)
    {
        assert(v.get_feature_num() == n && "add_sparse_vector: error\n");
        for (auto it = v.begin(); it != v.end(); it++)
        {
            w[(*it).fea] += (*it).val * coef;
        }
    }

    static void add_sparse(double* w, const double* xsp, const double coef, const int n)
    {
        for (int i = 0; i < n; i++)
        {
            w[i] += xsp[i] * coef;
        }
    }

    // this is for pure dot_product
    static double dot_product(const double* v1, const double* v2, const int n)
    {
        double ret = 0;
        for (int i = 0; i < n; i++)
        {
            ret += v1[i] * v2[i];
        }
        return ret;
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
    auto& train_set = husky::ObjListStore::create_objlist<ObjT>("train_set");
    auto& test_set = husky::ObjListStore::create_objlist<ObjT>("test_set");
    data_->train_set = &train_set;
    data_->test_set = &test_set;

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
    model_->B = std::stoi(husky::Context::get_param("B"));
    model_->C = std::stod(husky::Context::get_param("C"));
    model_->max_out_iter = std::stoi(husky::Context::get_param("max_out_iter"));
    model_->max_inn_iter = std::stoi(husky::Context::get_param("max_inn_iter"));

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
    data_->n = n;
    data_->l = l;

    husky::LOG_I << "number of samples: " + std::to_string(l);
    husky::LOG_I << "number of features: " + std::to_string(n);
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
    double* alpha = new double[data_->l];

    if (solu_ == NULL) 
    {
        // set alpha to 1 because we do not know how much data others have
        std::fill_n(alpha, data_->l, 1.0);
    } 
    else 
    {
        for (i = 0; i < data_->l; i++)
        {
            alpha[i] = solu_->alpha[i];
        }
    }

    std::vector<std::pair<int, double>> fea_score;

    double* w = new double[data_->n];
    std::fill_n(w, data_->n, 0);
    for (i = 0; i < data_->l; i++)
    {
        vector_operator::add_sparse(w, train_set_data[i].x, train_set_data[i].y * alpha[i], data_->n);
    }

    for (i = 0; i < data_->n; i++) 
    {
        fea_score.push_back(std::make_pair(i, fabs(w[i])));
    }
    std::sort(fea_score.begin(), fea_score.end(), [](auto& left, auto& right) {
        return left.second > right.second;
    });

    int *dt = new int[model_->B];
    for (i = 0; i < model_->B; i++) {
        int fea = fea_score[i].first;
        dt[i] = fea;
    }
    std::sort(dt, dt + model_->B);
    return dt;
}

// this function caches the new kernel corresponding to the given dt
// this function modifies n_kernel, mu_set and dt_set inside model
void cache_xsp(data* data_, model* model_, int *dt, int B) 
{
    int i;
    int l = data_->l;
    int n_kernel = model_->n_kernel;
    auto& train_set_data = data_->train_set->get_data();

    // cache new kernel
    data_->xsp[n_kernel] = new double*[l];
    for (i = 0; i < l; i++)
    {
        double* tmp = new double[B];
        vector_operator::void_elem_wise_dot(tmp, train_set_data[i].x, dt, B);
        data_->xsp[n_kernel][i] = tmp;
    }

    // modify model_
    model_->add_dt(dt, B);
}

void dcd_svm(data* data_, model* model_, solu* output_solu_, solu* input_solu_ = NULL, double* QD = NULL, int* index = NULL, bool cache = false) 
{
    // Declaration and Initialization
    int l = data_->l;
    int n = data_->n;
    const auto& train_set_data = data_->train_set->get_data();
    double* mu_set = model_->mu_set;
    int** dt_set = model_->dt_set;
    double*** xsp = data_->xsp;

    const double C = model_->C;
    const int B = model_->B;
    const int n_kernel = model_->n_kernel;

    double diag = 0.5 / model_->C;
    double UB = INF;

    double G, PG, PGmax_new, PGmin_new;
    double PGmax_old = INF;
    double PGmin_old = -INF;

    int i, j, k, s;
    int active_size = l;

    double* alpha = new double[l];
    double* wsp = new double[n_kernel * B];

    double diff;

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
        for (i = 0; i < l; i++) 
        {
            QD[i] = 0;
            for (k = 0; k < n_kernel; k++) 
            {
                QD[i] += mu_set[k] * vector_operator::self_dot_product(xsp[k][i], B);
            }
            QD[i] += diag;
            index[i] = i;
        }
    }

    int iter = 0;
    while (iter < model_->max_inn_iter) 
    {
        int clock_start_time = clock();

        PGmax_new = 0;
        PGmin_new = 0;

        for (i = 0; i < active_size; i++)
        {
            int j = i + rand() % (active_size - i);
            vector_operator::swap(index[i], index[j]);
        }

        for (s = 0; s < active_size; s++)
        {
            i = index[s];
            G = 0;
            double yi = train_set_data[i].y;

            G = 0;
            for (j = 0; j < n_kernel; j++) 
            {
                if (mu_set[j] != 0)
                {
                    G += mu_set[j] * vector_operator::dot_product(&wsp[j * B], xsp[j][i], B);
                }
            }
            G = G * yi - 1 + diag * alpha[i];

            PG = 0;
            if (alpha[i] ==0)
            {
                if (G > PGmax_old)
                {
                    active_size--;
                    vector_operator::swap(index[s], index[active_size]);
                    s--;
                    continue;
                }
                else if (G < 0)
                {
                    PG = G;
                    PGmin_new = std::min(PGmin_new, PG);
                }
            }
            else if (alpha[i] == INF)
            {
                if (G < PGmin_old)
                {
                    active_size--;
                    vector_operator::swap(index[s], index[active_size]);
                    s--;
                    continue;
                }
                else if (G > 0)
                {
                    PG = G;
                    PGmax_new = std::max(PGmax_new, PG);
                }
            }
            else
            {
                PG = G;
                PGmax_new = std::max(PGmax_new, PG);
                PGmin_new = std::min(PGmin_new, PG);

            }

            if(fabs(PG) > 1.0e-12)
            {
                double alpha_old = alpha[i];
                alpha[i] = std::min(std::max(alpha[i] - G / QD[i], 0.0), INF);
                diff = yi * (alpha[i] - alpha_old);
                for (j = 0; j < n_kernel; j++) 
                {
                    vector_operator::add_sparse(&wsp[j * B], xsp[j][i], diff, B);
                }
            }
        }

        iter++;

        if(PGmax_new - PGmin_new <= EPS)
        {
            if(active_size == l)
                break;
            else
            {
                active_size = l;
                PGmax_old = INF;
                PGmin_old = -INF;
                continue;
            }
        }
    
        PGmax_old = PGmax_new;
        PGmin_old = PGmin_new;
        if (PGmax_old == 0)
        {
            PGmax_old = INF;
        }
        if (PGmin_old == 0)
        {
            PGmin_old = -INF;
        }
    }

    double obj = 0;
    for (i = 0; i < l; i++) 
    {
        obj += alpha[i] * (1 - diag * alpha[i]);
    }
    for (i = 0; i < n_kernel; i++)
    {
        obj -= 0.5 * mu_set[i] * vector_operator::self_dot_product(&wsp[i * B], B);
    }

    if (!cache) 
    {
        delete[] index;
        delete[] QD;
    }
    output_solu_->obj = obj;
    output_solu_->n_kernel = n_kernel;
    output_solu_->wsp = wsp;
    output_solu_->alpha = alpha;
}

void simpleMKL(data* data_, model* model_, solu* solu_) 
{
    int i, j;
    int nloop, loop, maxloop;
    nloop = 1;
    loop = 1;
    maxloop = 12;

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
    double ***xsp = data_->xsp;
    double diag = 0.5 / model_->C;
    double* QD = new double[l];
    int* index = new int[l];
    for (i = 0; i < l; i++) 
    {
        QD[i] = 0;
        for (j = 0; j < n_kernel; j++) 
        {
            QD[i] += mu_set[j] * vector_operator::self_dot_product(xsp[j][i], B);
        }
        QD[i] += diag;
        index[i] = i;
    }

    dcd_svm(data_, model_, solu_, NULL, QD, index, true);
    double obj = solu_->obj;
    double* alpha = solu_->alpha;
    double* wsp = solu_->wsp;
    // compute gradient
    double* grad = new double[n_kernel];
    for (i = 0; i < n_kernel; i++) 
    {
        grad[i] = -0.5 * vector_operator::self_dot_product(&wsp[i * B], B);
    }

    while (loop == 1 && maxloop > 0 && n_kernel > 1) {
        nloop++;

        double old_obj = obj;
        husky::LOG_I << "[outer loop: " + std::to_string(nloop) + "][old_obj]: " + std::to_string(old_obj);

        model* new_model = new model(model_);

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
        vector_operator::my_soft_min(new_model->mu_set, desc, n_kernel, &step_max);

        double delta_max = step_max;

        int flag = 1;
        if (step_max == 0) {
            flag = 0;
        }

        if (flag == 1) 
        {
            if (step_max > 0.1) 
            {
                step_max = 0.1;
                delta_max = step_max;
            }

            model* tmp_model = new model(model_);
            solu* tmp_solu = new solu(l, n, B, n_kernel);

            while (cost_max < cost_min) 
            {
                husky::LOG_I << "[inner loop][cost_max]: " + std::to_string(cost_max);

                for(i = 0; i < n_kernel; i++)
                {
                    tmp_model->mu_set[i] = new_model->mu_set[i] + step_max * desc[i];
                }
                // use descent direction to compute new objective
                dcd_svm(data_, tmp_model, tmp_solu, solu_, QD, index, true);
                cost_max = tmp_solu->obj;
                if (cost_max < cost_min) 
                {
                    cost_min = cost_max;

                    for (i = 0; i < n_kernel; i++) 
                    {
                        new_model->mu_set[i] = tmp_model->mu_set[i];
                        mu_set[i] = new_model->mu_set[i];
                    }

                    sum_desc = 0;
                    int fflag = 1;
                    for (i = 0; i < n_kernel; i++) 
                    {
                        if (new_model->mu_set[i] > 1e-12 || desc[i] > 0) 
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
                    for (i = 0; i < l; i++) 
                    {
                        alpha[i] = tmp_solu->alpha[i];
                    }
                    for (i = 0; i < n_kernel * B; i++)
                    {
                        wsp[i] = tmp_solu->wsp[i];
                    }

                    if (fflag) 
                    {
                        step_max = 0;
                        delta_max = 0;
                    } 
                    else 
                    {
                        // we can still descend, loop again
                        vector_operator::my_soft_min(new_model->mu_set, desc, n_kernel, &step_max);
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

            model* tmp_ls_model_1 = new model(model_);
            model* tmp_ls_model_2 = new model(model_);
            solu* tmp_ls_solu_1 = new solu(l, n, B, n_kernel);
            solu* tmp_ls_solu_2 = new solu(l, n, B, n_kernel);

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
                for (i = 0; i < n_kernel; i++) 
                {
                    tmp_ls_model_1->mu_set[i] = new_model->mu_set[i] + step_medr * desc[i];
                }
                dcd_svm(data_, tmp_ls_model_1, tmp_ls_solu_1, solu_, QD, index, true);

                // half half
                for (i = 0; i < n_kernel; i++) 
                {
                    tmp_ls_model_2->mu_set[i] = new_model->mu_set[i] + step_medr * desc[i];
                }
                dcd_svm(data_, tmp_ls_model_2, tmp_ls_solu_2, solu_, QD, index, true);

                step[0] = step_min;
                step[1] = step_medl;
                step[2] = step_medr;
                step[3] = step_max;

                cost[0] = cost_min;
                cost[1] = tmp_ls_solu_2->obj;
                cost[2] = tmp_ls_solu_1->obj;
                cost[3] = cost_max;

                vector_operator::my_min(cost, 4, &min_val, &min_idx);

                switch(min_idx) 
                {
                    case 0:
                        step_max = step_medl;
                        cost_max = cost[1];
                        solu_->obj = tmp_ls_solu_2->obj;
                        for (i = 0; i < n_kernel * B; i++) 
                        {
                            wsp[i] = tmp_ls_solu_2->wsp[i];
                        }
                        for (i = 0; i < l; i++) 
                        {
                            alpha[i] = tmp_ls_solu_2->alpha[i];
                        }
                    break;

                    case 1:
                        step_max = step_medr;
                        cost_max = cost[2];
                        solu_->obj = tmp_ls_solu_1->obj; 
                        for (i = 0; i < n_kernel * B; i++) 
                        {
                            wsp[i] = tmp_ls_solu_1->wsp[i];
                        }
                        for (i = 0; i < l; i++) 
                        {
                            alpha[i] = tmp_ls_solu_1->alpha[i];
                        }
                    break;

                    case 2:
                        step_min = step_medl;
                        cost_min = cost[1];
                        solu_->obj = tmp_ls_solu_2->obj;
                        for (i = 0; i < n_kernel * B; i++) 
                        {
                            wsp[i] = tmp_ls_solu_2->wsp[i];
                        }
                        for (i = 0; i < l; i++) 
                        {
                            alpha[i] = tmp_ls_solu_2->alpha[i];
                        }                  
                    break;

                    case 3:
                        step_min = step_medr;
                        cost_min = cost[2];
                        solu_->obj = tmp_ls_solu_1->obj;
                        for (i = 0; i < n_kernel * B; i++) 
                        {
                            wsp[i] = tmp_ls_solu_1->wsp[i];
                        }
                        for (i = 0; i < l; i++) 
                        {
                            alpha[i] = tmp_ls_solu_1->alpha[i];
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
                    new_model->mu_set[i] += step_size * desc[i];
                    mu_set[i] = new_model->mu_set[i];
                }
            }
            delete tmp_ls_model_1;
            delete tmp_ls_model_2;
            delete tmp_ls_solu_1;
            delete tmp_ls_solu_2;
            delete tmp_solu;
            delete tmp_model;
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
            grad[i] = -0.5 * vector_operator::self_dot_product(&wsp[i * B], B);
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

        husky::LOG_I << "min_grad: " + std::to_string(min_grad) + ", max_grad: " + std::to_string(max_grad);
        double KKTconstraint = fabs(min_grad - max_grad) / fabs(min_grad);
        // note we find min idx in grad, corresponding to max idx in -grad

        double* tmp_grad = new double[n_kernel];
        for (i = 0; i < n_kernel; i++) 
        {
            tmp_grad[i] = -1 * grad[i];
        }
        double max_tmp;
        int max_tmp_idx;
        vector_operator::my_max(tmp_grad, n_kernel, &max_tmp, &max_tmp_idx);

        Aggregator<double> et_alpha_agg(0.0, [](double& a, const double& b) { a += b; });
        et_alpha_agg.to_reset_each_iter();
        for (i = 0; i < n_kernel; i++) 
        {
            et_alpha_agg.update(alpha[i]);
        }
        AggregatorFactory::sync();
        double tmp_sum = et_alpha_agg.get_value();

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

        delete [] tmp_grad;
        delete [] desc;
        delete new_model;
    }

    delete [] grad;
}

void evaluate(data* data_, model* model_, solu* solu_) 
{
    assert(model_->n_kernel == solu_->n_kernel && "evaluate: error\n");
    int i, j;
    int n = data_->n;
    int B = model_->B;
    int n_kernel = model_->n_kernel;
    int** dt_set = model_->dt_set;
    double* mu_set = model_->mu_set;
    double* wsp = solu_->wsp;
    double* w = new double[n];
    // recover w from wsp
    std::fill_n(w, n, 0);
    for (i = 0; i < n_kernel; i++)
    {
        for (j = 0; j < B; j++)
        {
            w[dt_set[i][j]] += mu_set[i] * wsp[i * B + j];
        }
    }
    const auto& test_set_data = data_->test_set->get_data();

    double indicator;
    int error_agg = 0;

    for (auto labeled_point : test_set_data)
    {
        indicator = labeled_point.y * vector_operator::dot_product(w, labeled_point.x, data_->n);
        if (indicator <= 0) 
        {
            error_agg += 1;
        }
    }

    husky::LOG_I << "Classification accuracy on testing set with [B = " + 
                    std::to_string(model_->B) + "][C = " + 
                    std::to_string(model_->C) + "], " +
                    "[max_out_iter = " + std::to_string(model_->max_out_iter) + "], " +
                    "[max_inn_iter = " + std::to_string(model_->max_inn_iter) + "], " +
                    "[test set size = " + std::to_string(test_set_data.size()) + "]: " +
                    std::to_string(1.0 - static_cast<double>(error_agg) / test_set_data.size());  

    delete [] w;
}

void run_dcd_svm() 
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
    cache_xsp(data_, model_, dt, model_->B); // cache xsp, add dt to model and increment number of kernels
    husky::LOG_I << "cache completed! number of kernel: " + std::to_string(model_->n_kernel);
    dcd_svm(data_, model_, solu_);
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

    int iter = 0;
    int max_out_iter = model_->max_out_iter;
    double mkl_obj = INF;
    double last_obj = INF;
    double obj_diff = 0.0;
    int *dt;
    while(iter < max_out_iter) 
    {
        last_obj = mkl_obj;
        if (iter == 0) 
        {
            dt = most_violated(data_, model_);
        } 
        else 
        {
            dt = most_violated(data_, model_, solu_);
        }
        cache_xsp(data_, model_, dt, model_->B);
        husky::LOG_I << "cache completed! number of kernel: " + std::to_string(model_->n_kernel);
        simpleMKL(data_, model_, solu_);
        mkl_obj = solu_->obj;
        obj_diff = fabs(mkl_obj - last_obj);
        husky::LOG_I << "[iteration " + std::to_string(iter) + "][mkl_obj " + std::to_string(mkl_obj) + "][obj_diff " + std::to_string(obj_diff) + "]";
        vector_operator::show(model_->mu_set, model_->n_kernel, "mu_set");
        if (obj_diff < 0.001 * abs(last_obj)) 
        {
            husky::LOG_I << "FGM converged";
            break;
        }
        if (model_->mu_set[iter] < 0.0001) 
        {
            husky::LOG_I << "FGM converged";
            break;
        }
        iter++;
        evaluate(data_, model_, solu_);
    }
    vector_operator::show(model_->mu_set, model_->n_kernel, "mu_set");
    evaluate(data_, model_, solu_);
}


void init() 
{
    if (husky::Context::get_param("is_sparse") == "true") 
    {
        run_dcd_svm();
    } 
    else 
    {
        husky::LOG_I << "Dense data format is not supported";
    }
}

int main(int argc, char** argv) 
{
    std::vector<std::string> args({"hdfs_namenode", "hdfs_namenode_port", "train", "test", "B", "C", "format", "is_sparse", "max_out_iter", "max_inn_iter"});
    if (husky::init_with_args(argc, argv, args)) 
    {
        husky::run_job(init);
        return 0;
    }
    return 1;
}
