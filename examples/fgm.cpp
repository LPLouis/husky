/***
    Performance discussion:
    1. Underlying implementation of the DenseVector class and SparseVector class.
    2. Representation of the problem , solution w and alpha
        => Right now I am using DenseVector for both w and alpha, which is not going to work well if the problem is
really of high dimensional
        => Alternatively I can also use SparseVector for w, but this creates a problem when we try to implement it in a
distributed manner because the ParameterServer in Husky accepts a DenseVector, though this problem can be resolved by
mapping SparseIndex to DenseIndex. But this seems too much of a burden on the programmer's side
        => In the original FGM implementation, they actually cache the kernel X_tX_t^T for every dt! This idea seems
implausible and crazy to me at first and I am supprised that they actually take this approach for faster speed
    3. Analysis of the parameter B
        => When I try the FGM algorithm (the one i implemented) on the a9 dataset available on LIBSVM website, I foudn
that if we set B to be 5, 10, 20, or anything less than 100, then the FGM algorithm will actually converge after the
first iteration, i.e., after solving the problem with one violated constraint, it still generates the same constraint in
the next iteration. I think this may be a good sign that this algorithm is not suitable for this dataset in the sense
that although some features may have high violated coefficient (maybe because its features are categorical?), they may
not have enough discriminative power. Interestingly, I try using about 20 features (124 in total), which gives
accuracies of only 12%. If you do the math, if all features are equally important, then having 20 features should give
you 16% accuracy. So is this a good sign that most violated constraint is not a good criterion for selecting features,
though intuitively it looks good (coeff of w)? Or is it all just because the features of this dataset are cetegorical?
    4. Design decison
        => What should SparseVector + SparseVector equal to ? DenseVector or SparseVector, is differentiating these only
during run time a good idea?
    5. Implementation
        => Actually self_dot_elem_wise_dot can be switched back to the original version using combined control variable
        => in the actual FGM implementation by Tan Mingkui, they actually do not use the convergence criterion d_k+1 \in
C_k. Instead, they use the difference of objective (whether it is smaller than epsion) and the values of the newly add
mu_set (whether it is less than 1e-3).
        => Also i realized that we actually do not need to split the data featurewise since we need to compute global w
anyway


    This implementation uses sparse xsp where each x is of dimension B
    it replaces all the functions with the actual implementation to speed up
    it has a fast dcd svm solver
    it incorporates normalization when selecting the most violated constraint
    it also changes the computation of the duality gap
    it normalizes the sparse matrix xsp inside cache_xsp

    In terms of classification accuracy it is quite consistent but the speed is still too slow
    next: instead of letting each x have a fixed dimension of B, use feature node notation

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
#define EPS 1.0e-2
#define MAX_NUM_KERNEL 20

struct feature_node {
    int index;  // index of the feature inside wsp, inside the range of [0, B)
    double value;
};

class data {
   public:
    int n;  // number of features (appended 1 included)
    int l;
    husky::ObjList<ObjT>* train_set;
    husky::ObjList<ObjT>* test_set;
    feature_node*** xsp;
    double* norm;

    data() { xsp = new feature_node**[MAX_NUM_KERNEL]; }

    ~data() {
        if (xsp != NULL) {
            delete[] xsp;
        }
        if (norm != NULL) {
            delete[] norm;
        }
    }
};

class model {
   public:
    int B;
    double C;
    int n_kernel;
    int max_out_iter;
    int max_inn_iter;
    double* mu_set;
    int** dt_set;

    model() {
        n_kernel = 0;
        mu_set = new double[MAX_NUM_KERNEL];
        dt_set = new int*[MAX_NUM_KERNEL];
    }

    model(const model* m) {
        int i, j;
        B = m->B;
        C = m->C;
        max_out_iter = m->max_out_iter;
        max_inn_iter = m->max_inn_iter;
        n_kernel = m->n_kernel;
        mu_set = new double[MAX_NUM_KERNEL];
        for (i = 0; i < n_kernel; i++) {
            mu_set[i] = m->mu_set[i];
        }
        dt_set = new int*[MAX_NUM_KERNEL];
        for (i = 0; i < n_kernel; i++) {
            dt_set[i] = new int[B];
            for (j = 0; j < B; j++) {
                dt_set[i][j] = m->dt_set[i][j];
            }
        }
    }

    void add_dt(int* dt, int B) {
        assert(n_kernel <= MAX_NUM_KERNEL && "n_kernel exceeds MAX_NUM_KERNEL");
        dt_set[n_kernel] = dt;
        n_kernel += 1;
        std::fill_n(mu_set, n_kernel, 1.0 / n_kernel);
    }

    ~model() {
        if (mu_set != NULL) {
            delete[] mu_set;
        }
        if (dt_set != NULL) {
            for (int i = 0; i < n_kernel; i++) {
                delete[] dt_set[i];
            }
            delete[] dt_set;
        }
    }
};

class solu {
   public:
    double obj;
    double* alpha;
    double* wsp;

    solu() {
        alpha = NULL;
        wsp = NULL;
    }

    ~solu() {
        if (alpha != NULL) {
            delete[] alpha;
        }
        if (wsp != NULL) {
            delete[] wsp;
        }
    }
};

class vector_operator {
   public:
    static bool sparse_equal(int* dt, int* v, int B) {
        bool flag = true;
        for (int i = 0; i < B; i++) {
            if (dt[i] != v[i]) {
                flag = false;
                break;
            }
        }
        return flag;
    }

    static bool element_at(int* dt, int** dt_set, int n_kernel, int B) {
        bool flag = false;
        for (int i = 0; i < n_kernel; i++) {
            if (sparse_equal(dt, dt_set[i], B)) {
                flag = true;
                break;
            }
        }
        return flag;
    }

    static inline bool double_equals(double a, double b, double epsilon = 1.0e-6) { return std::abs(a - b) < epsilon; }

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
        husky::LOG_I << ret;
    }

    static void show(const SparseVector<double>& vec, std::string message_head) {
        std::string ret = message_head + ": ";
        for (auto it = vec.begin(); it != vec.end(); it++) {
            ret += std::to_string((*it).fea) + ":" + std::to_string((*it).val) + " ";
        }
        husky::LOG_I << ret;
    }

    static void show(int* dt, int B, std::string message_head) {
        std::string ret = message_head + ": ";
        for (int i = 0; i < B; i++) {
            ret += std::to_string(i) + ":" + std::to_string(dt[i]) + " ";
        }
        husky::LOG_I << ret;
    }

    static void show(double* dt, int B, std::string message_head) {
        std::string ret = message_head + ": ";
        for (int i = 0; i < B; i++) {
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

    static void my_min(const double* vet, const int size, double* min_value, int* min_index) {
        int i;
        double tmp = vet[0];
        min_index[0] = 0;

        for (i = 0; i < size; i++) {
            if (vet[i] < tmp) {
                tmp = vet[i];
                min_index[0] = i;
            }
        }
        min_value[0] = tmp;
    }

    static void my_soft_min(const double* mu_set, const double* desc, const int n_kernel, double* step_max) {
        int i;
        int flag = 1;
        for (i = 0; i < n_kernel; i++) {
            if (desc[i] < 0) {
                if (flag == 1) {
                    step_max[0] = -mu_set[i] / desc[i];
                    flag = 0;
                } else {
                    double tmp = -mu_set[i] / desc[i];
                    if (tmp < step_max[0]) {
                        step_max[0] = tmp;
                    }
                }
            }
        }
    }

    static void my_max(const double* vet, const int size, double* max_value, int* max_index) {
        int i;
        double tmp = vet[0];
        max_index[0] = 0;

        for (i = 0; i < size; i++) {
            if (vet[i] > tmp) {
                tmp = vet[i];
                max_index[0] = i;
            }
        }
        max_value[0] = tmp;
    }

    static double dot_product(const double* w, const SparseVector<double>& v, const int n) {
        assert(v.get_feature_num() == n && "dot_product: error\n");
        double ret = 0;
        auto it = v.begin();
        while (it != v.end()) {
            ret += w[(*it).fea] * (*it).val;
            it++;
        }
        return ret;
    }
};

void initialize(data* data_, model* model_) {
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
    // uncomment the following if bias is needed (note we will train the bias instead of obtaining it after w is solved
    // for sake of simplicity)
    // for (auto& labeled_point : train_set_data) {
    //     labeled_point.x.resize(n + 1);
    //     labeled_point.x.set(n, 1);
    // }
    // for (auto& labeled_point : test_set.get_data()) {
    //     labeled_point.x.resize(n + 1);
    //     labeled_point.x.set(n, 1);
    // }

    // n += 1;

    int l = train_set_data.size();
    data_->n = n;
    data_->l = l;

    double* norm = new double[n];
    std::fill_n(norm, n, 0);
    for (auto& labeled_point : train_set_data) {
        auto it = labeled_point.x.begin();
        while (it != labeled_point.x.end()) {
            norm[(*it).fea] += (*it).val * (*it).val;
            it++;
        }
    }
    for (int i = 0; i < n; i++) {
        norm[i] = sqrt(norm[i]);
    }
    data_->norm = norm;

    husky::LOG_I << "number of samples: " + std::to_string(l);
    husky::LOG_I << "number of features: " + std::to_string(n);
}

void destroy(data* data_, model* model_, solu* solu_) {
    int i, j;
    int l = data_->l;
    int n_kernel = model_->n_kernel;

    for (i = 0; i < n_kernel; i++) {
        for (j = 0; j < l; j++) {
            delete[] data_->xsp[i][j];
        }
        delete[] data_->xsp[i];
    }
}

int* most_violated(data* data_, model* model_, solu* solu_ = NULL) {
    int i, j;
    auto& train_set_data = data_->train_set->get_data();
    double* alpha;

    if (solu_ == NULL) {
        // set alpha to 1 because we do not know how much data others have
        alpha = new double[data_->l];
        std::fill_n(alpha, data_->l, 1.0);
    } else {
        alpha = solu_->alpha;
    }

    std::vector<std::pair<int, double>> fea_score;

    double* w = new double[data_->n];
    std::fill_n(w, data_->n, 0);
    for (i = 0; i < data_->l; i++) {
        auto xi = train_set_data[i].x;
        double diff = train_set_data[i].y * alpha[i];
        auto it = xi.begin();
        while (it != xi.end()) {
            w[(*it).fea] += diff * (*it).val;
            it++;
        }
        // vector_operator::add_sparse(w, train_set_data[i].x, train_set_data[i].y * alpha[i], data_->n);
    }

    double* norm = data_->norm;
    for (i = 0; i < data_->n; i++) {
        if (norm[i] == 0) {
            fea_score.push_back(std::make_pair(i, 0));
        } else {
            fea_score.push_back(std::make_pair(i, fabs(w[i]) / norm[i]));
        }
    }
    std::sort(fea_score.begin(), fea_score.end(), [](auto& left, auto& right) { return left.second > right.second; });

    int* dt = new int[model_->B];
    for (i = 0; i < model_->B; i++) {
        int fea = fea_score[i].first;
        dt[i] = fea;
    }

    delete[] w;
    if (solu_ == NULL) {
        delete[] alpha;
    }

    std::sort(dt, dt + model_->B);
    return dt;
}

// this function caches the new kernel corresponding to the given dt
// this function modifies n_kernel, mu_set and dt_set inside model
void cache_xsp(data* data_, model* model_, int* dt, int B) {
    int i, j;
    int l = data_->l;
    int n_kernel = model_->n_kernel;
    auto& train_set_data = data_->train_set->get_data();

    // cache new kernel
    data_->xsp[n_kernel] = new feature_node*[l];
    for (i = 0; i < l; i++) {
        auto xi = train_set_data[i].x;
        // at most B + 1 elements;
        feature_node* tmp = new feature_node[B + 1];
        feature_node* tmp_orig = tmp;
        auto it = xi.begin();
        j = 0;
        while (it != xi.end() && j != B) {
            int fea = (*it).fea;
            int dt_idx = dt[j];
            if (dt_idx == fea) {
                tmp->index = j;
                tmp->value = (*it).val;
                // tmp->value = (*it).val / norm[fea];
                j++;
                it++;
                tmp++;
            } else if (dt_idx < fea) {
                j++;
            } else {
                it++;
            }
        }
        tmp->index = -1;
        data_->xsp[n_kernel][i] = tmp_orig;
    }
    // modify model_
    model_->add_dt(dt, B);
}

// this function assumes data inside output_solu_ is not newed
void dcd_svm(data* data_, model* model_, solu* output_solu_, solu* input_solu_ = NULL, double* QD = NULL,
             int* index = NULL, bool cache = false) {
    // Declaration and Initialization
    int l = data_->l;
    int n = data_->n;
    const auto& train_set_data = data_->train_set->get_data();
    double* mu_set = model_->mu_set;
    int** dt_set = model_->dt_set;
    feature_node*** xsp = data_->xsp;

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
    if (input_solu_ == NULL) {
        std::fill_n(alpha, l, 0);
        std::fill_n(wsp, n_kernel * B, 0);
    } else {
        for (i = 0; i < l; i++) {
            alpha[i] = input_solu_->alpha[i];
        }
        for (i = 0; i < n_kernel * B; i++) {
            wsp[i] = input_solu_->wsp[i];
        }
    }
    if (!cache) {
        QD = new double[l];
        index = new int[l];
        for (i = 0; i < l; i++) {
            QD[i] = 0;
            for (k = 0; k < n_kernel; k++) {
                feature_node* x = xsp[k][i];
                double tmp = 0;
                while (x->index != -1) {
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

    int iter = 0;
    while (iter < model_->max_inn_iter) {
        int clock_start_time = clock();

        PGmax_new = 0;
        PGmin_new = 0;

        for (i = 0; i < active_size; i++) {
            int j = i + rand() % (active_size - i);
            vector_operator::swap(index[i], index[j]);
        }

        for (s = 0; s < active_size; s++) {
            i = index[s];
            G = 0;
            double yi = train_set_data[i].y;

            G = 0;
            for (j = 0; j < n_kernel; j++) {
                double* wsp_tmp_ptr = &wsp[j * B];
                feature_node* x = xsp[j][i];
                double tmp = 0;
                while (x->index != -1) {
                    tmp += wsp_tmp_ptr[x->index] * x->value;
                    x++;
                }
                G += mu_set[j] * tmp;
                // G += mu_set[j] * vector_operator::dot_product(&wsp[j * B], xsp[j][i], B);
            }
            G = G * yi - 1 + diag * alpha[i];

            PG = 0;
            if (alpha[i] == 0) {
                if (G > PGmax_old) {
                    active_size--;
                    vector_operator::swap(index[s], index[active_size]);
                    s--;
                    continue;
                } else if (G < 0) {
                    PG = G;
                    PGmin_new = std::min(PGmin_new, PG);
                }
            } else if (alpha[i] == INF) {
                if (G < PGmin_old) {
                    active_size--;
                    vector_operator::swap(index[s], index[active_size]);
                    s--;
                    continue;
                } else if (G > 0) {
                    PG = G;
                    PGmax_new = std::max(PGmax_new, PG);
                }
            } else {
                PG = G;
                PGmax_new = std::max(PGmax_new, PG);
                PGmin_new = std::min(PGmin_new, PG);
            }

            if (fabs(PG) > 1.0e-12) {
                double alpha_old = alpha[i];
                alpha[i] = std::min(std::max(alpha[i] - G / QD[i], 0.0), INF);
                diff = yi * (alpha[i] - alpha_old);
                for (j = 0; j < n_kernel; j++) {
                    double* wsp_tmp_ptr = &wsp[j * B];
                    feature_node* x = xsp[j][i];
                    while (x->index != -1) {
                        wsp_tmp_ptr[x->index] += x->value * diff;
                        x++;
                    }
                    // vector_operator::add_sparse(&wsp[j * B], xsp[j][i], diff, B);
                }
            }
        }

        iter++;

        if (PGmax_new - PGmin_new <= EPS) {
            if (active_size == l)
                break;
            else {
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

    double obj = 0;
    for (i = 0; i < l; i++) {
        obj += alpha[i] * (1 - diag * alpha[i]);
    }
    for (i = 0; i < n_kernel; i++) {
        double* wsp_tmp_ptr = &wsp[i * B];
        double tmp = 0;
        for (j = 0; j < B; j++) {
            tmp += wsp_tmp_ptr[j] * wsp_tmp_ptr[j];
        }
        obj -= 0.5 * mu_set[i] * tmp;
        // obj -= 0.5 * mu_set[i] * vector_operator::self_dot_product(&wsp[i * B], B);
    }

    if (!cache) {
        delete[] index;
        delete[] QD;
    }
    output_solu_->obj = obj;
    output_solu_->wsp = wsp;
    output_solu_->alpha = alpha;
}

// mu_set, wsp, alpha and obj inside model actually will not be used
// this function assumes mu_set, wsp and alpha are newed
void fast_dcd_svm(data* data_, model* model_, double* mu_set, double* wsp, double* alpha, double* obj_,
                  double* QD = NULL, int* index = NULL, bool cache = false) {
    // clock_t start = clock();
    // Declaration and Initialization

    int l = data_->l;
    int n = data_->n;
    const auto& train_set_data = data_->train_set->get_data();

    int** dt_set = model_->dt_set;
    feature_node*** xsp = data_->xsp;

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

    double diff;

    std::fill_n(alpha, l, 0);
    std::fill_n(wsp, n_kernel * B, 0);

    if (!cache) {
        QD = new double[l];
        index = new int[l];
        for (i = 0; i < l; i++) {
            QD[i] = 0;
            for (k = 0; k < n_kernel; k++) {
                double tmp = 0;
                feature_node* x = xsp[k][i];
                while (x->index != -1) {
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

    int iter = 0;
    while (iter < model_->max_inn_iter) {
        PGmax_new = 0;
        PGmin_new = 0;

        for (i = 0; i < active_size; i++) {
            int j = i + rand() % (active_size - i);
            vector_operator::swap(index[i], index[j]);
        }

        for (s = 0; s < active_size; s++) {
            i = index[s];
            G = 0;
            double yi = train_set_data[i].y;

            for (j = 0; j < n_kernel; j++) {
                double tmp = 0;
                feature_node* x = xsp[j][i];
                double* wsp_tmp_ptr = &wsp[j * B];
                while (x->index != -1) {
                    tmp += wsp_tmp_ptr[x->index] * x->value;
                    x++;
                }
                G += mu_set[j] * tmp;
                // G += mu_set[j] * vector_operator::dot_product(&wsp[j * B], xsp[j][i], B);
            }
            G = G * yi - 1 + diag * alpha[i];

            PG = 0;
            if (alpha[i] == 0) {
                if (G > PGmax_old) {
                    active_size--;
                    vector_operator::swap(index[s], index[active_size]);
                    s--;
                    continue;
                } else if (G < 0) {
                    PG = G;
                    PGmin_new = std::min(PGmin_new, PG);
                }
            } else if (alpha[i] == INF) {
                if (G < PGmin_old) {
                    active_size--;
                    vector_operator::swap(index[s], index[active_size]);
                    s--;
                    continue;
                } else if (G > 0) {
                    PG = G;
                    PGmax_new = std::max(PGmax_new, PG);
                }
            } else {
                PG = G;
                PGmax_new = std::max(PGmax_new, PG);
                PGmin_new = std::min(PGmin_new, PG);
            }

            if (fabs(PG) > 1.0e-12) {
                double alpha_old = alpha[i];
                alpha[i] = std::min(std::max(alpha[i] - G / QD[i], 0.0), INF);
                diff = yi * (alpha[i] - alpha_old);
                for (j = 0; j < n_kernel; j++) {
                    // vector_operator::add_sparse(&wsp[j * B], xsp[j][i], diff, B);
                    feature_node* x = xsp[j][i];
                    double* wsp_tmp_ptr = &wsp[j * B];
                    while (x->index != -1) {
                        wsp_tmp_ptr[x->index] += diff * x->value;
                        x++;
                    }
                }
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
    }

    double obj = 0;
    for (i = 0; i < l; i++) {
        obj += alpha[i] * (1 - diag * alpha[i]);
    }
    double tmp;
    for (i = 0; i < n_kernel; i++) {
        // obj -= 0.5 * mu_set[i] * vector_operator::self_dot_product(&wsp[i * B], B);
        tmp = 0;
        double* wsp_tmp_ptr = &wsp[i * B];
        for (j = 0; j < B; j++) {
            tmp += wsp_tmp_ptr[j] * wsp_tmp_ptr[j];
        }
        obj -= 0.5 * mu_set[i] * tmp;
    }

    double primal = 0;
    for (i = 0; i < n_kernel; i++) {
        tmp = 0;
        double* wsp_tmp_ptr = &wsp[i * B];
        for (j = 0; j < B; j++) {
            tmp += wsp_tmp_ptr[j] * wsp_tmp_ptr[j];
        }
        primal += 0.5 * mu_set[i] * tmp;
    }
    for (i = 0; i < l; i++) {
        double yi = train_set_data[i].y;
        double loss = 0;
        for (j = 0; j < n_kernel; j++) {
            tmp = 0;
            double* tmp_wsp = &wsp[j * B];
            feature_node* x = xsp[j][i];
            while (x->index != -1) {
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
            primal += C * loss * loss;
        }
    }

    *obj_ = primal;

    if (!cache) {
        delete[] index;
        delete[] QD;
    }
    // *obj_ = obj;

    // clock_t end = clock();
    // double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    // husky::LOG_I << "DCD_SVM: time elapsed: " + std::to_string(time_spent);
    // husky::LOG_I << "DCD_SVM: inn_iter: " + std::to_string(iter);
}

void simpleMKL(data* data_, model* model_, solu* solu_) {
    clock_t start = clock();
    int i, j, k;
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
    feature_node*** xsp = data_->xsp;
    double diag = 0.5 / model_->C;
    double* QD = new double[l];
    int* index = new int[l];

    for (i = 0; i < l; i++) {
        QD[i] = 0;
        for (j = 0; j < n_kernel; j++) {
            tmp = 0;
            feature_node* x = xsp[j][i];
            while (x->index != -1) {
                tmp += x->value * x->value;
                x++;
            }
            QD[i] += mu_set[j] * tmp;
            // QD[i] += mu_set[j] * vector_operator::self_dot_product(xsp[j][i], B);
        }
        QD[i] += diag;
        index[i] = i;
    }

    fast_dcd_svm(data_, model_, mu_set, solu_->wsp, solu_->alpha, &solu_->obj, QD, index, true);
    double obj = solu_->obj;
    double* alpha = solu_->alpha;
    double* wsp = solu_->wsp;
    // compute gradient
    double* grad = new double[n_kernel];
    for (i = 0; i < n_kernel; i++) {
        tmp = 0;
        double* wsp_tmp_ptr = &wsp[i * B];
        for (j = 0; j < B; j++) {
            tmp += wsp_tmp_ptr[j] * wsp_tmp_ptr[j];
        }
        grad[i] = -0.5 * tmp;
        // grad[i] = -0.5 * vector_operator::self_dot_product(&wsp[i * B], B);
    }

    while (loop == 1 && maxloop > 0 && n_kernel > 1) {
        nloop++;

        double old_obj = obj;

        double* new_mu_set = new double[n_kernel];
        for (i = 0; i < n_kernel; i++) {
            new_mu_set[i] = mu_set[i];
        }

        // normalize gradient
        double sum_grad = 0;
        for (i = 0; i < n_kernel; i++) {
            sum_grad += grad[i] * grad[i];
        }
        double sqrt_grad = sqrt(sum_grad);
        for (i = 0; i < n_kernel; i++) {
            grad[i] /= sqrt_grad;
        }

        // compute descent direction
        double max_mu = 0;
        int max_index = 0;

        vector_operator::my_max(mu_set, n_kernel, &max_mu, &max_index);
        double grad_tmp = grad[max_index];
        for (i = 0; i < n_kernel; i++) {
            grad[i] -= grad_tmp;
        }

        double* desc = new double[n_kernel];
        double sum_desc = 0;
        for (i = 0; i < n_kernel; i++) {
            if (mu_set[i] > 0 || grad[i] < 0) {
                desc[i] = -grad[i];
            } else {
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
        if (step_max == 0) {
            flag = 0;
        }

        if (flag == 1) {
            if (step_max > 0.1) {
                step_max = 0.1;
                delta_max = step_max;
            }

            // model* tmp_model = new model(model_);
            // solu* tmp_solu = new solu(l, n, B, n_kernel);
            double* tmp_mu_set = new double[n_kernel];
            double* tmp_wsp = new double[n_kernel * B];
            double* tmp_alpha = new double[l];
            double tmp_obj;

            while (cost_max < cost_min) {
                // husky::LOG_I << "descent direction search";
                for (i = 0; i < n_kernel; i++) {
                    tmp_mu_set[i] = new_mu_set[i] + step_max * desc[i];
                }
                // use descent direction to compute new objective
                fast_dcd_svm(data_, model_, tmp_mu_set, tmp_wsp, tmp_alpha, &tmp_obj, QD, index, true);
                cost_max = tmp_obj;
                if (cost_max < cost_min) {
                    cost_min = cost_max;

                    for (i = 0; i < n_kernel; i++) {
                        new_mu_set[i] = tmp_mu_set[i];
                        mu_set[i] = tmp_mu_set[i];
                    }

                    sum_desc = 0;
                    int fflag = 1;
                    for (i = 0; i < n_kernel; i++) {
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
                    solu_->obj = tmp_obj;
                    for (i = 0; i < l; i++) {
                        alpha[i] = tmp_alpha[i];
                    }
                    for (i = 0; i < n_kernel * B; i++) {
                        wsp[i] = tmp_wsp[i];
                    }

                    if (fflag) {
                        step_max = 0;
                        delta_max = 0;
                    } else {
                        // we can still descend, loop again
                        vector_operator::my_soft_min(new_mu_set, desc, n_kernel, &step_max);
                        delta_max = step_max;
                        cost_max = 0;
                    }  // if (fflag)
                }      // if (cost_max < cost_min)
            }          // while (cost_max < cost_min)

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
            while ((step_max - step_min) > 1e-1 * fabs(delta_max) && step_max > 1e-12) {
                // husky::LOG_I << "line search";
                step_loop += 1;
                if (step_loop > 8) {
                    break;
                }
                double step_medr = step_min + (step_max - step_min) / gold_ratio;
                double step_medl = step_min + (step_medr - step_min) / gold_ratio;

                // half
                for (i = 0; i < n_kernel; i++) {
                    tmp_mu_set_1[i] = new_mu_set[i] + step_medr * desc[i];
                }
                fast_dcd_svm(data_, model_, tmp_mu_set_1, tmp_wsp_1, tmp_alpha_1, &tmp_obj_1, QD, index, true);

                // half half
                for (i = 0; i < n_kernel; i++) {
                    tmp_mu_set_2[i] = new_mu_set[i] + step_medl * desc[i];
                }
                fast_dcd_svm(data_, model_, tmp_mu_set_2, tmp_wsp_2, tmp_alpha_2, &tmp_obj_2, QD, index, true);

                step[0] = step_min;
                step[1] = step_medl;
                step[2] = step_medr;
                step[3] = step_max;

                cost[0] = cost_min;
                cost[1] = tmp_obj_2;
                cost[2] = tmp_obj_1;
                cost[3] = cost_max;

                vector_operator::my_min(cost, 4, &min_val, &min_idx);

                switch (min_idx) {
                case 0:
                    step_max = step_medl;
                    cost_max = cost[1];
                    solu_->obj = tmp_obj_2;
                    for (i = 0; i < n_kernel * B; i++) {
                        wsp[i] = tmp_wsp_2[i];
                    }
                    for (i = 0; i < l; i++) {
                        alpha[i] = tmp_alpha_2[i];
                    }
                    break;

                case 1:
                    step_max = step_medr;
                    cost_max = cost[2];
                    solu_->obj = tmp_obj_1;
                    for (i = 0; i < n_kernel * B; i++) {
                        wsp[i] = tmp_wsp_1[i];
                    }
                    for (i = 0; i < l; i++) {
                        alpha[i] = tmp_alpha_1[i];
                    }
                    break;

                case 2:
                    step_min = step_medl;
                    cost_min = cost[1];
                    solu_->obj = tmp_obj_2;
                    for (i = 0; i < n_kernel * B; i++) {
                        wsp[i] = tmp_wsp_2[i];
                    }
                    for (i = 0; i < l; i++) {
                        alpha[i] = tmp_alpha_2[i];
                    }
                    break;

                case 3:
                    step_min = step_medr;
                    cost_min = cost[2];
                    solu_->obj = tmp_obj_1;
                    for (i = 0; i < n_kernel * B; i++) {
                        wsp[i] = tmp_wsp_1[i];
                    }
                    for (i = 0; i < l; i++) {
                        alpha[i] = tmp_alpha_1[i];
                    }
                    break;
                }  // switch(min_idx);
            }      // while ((step_max - step_min) > 1e-1 * fabs(delta_max) && step_max > 1e-12)

            // assignment
            double step_size = step[min_idx];
            if (solu_->obj < old_obj) {
                for (i = 0; i < n_kernel; i++) {
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
            delete[] tmp_alpha_2;
            delete[] tmp_wsp_2;
            delete[] tmp_mu_set_2;

            delete[] tmp_alpha_1;
            delete[] tmp_wsp_1;
            delete[] tmp_mu_set_1;

            delete[] tmp_alpha;
            delete[] tmp_wsp;
            delete[] tmp_mu_set;

            delete cost;
            delete step;
        }  // if(flag)

        // test convergence
        double mu_max;
        int mu_max_idx;

        vector_operator::my_max(mu_set, n_kernel, &mu_max, &mu_max_idx);
        // normalize mu_max
        if (mu_max > 1e-12) {
            double mu_sum = 0;
            for (i = 0; i < n_kernel; i++) {
                if (mu_set[i] < 1e-12) {
                    mu_set[i] = 0;
                }
                mu_sum += mu_set[i];
            }
            for (i = 0; i < n_kernel; i++) {
                mu_set[i] /= mu_sum;
            }
        }

        for (i = 0; i < n_kernel; i++) {
            tmp = 0;
            double* wsp_tmp_ptr = &wsp[i * B];
            for (j = 0; j < B; j++) {
                tmp += wsp_tmp_ptr[j] * wsp_tmp_ptr[j];
            }
            grad[i] = -0.5 * tmp;
            // grad[i] = -0.5 * vector_operator::self_dot_product(&wsp[i * B], B);
        }
        double min_grad = 0;
        double max_grad = 0;
        int ffflag = 1;
        for (i = 0; i < n_kernel; i++) {
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

        // double KKTconstraint = fabs(min_grad - max_grad) / fabs(min_grad);

        double* tmp_grad = new double[n_kernel];
        for (i = 0; i < n_kernel; i++) {
            tmp_grad[i] = -1 * grad[i];
        }
        double max_tmp;
        int max_tmp_idx;
        vector_operator::my_max(tmp_grad, n_kernel, &max_tmp, &max_tmp_idx);

        double tmp_sum = 0;
        for (i = 0; i < l; i++) {
            tmp_sum += alpha[i] * (1 - alpha[i] * diag);
            // tmp_sum += alpha[i];
        }

        double mkl_grad = 0;
        for (i = 0; i < n_kernel; i++) {
            tmp = 0;
            double* tmp_wsp = &wsp[i * B];
            for (j = 0; j < B; j++) {
                tmp += tmp_wsp[j] * tmp_wsp[j];
            }
            mkl_grad += mu_set[i] * tmp;
        }
        mkl_grad *= 0.5;
        double dual_gap = (mkl_grad - max_tmp) / mkl_grad;
        // double dual_gap = (solu_->obj + max_tmp - tmp_sum) / solu_->obj;
        // husky::LOG_I << "[outer loop][dual_gap]: " + std::to_string(fabs(dual_gap));
        // husky::LOG_I << "[outer loop][KKTconstraint]: " + std::to_string(KKTconstraint);
        // if (KKTconstraint < 0.05 || fabs(dual_gap) < 0.001) {
        if (fabs(dual_gap) < 0.01) {
            loop = 0;
        }
        if (nloop > maxloop) {
            loop = 0;
            break;
        }

        delete[] tmp_grad;
        delete[] desc;
        delete[] new_mu_set;
    }

    delete[] grad;
    // clock_t end = clock();
    // husky::LOG_I << "time elapsed " + std::to_string((double)(end - start) / CLOCKS_PER_SEC);
}

double evaluate(data* data_, model* model_, solu* solu_) {
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
    for (i = 0; i < n_kernel; i++) {
        double* wsp_tmp_ptr = &wsp[i * B];
        int* dt_set_tmp = dt_set[i];
        for (j = 0; j < B; j++) {
            w[dt_set_tmp[j]] += mu_set[i] * wsp_tmp_ptr[j];
            // w[dt_set[i][j]] += mu_set[i] * wsp[i * B + j];
        }
    }
    const auto& test_set_data = data_->test_set->get_data();

    double indicator;
    int error_agg = 0;

    for (auto labeled_point : test_set_data) {
        double tmp = 0;
        auto it = labeled_point.x.begin();
        while (it != labeled_point.x.end()) {
            tmp += w[(*it).fea] * (*it).val;
            it++;
        }
        indicator = labeled_point.y * tmp;
        // indicator = labeled_point.y * vector_operator::dot_product(w, labeled_point.x, data_->n);
        if (indicator <= 0) {
            error_agg += 1;
        }
    }

    husky::LOG_I << "Classification accuracy on testing set with [B = " + std::to_string(model_->B) + "][C = " +
                        std::to_string(model_->C) + "], " + "[max_out_iter = " + std::to_string(model_->max_out_iter) +
                        "], " + "[max_inn_iter = " + std::to_string(model_->max_inn_iter) + "], " +
                        "[test set size = " + std::to_string(test_set_data.size()) + "]: " +
                        std::to_string(1.0 - static_cast<double>(error_agg) / test_set_data.size());

    delete[] w;
    return 1.0 - static_cast<double>(error_agg) / test_set_data.size();
}

void run_dcd_svm() {
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
    int* dt = most_violated(data_, model_);
    cache_xsp(data_, model_, dt, model_->B);  // cache xsp, add dt to model and increment number of kernels
    husky::LOG_I << "cache completed! number of kernel: " + std::to_string(model_->n_kernel);
    dcd_svm(data_, model_, solu_);
    vector_operator::show(model_->mu_set, model_->n_kernel, "mu_set");
    husky::LOG_I << "trainning objective: " + std::to_string(solu_->obj);
    evaluate(data_, model_, solu_);
    destroy(data_, model_, solu_);
    auto end = std::chrono::steady_clock::now();
    husky::LOG_I << "Time elapsed: " << std::chrono::duration_cast<std::chrono::duration<float>>(end - start).count();
}

void job_runner() {
    data* data_ = new data;
    model* model_ = new model;
    solu* solu_ = new solu;

    initialize(data_, model_);
    clock_t tot_time = clock();
    double* kernel_acc = new double[MAX_NUM_KERNEL];
    double* kernel_time = new double[MAX_NUM_KERNEL];

    int iter = 0;
    int max_out_iter = model_->max_out_iter;
    double mkl_obj = INF;
    double last_obj = INF;
    double obj_diff = 0.0;
    int* dt;

    clock_t start, end;
    while (iter < max_out_iter) {
        start = clock();
        last_obj = mkl_obj;
        if (iter == 0) {
            dt = most_violated(data_, model_);
        } else {
            dt = most_violated(data_, model_, solu_);
        }

        cache_xsp(data_, model_, dt, model_->B);
        if (solu_->wsp == NULL && solu_->alpha == NULL) {
            solu_->wsp = new double[model_->n_kernel * model_->B];
            solu_->alpha = new double[data_->l];
        } else {
            delete[] solu_->wsp;
            delete[] solu_->alpha;
            solu_->wsp = new double[model_->n_kernel * model_->B];
            solu_->alpha = new double[data_->l];
        }
        simpleMKL(data_, model_, solu_);
        end = clock();
        kernel_time[iter] = (double) (end - start) / CLOCKS_PER_SEC;
        kernel_acc[iter] = evaluate(data_, model_, solu_);

        mkl_obj = solu_->obj;
        obj_diff = fabs(mkl_obj - last_obj);
        husky::LOG_I << "[iteration " + std::to_string(iter) + "][mkl_obj " + std::to_string(mkl_obj) + "][obj_diff " +
                            std::to_string(obj_diff) + "]";
        // if (mkl_obj > last_obj)
        // {
        //     husky::LOG_I << "mkl_obj > last_obj: FGM converged";
        //     break;
        // }
        // if (obj_diff < 0.0001 * abs(last_obj))
        // {
        //     husky::LOG_I << "obj_diff < 0.001 * abs(last_obj): FGM converged";
        //     break;
        // }
        // if (model_->mu_set[iter] < 0.0001)
        // {
        //     husky::LOG_I << "model_->mu_set[iter] < 0.0001: FGM converged";
        //     break;
        // }
        iter++;
    }
    double elapsed = (double) (clock() - tot_time) / CLOCKS_PER_SEC;
    if (iter == max_out_iter) {
        iter = max_out_iter - 1;
    }
    int** dt_set = model_->dt_set;

    FILE* dout = fopen("fgm_rcv1_dt", "w");
    for (int i = 0; i <= iter; i++) {
        for (int j = 0; j < model_->B; j++) {
            fprintf(dout, "%d ", dt_set[i][j]);
        }
        fprintf(dout, "\n");
    }

    // FILE* fout = fopen("fgm_syn_large_plot.csv", "w");
    FILE* fout = fopen("fgm_rcv1_plot.csv", "w");
    fprintf(fout, "n_kernel.Vs.accuracy ");
    for (int i = 0; i <= iter; i++) {
        fprintf(fout, "%f ", kernel_acc[i]);
    }
    fprintf(fout, "\n");
    fprintf(fout, "n_kernel.Vs.train_time ");
    for (int i = 0; i <= iter; i++) {
        fprintf(fout, "%f ", kernel_time[i]);
    }
    fprintf(fout, "\n");
    fprintf(fout, "mu_set ");
    for (int i = 0; i <= iter; i++) {
        fprintf(fout, "%f ", model_->mu_set[i]);
    }
    fprintf(fout, "\n");
    fprintf(fout, "total time: %f\n", elapsed);

    delete[] kernel_acc;
    delete[] kernel_time;

    evaluate(data_, model_, solu_);
    destroy(data_, model_, solu_);
}

void init() {
    if (husky::Context::get_param("is_sparse") == "true") {
        job_runner();
    } else {
        husky::LOG_I << "Dense data format is not supported";
    }
}

int main(int argc, char** argv) {
    std::vector<std::string> args({"hdfs_namenode", "hdfs_namenode_port", "train", "test", "B", "C", "format",
                                   "is_sparse", "max_out_iter", "max_inn_iter"});
    if (husky::init_with_args(argc, argv, args)) {
        husky::run_job(init);
        return 0;
    }
    return 1;
}
