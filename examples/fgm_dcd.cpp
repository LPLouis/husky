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

#include "glpk.h"

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

#define EQUAL_VALUE(a, b) (a - b < 1.0e-6) && (a - b > -1.0e-6)
#define NOT_EQUAL(a, b) (a - b > 1.0e-6) || (a - b < -1.0e-6)
#define EPS 1.0e-12
#define MU_EPS 1.0e-2
#define ALPHA_EPS 1.0e-2
#define INF std::numeric_limits<double>::max()

class data {
public:
    int n;          // number of features (appended 1 included)
    int l;         
    // DenseVector<double> label;
    husky::ObjList<ObjT>* train_set;
    // husky::ObjList<ObjT>* train_set_fw;
    husky::ObjList<ObjT>* test_set;
};

class model {
public:
    int B;
    double C;
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
    // sparse d + sparse d equals to dense d, so may be don't store \sum_mu_dt may be better?
    // wt in wt_list is controlled but unweighted
    std::vector<SparseVector<double>> wt_list;
    // w_controlled is weighted
    DenseVector<double> w_controlled;

    solu() {}

    solu(int l, int n, int T) {
        obj = 0.0;
        alpha = DenseVector<double>(l, 0.0);
        w = DenseVector<double>(n, 0.0);
        wt_list = std::vector<SparseVector<double>>(T);
        w_controlled = DenseVector<double>(n);
    }
};

// this is no longer needed in this implementation since we now use SimpleMKL to solve the MKL problem
void solve_lp(const std::vector<double>& mu_scores, model* model_) {
    /*
        minimization:
            co1 x1 + co2 x2 + co3 x3 + co4 x4 + ... + con xn
        subject To
            c1: x1 + x2 + x3 + x4 + ... + xn= 1
        Bounds
            0 <= x1
            0 <= x2
            0 <= x3 
            0 <= x4 
            .
            .
            .
            0 <= xn
     */
    int n = mu_scores.size();
    glp_prob *lp = glp_create_prob();
    // suppress terminal output
    glp_term_out(GLP_OFF);
    glp_set_prob_name(lp, "sample");
    glp_set_obj_dir(lp, GLP_MIN);

    // constraint c1: x1 + x2 + x3 + x4 + ... + xn= 1
    glp_add_rows(lp, 1);
    glp_set_row_name(lp, 1, "c1");
    glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0);

    glp_add_cols(lp, n);
    std::string prefix = "x";
    // ** DB OR LB ? **
    for (int i = 0; i < n; i++) {
        // set co1 x1
        glp_set_col_name(lp, i + 1, (prefix + std::to_string(i)).c_str());
        glp_set_col_bnds(lp, i + 1, GLP_LO, 0.0, 0.0);
        glp_set_obj_coef(lp, i + 1, mu_scores[i]);
    }
    // a[1, i] = 1
    int *ia = new int[n + 1];
    int *ja = new int[n + 1];
    double *ar = new double[n + 1];
    for (int i = 1; i <= n; i++) {
        ia[i] = 1;
        ja[i] = i;
        ar[i] = 1;
    }
    glp_load_matrix(lp, n, ia, ja, ar);

    glp_simplex(lp, NULL);
    double z = glp_get_obj_val(lp);
    std::vector<double> ret;
    for (int i = 0; i < n; i++) {
        model_->mu_set[i] =  glp_get_col_prim(lp, i + 1);
    }

    glp_delete_prob(lp);
    if (ia == NULL) {
        delete [] ia;
        delete [] ja;
        delete [] ar;
    }
}

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

SparseVector<double> find_most_violated(data* data_, model* model_, solu* solu_ = NULL) {
    int B = model_->B;
    auto& train_set = data_->train_set;
    DenseVector<double> alpha;
    DenseVector<double> w;
    if (solu_ == NULL) {
        alpha = DenseVector<double>(data_->l, 1.0 / data_->l);
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
    // for (auto& labeled_point_vector : train_set_fw->get_data()) {
    //     fea_score.push_back(std::make_pair(labeled_point_vector.id() - 1, labeled_point_vector.x.dot(cache)));
    // }
    // fea_score.push_back(std::make_pair(data_->n - 1, cache.dot(DenseVector<double>(data_->l, 1.0))));
    int i = 0;
    if (solu_ == NULL) {
        for (auto& labeled_point_vector : train_set->get_data()) {
            w += labeled_point_vector.x * labeled_point_vector.y * alpha[i++];
        }
    }

    for (i = 0; i < data_->n; i++) {
        fea_score.push_back(std::make_pair(i, w[i] * w[i]));
    }

    std::sort(fea_score.begin(), fea_score.end(), [](auto& left, auto& right) {
        return left.second > right.second;
    });
    SparseVector<double> control_variable(data_->n);
    for (int i = 0; i < B; i++) {
        int fea = fea_score[i].first;
        double val = fea_score[i].second;
        control_variable.set(fea, 1.0);
    }
    control_variable.sort_asc();
    return control_variable;
}

template <bool is_sparse = true>
void dcd_svm(data* data_, model* model_, solu* output_solu_, solu* input_solu_ = NULL, double* QD = NULL, int* index = NULL, bool cache = false) {
    // Declaration and Initialization
    int l = data_->l;
    int n = data_->n;
    const auto& labeled_point_vector = data_->train_set->get_data();
    const auto& mu_set = model_->mu_set;
    const auto& dt_set = model_->dt_set;

    // combine all control variables to a single control variable (Assumption: coef of dt sum to 1)
    const DenseVector<double> control_variable = vector_operator::sum_mu_dt(mu_set, dt_set);
    // DenseVector<double>& alpha = problem_->alpha;
    DenseVector<double> alpha;
    if (input_solu_ == NULL) {
        alpha = DenseVector<double>(l, 0.0);
    } else {
        alpha = input_solu_->alpha;
    }
    double diag = 0.5 / model_->C;
    double UB = INF;

    if (!cache) {
        QD = new double[l];
        index = new int[l];
    }

    int iter = 0;

    double G, PG, PGmax_new, PGmin_new;
    double PGmax_old = INF;
    double PGmin_old = -INF;

    int i, k;
    int active_size = l;

    double diff;

    DenseVector<double> w(n, 0.0);
    DenseVector<double> w_controlled(n , 0.0);
    DenseVector<double> LB(l, 0.0);
    DenseVector<double> beta(l, -1.0);

    // QD and index not provided
    if (!cache) {
        for (i = 0; i < l; i++) {
            QD[i] = vector_operator::self_dot_elem_wise_dot(labeled_point_vector[i].x, mu_set, dt_set) + diag;
            // QD[i] = vector_operator::self_dot_product(labeled_point_vector[i].x) + diag;
            index[i] = i;
            w += labeled_point_vector[i].x * labeled_point_vector[i].y * alpha[i];
        }
    } else {
        for (i = 0; i < l; i++) {
            w += labeled_point_vector[i].x * labeled_point_vector[i].y * alpha[i];
        }
    }
    w_controlled = vector_operator::elem_wise_dot(w, control_variable);

    while (iter < model_->max_inn_iter) {
        PGmax_new = 0;
        PGmin_new = 0;
        for (i = 0; i < active_size; i++) {
            int j = i + std::rand() % (active_size - i);
            vector_operator::swap(index[i], index[j]);
        }

        for (k = 0; k < active_size; k++) {
            i = index[k];
            int yi = labeled_point_vector[i].y;
            auto& xi = labeled_point_vector[i].x;

            G = (w_controlled.dot(xi)) * yi + beta[i] + diag * alpha[i];
            // G = (w.dot(xi)) * yi + beta[i] + diag * alpha[i];

            PG = 0;
            if (EQUAL_VALUE(alpha[i], LB[i])) {
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
                alpha[i] = std::min(std::max(alpha[i] - G / QD[i], LB[i]), UB);
                diff = yi * (alpha[i] - alpha_old);
                w += xi * diff;
                w_controlled = vector_operator::elem_wise_dot(w, control_variable);
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

    std::vector<SparseVector<double>> wt_list(mu_set.size());
    for (i = 0; i < wt_list.size(); i++) {
        // ideally this should be quick because we only need to iterate over the sparse entries
        // might want to compute wt_list instead of w_controlled when B is very small
        wt_list[i] = vector_operator::elem_wise_dot(w, dt_set[i]);
    }
    double obj = 0;
    for (i = 0; i < l; i++) {
        obj += alpha[i];
    }

    obj = obj - 0.5 * w.dot(w_controlled) - diag * alpha.dot(alpha);

    if (!cache) {
        delete[] index;
        delete[] QD;
    }
    output_solu_->alpha = alpha;
    output_solu_->w = w;
    output_solu_->w_controlled = w_controlled;
    output_solu_->wt_list = wt_list;
    output_solu_->obj = obj;
}

// this is the dcd_svm using cached kernel X_tX_t^T to prove that our way of using w_controlled is correct
template <bool is_sparse = true>
double dcd_svm_test(data* data_, model* model_, solu* output_solu_, solu* input_solu_ = NULL, double* QD = NULL, int* index = NULL, bool cache = false) {
    // Declaration and Initialization
    int l = data_->l;
    int n = data_->n;
    int i, k;
    int active_size = l;
    const auto& labeled_point_vector = data_->train_set->get_data();
    const auto& mu_set = model_->mu_set;
    const auto& dt_set = model_->dt_set;
    const int T = mu_set.size();

    std::vector<std::vector<SparseVector<double>>> xsp;
    for (i = 0; i < l; i++) {
        auto& x = labeled_point_vector[i].x;
        xsp.push_back(std::vector<SparseVector<double>>());
        for (k = 0; k < T; k++) {
            xsp[i].push_back(vector_operator::elem_wise_dot(x, dt_set[k]));
        }
    }
    husky::LOG_I << "done calculating xsp";

    // combine all control variables to a single control variable (Assumption: coef of dt sum to 1)
    DenseVector<double> alpha = DenseVector<double>(l, 0.0);
    double diag = 0.5 / model_->C;
    double UB = INF;

    if (!cache) {
        QD = new double[l];
        index = new int[l];
    }

    int iter = 0;

    double G, PG, PGmax_new, PGmin_new;
    double PGmax_old = INF;
    double PGmin_old = -INF;

    double diff;

    DenseVector<double> w(n, 0.0);
    std::vector<DenseVector<double>> wt_list(T);
    for (i = 0; i < T; i++) {
        wt_list[i] = DenseVector<double>(n, 0.0);
    }
    DenseVector<double> LB(l, 0.0);
    DenseVector<double> beta(l, -1.0);

    // QD and index not provided
    if (!cache) {
        for (i = 0; i < l; i++) {
            QD[i] = 0;
            for (k = 0; k < T; k++) {
                QD[i] += vector_operator::self_dot_product(xsp[i][k]);
                wt_list[k] += xsp[i][k] * labeled_point_vector[i].y * alpha[i];
            }
            QD[i] += diag;
            index[i] = i;
        }
    } else {
        for (i = 0; i < l; i++) {
            for (k = 0; k < T; k++) {
                wt_list[k] += xsp[i][k] * labeled_point_vector[i].y * alpha[i];
            }
        }
    }

    while (iter < model_->max_inn_iter) {
        PGmax_new = 0;
        PGmin_new = 0;
        for (i = 0; i < active_size; i++) {
            int j = i + std::rand() % (active_size - i);
            vector_operator::swap(index[i], index[j]);
        }
        for (k = 0; k < active_size; k++) {
            i = index[k];
            int yi = labeled_point_vector[i].y;
            auto& xi = labeled_point_vector[i].x;

            G = 0.0;
            for (int h = 0; h < T; h++) {
                G += wt_list[h].dot(xsp[i][h]) * mu_set[h];
            }
            // G = (w.dot(xi)) * yi + beta[i] + diag * alpha[i];
            G = G * yi + beta[i] + diag * alpha[i];

            PG = 0;
            if (EQUAL_VALUE(alpha[i], LB[i])) {
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
                alpha[i] = std::min(std::max(alpha[i] - G / QD[i], LB[i]), UB);
                diff = yi * (alpha[i] - alpha_old);
                for (int h = 0; h < T; h++) {
                    wt_list[h] += xsp[i][h] * diff;
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
    // problem_->w = w;
    // problem_->w_controlled = w_controlled;
    // auto& wt_list = problem_->wt_list;
    for (i = 0; i < T; i++) {
        w += mu_set[i] * wt_list[i];
    }
    double obj = 0;
    for (i = 0; i < l; i++) {
        obj += alpha[i];
    }
    obj = obj - diag * alpha.dot(alpha);
    for (i = 0; i < T; i++) {
        obj -= 0.5 * mu_set[i] * vector_operator::self_dot_product(wt_list[i]);
    }

    husky::LOG_I << "number of kernel: " + std::to_string(T);

    output_solu_->w = w;
    output_solu_->w_controlled = w;

    if (!cache) {
        delete[] index;
        delete[] QD;
    }

    return obj;
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

    dcd_svm(data_, model_, solu_, NULL, QD, index, true);
    double obj = solu_->obj;
    // compute gradient
    DenseVector<double> grad(T);
    for (i = 0; i < T; i++) {
        grad[i] = -0.5 * solu_->w.dot(solu_->wt_list[i]);
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
        husky::LOG_I << "[outer loop][old_obj]: " + std::to_string(old_obj);

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
                husky::LOG_I << "[inner loop][cost_max]: " + std::to_string(cost_max);
                for (i = 0; i < T; i++) {
                    tmp_mu_set[i] = new_mu_set[i] + step_max * desc[i];
                }
                // use descent direction to compute new objective
                // consider modifying input solution to speed up
                dcd_svm(data_, tmp_model, tmp_solu, solu_, QD, index, true);
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
                dcd_svm(data_, tmp_ls_model_1, tmp_ls_solu_1, solu_, QD, index, true);

                // half half
                for (i = 0; i < T; i++) {
                    tmp_ls_mu_set_2[i] = new_mu_set[i] + step_medl * desc[i];
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
            solu_->wt_list[i] = vector_operator::elem_wise_dot(solu_->w, dt_set[i]);
            grad[i] = -0.5 * solu_->w.dot(solu_->wt_list[i]);
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

void initialize(data* data_, model* model_) {
    // auto& train_set = husky::ObjListStore::create_objlist<ObjT>("train_sw");
    // auto& train_set_fw = husky::ObjListStore::create_objlist<ObjT>("train_fw");
    auto& train_set = husky::ObjListStore::create_objlist<ObjT>("train_set");
    auto& test_set = husky::ObjListStore::create_objlist<ObjT>("test_set");
    // data_->train_set = &train_set;
    data_->train_set = &train_set;
    data_->test_set = &test_set;
    // data_->train_set_fw = &train_set_fw;

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
    // customize_load_data(husky::Context::get_param("train_fw"), train_set_fw, format, false);


    // get model config parameters
    model_->B = std::stoi(husky::Context::get_param("B"));
    model_->C = std::stod(husky::Context::get_param("C"));
    model_->max_inn_iter = std::stoi(husky::Context::get_param("max_inn_iter"));
    model_->dt_set = std::vector<SparseVector<double>>();
    model_->mu_set = std::vector<double>();

    auto& train_set_data = train_set.get_data();
    // auto& train_set_fw_data = train_set_fw.get_data();

    // data_->label = DenseVector<double>(train_set_data.size(), 0.0);
    for (auto& labeled_point : train_set_data) {
        labeled_point.x.resize(n + 1);
        labeled_point.x.set(n, 1);
        // data_->label.set(labeled_point.id() - 1, labeled_point.y);
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

void evaluate(data* data_, model* model_, solu* solu_) {
    const auto& test_set_data = data_->test_set->get_data();
    const auto& w = solu_->w_controlled;
    double error = 0;
    double indicator;
    for (auto& labeled_point : test_set_data) {
        indicator = w.dot(labeled_point.x);
        indicator *= labeled_point.y;
        if (indicator <= 0) {
            error += 1;
        }
    }
    husky::LOG_I << "Classification accuracy on testing set with [B = " + std::to_string(model_->B) + "], " +
                        "[C = " + std::to_string(model_->C) + "], " +
                        "[max_inn_iter = " + std::to_string(model_->max_inn_iter) + "], " +
                        "[test set size = " + std::to_string(test_set_data.size()) + "]: " +
                        std::to_string(1.0 - static_cast<double>(error / test_set_data.size()));
}

// for testing of dcd_svm
void run_dcd_svm() {
    data* data_ = new data;
    model* model_ = new model;
    solu* solu_ = new solu;
    initialize(data_, model_);
    SparseVector<double> dt = find_most_violated(data_, model_);
    auto start = std::chrono::steady_clock::now();
    model_->dt_set.push_back(dt);
    model_->mu_set.push_back(1.0);
    dcd_svm(data_, model_, solu_);
    auto end = std::chrono::steady_clock::now();
    husky::LOG_I << "time elapsed: " + std::to_string(std::chrono::duration_cast<std::chrono::duration<float>>(end - start).count());
    evaluate(data_, model_, solu_);
}

// for testing of simple_mkl using designated kernel (B is not used)
void run_simple_mkl() {
    data* data_ = new data;
    model* model_ = new model;
    solu* solu_ = new solu;
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

    simpleMKL(data_, model_, solu_);
    vector_operator::show(model_->mu_set, "mu_set");
    husky::LOG_I << "trainning objective: " + std::to_string(solu_->obj);
    evaluate(data_, model_, solu_);
}

// for running FGM_DCD
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
            husky::LOG_I << "FGM converged";
            break;
        }
        model_->dt_set.push_back(dt);
        model_->mu_set.push_back(1.0);
        simpleMKL(data_, model_, solu_);
        mkl_obj = solu_->obj;
        obj_diff = fabs(mkl_obj - last_obj);
        husky::LOG_I << "[iteration " + std::to_string(iter) + "][mkl_obj " + std::to_string(mkl_obj) + "][obj_diff " + std::to_string(obj_diff) + "]";
        vector_operator::show(model_->mu_set, "mu_set");
        if (obj_diff < 0.001 * abs(last_obj)) {
            break;
        }
        if (model_->mu_set[iter] < 0.0001) {
            break;
        }
        iter++;
    }
    vector_operator::show(model_->mu_set, "mu_set");
    evaluate(data_, model_, solu_);
}

void init() {
    if (husky::Context::get_param("is_sparse") == "true") {
        job_runner();
    } else {
        husky::LOG_I << "Dense data format is not supported";
    }
}

int main(int argc, char** argv) {
    std::vector<std::string> args({"hdfs_namenode", "hdfs_namenode_port", "train", "test", "B", "C", "format", "is_sparse", "max_inn_iter"});
    if (husky::init_with_args(argc, argv, args)) {
        husky::run_job(init);
        return 0;
    }
    return 1;
}
