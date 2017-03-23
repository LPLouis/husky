/***
    Thoughts for improvement after algorithm is done:
    1. restructure problem to contain only specification of problem and use another structure perhaps called 
    solution to store the solutions such as alpha, w, wt_list and w_controlled.
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

#define EQUAL_VALUE(a, b) (a - b < 1.0e-12) && (a - b > -1.0e-12)
#define NOT_EQUAL(a, b) (a - b > 1.0e-12) || (a - b < -1.0e-12)
#define EPS 1.0e-12
#define MU_EPS 1.0e-2
#define ALPHA_EPS 1.0e-2
#define INF std::numeric_limits<double>::max()

class problem {
    public:
        int B;
        double C;
        int n;          // number of features (appended 1 included)
        int l;         
        int max_inn_iter;
        DenseVector<double> label;
        DenseVector<double> alpha;
        std::vector<double> mu_set;
        std::vector<SparseVector<double>> dt_set;
        // note that w is uncontrolled
        DenseVector<double> w;
        // sparse d + sparse d equals to dense d, so may be don't store \sum_mu_dt may be better?
        // wt in wt_list is controlled but unweighted
        std::vector<SparseVector<double>> wt_list;
        // w_controlled is weighted
        DenseVector<double> w_controlled;
        husky::ObjList<ObjT>* train_set_sw;
        husky::ObjList<ObjT>* train_set_fw;
        husky::ObjList<ObjT>* test_set;
};

void solve_lp(const std::vector<double>& mu_scores, problem* problem_) {
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
    /*** DB OR LB ? ***/
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
        problem_->mu_set[i] =  glp_get_col_prim(lp, i + 1);
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
        husky::LOG_I << vec.get_feature_num();
    }

    static void show(const SparseVector<double>& vec, std::string message_head) {
        std::string ret = message_head + ": ";
        for (auto it = vec.begin(); it != vec.end(); it++) {
            ret += std::to_string((*it).fea) + ":" + std::to_string((*it).val) + " ";
        }
        husky::LOG_I << ret << "\n";
        husky::LOG_I << vec.get_feature_num();
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

    static int arg_min_dm_over_Dm(const std::vector<double>& mu_set, const DenseVector<double>& descent_direction) {
        assert(mu_set.size() != 0 && mu_set.size() == descent_direction.get_feature_num() && "arg_min_dm_over_Dm: error");
        int index = -1;
        double min = std::numeric_limits<int>::max();
        for (int i = 0; i < mu_set.size(); i++) {
            if (descent_direction[i] < 0) {
                double temp = -1 * mu_set[i] / descent_direction[i];
                if (min > temp) {
                    index = i;
                    min = temp;
                }
            }
        }
        return index;
    }

    static bool sum_equal_to(const std::vector<double>& v, double val) {
        if (v.size() == 0) {
            return false;
        }
        double sum = 0.0;
        for (int i = 0; i < v.size(); i++) {
            sum += v[i];
        }

        if (NOT_EQUAL(sum, val)) {
            husky::LOG_I << "sum_equal_to: " << std::to_string(sum);
            return false;
        }
        return true;
    }

    static bool sum_equal_to(const DenseVector<double>& v, double val) {
        if (v.get_feature_num() == 0) {
            return false;
        }
        double sum = 0.0;
        for (int i = 0; i < v.get_feature_num(); i++) {
            sum += v[i];
        }

        if (NOT_EQUAL(sum, val)) {
            husky::LOG_I << "sum_equal_to: " << std::to_string(sum);
            return false;
        }
        return true;
    }

};

SparseVector<double> find_most_violated(problem* problem_) {
    int B = problem_->B;
    auto& train_set_fw = problem_->train_set_fw;
    const DenseVector<double>& alpha = problem_->alpha;
    DenseVector<double> cache(problem_->l);
    DenseVector<double> label = problem_->label;
    // DenseVector<double> fea_score(problem_->n);
    std::vector<std::pair<int, double>> fea_score;


    for (int i = 0; i < problem_->l; i++) {
        cache[i] = label[i] * alpha[i];
    }
    for (auto& labeled_point_vector : train_set_fw->get_data()) {
        fea_score.push_back(std::make_pair(labeled_point_vector.id() - 1, labeled_point_vector.x.dot(cache)));
    }
    fea_score.push_back(std::make_pair(problem_->n - 1, cache.dot(DenseVector<double>(problem_->l, 1.0))));
    std::sort(fea_score.begin(), fea_score.end(), [](auto& left, auto& right) {
        return left.second > right.second;
    });

    // for (auto& labeled_point_vector : train_set_fw->get_data()) {
    //     fea_score[labeled_point_vector.id() - 1] = labeled_point_vector.x.dot(cache);
    // }
    // // train_set_fw does not store the last feature i.e., 1, because it is implicit
    // fea_score[problem_->n - 1] = cache.dot(DenseVector<double>(problem_->l, 1.0));

    // SparseVector<double> control_variable(problem_->n);
    // std::vector<std::pair<int, double>> temp_vector;
    // for (auto it = fea_score.begin_feaval(); it != fea_score.end_feaval(); it++) {
    //     int fea = (*it).fea;
    //     double val = (*it).val;
    //     temp_vector.push_back(std::make_pair(fea, val));
    // }
    // std::sort(temp_vector.begin(), temp_vector.end(), [](auto& left, auto& right) {
    //         return left.second > right.second;
    // });
    // for (int i = 0; i < B; i++) {
    //     int fea = temp_vector[i].first;
    //     double val = temp_vector[i].second;
    //     control_variable.set(fea, 1.0);
    // }
    SparseVector<double> control_variable(problem_->n);
    for (int i = 0; i < B; i++) {
        int fea = fea_score[i].first;
        double val = fea_score[i].second;
        control_variable.set(fea, 1.0);
    }
    control_variable.sort_asc();
    return control_variable;
}

template <bool is_sparse = true>
double dcd_svm(problem* problem_, double* QD = NULL, int* index = NULL, bool cache = false) {
    // observation
    // setting B = 124, C = 1.0 and max_inn_iter = 200
    // If I directly use just w, then 84.9% acc is achieved in 14 seconds
    // If I use controlled w, then 84.9% acc is achieved in 42 seconds
    // If I use controlled w with B = 5, then 92% acc is achieved in 13 seconds
    // Declaration and Initialization
    int l = problem_->l;
    int n = problem_->n;
    const auto& labeled_point_vector = problem_->train_set_sw->get_data();
    const auto& mu_set = problem_->mu_set;
    const auto& dt_set = problem_->dt_set;

    // combine all control variables to a single control variable (Assumption: coef of dt sum to 1)
    const DenseVector<double> control_variable = vector_operator::sum_mu_dt(mu_set, dt_set);
    DenseVector<double>& alpha = problem_->alpha;
    double diag = 0.5 / problem_->C;
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

    while (iter < problem_->max_inn_iter) {
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
    problem_->w = w;
    problem_->w_controlled = w_controlled;
    auto& wt_list = problem_->wt_list;
    for (i = 0; i < wt_list.size(); i++) {
        // ideally this should be quick because we only need to iterate over the sparse entries
        // might want to compute wt_list instead of w_controlled when B is very small
        wt_list[i] = vector_operator::elem_wise_dot(w, problem_->dt_set[i]);
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

    return obj;
}

double line_search(problem* problem_, const double max_step_size, const DenseVector<double>& descent_direction, const double last_obj, const DenseVector<double>& gradient, const std::vector<double>& mu_set, double *QD, int *index) {
    // refer to page 9
    if (max_step_size == 0.0) {
        return 0.0;
    }
    const int T = mu_set.size();
    const double incr = 1.2;
    const double desc = 0.5;
    const double rls = 0.01;
    double step_size = max_step_size;
    for (int i = 0; i < T; i++) {
        problem_->mu_set[i] = mu_set[i] + step_size * descent_direction[i];
    }
    double obj = dcd_svm(problem_, QD, index, true);
    int iter = 0;
    // for now assume gradient and last_obj won't be changed
    while (obj > last_obj + rls * gradient.dot(step_size * descent_direction)) {
        husky::LOG_I << "line search iteration: " + std::to_string(++iter);
        husky::LOG_I << "line search new objective: " + std::to_string(obj);
        step_size *= desc;
        for (int i = 0; i < T; i++) {
            problem_->mu_set[i] = mu_set[i] + step_size * descent_direction[i];
        }
        obj = dcd_svm(problem_, QD, index, true);
    }
    return step_size;
}

bool test_MKL_convergence(problem* problem_) {

}

void simpleMKL(problem* problem_) {
    assert(problem_->mu_set.size() == problem_->wt_list.size() && problem_->mu_set.size() == problem_->dt_set.size() && "size of mu_set, wt_list and dt_set do not agree");
    
    // initialize mu_set
    std::vector<double>& mu_set = problem_->mu_set;
    const auto& dt_set = problem_->dt_set;
    const int T = mu_set.size();
    double init = 1.0 / T;
    for (int i = 0; i < T; i++) {
        mu_set[i] = init;
    }

    // cache QD and index
    const auto& labeled_point_vector = problem_->train_set_sw->get_data();
    const DenseVector<double> control_variable = vector_operator::sum_mu_dt(mu_set, dt_set);
    int i, l;
    l = problem_->l;
    double diag = 0.5 / problem_->C;
    double* QD = new double[l];
    int* index = new int[l];
    for (i = 0; i < l; i++) {
        QD[i] = vector_operator::self_dot_elem_wise_dot(labeled_point_vector[i].x, mu_set, dt_set) + diag;
        index[i] = i;
    }

    // local variable used in simpleMKL
    const auto& w = problem_->w;
    const auto& wt_list = problem_->wt_list;
    DenseVector<double> gradient(T, 0.0);
    DenseVector<double> reduced_gradient(T, 0.0);
    std::vector<double> last_mu_set = problem_->mu_set;
    DenseVector<double> last_descent_direction(T, 0.0);
    DenseVector<double> descent_direction(T, 0.0);

    // mu is the index of the largest element in gradient
    int mu, v;
    double last_obj, obj, step_size, max_step_size;
    // while stopping criteria not met
    while(1) {
        last_obj = dcd_svm(problem_, QD, index, true);
        husky::LOG_I << "old Objective: " << std::to_string(last_obj);
        for (i = 0; i < T; i++) {
            gradient[i] = -1 * 0.5 * w.dot(wt_list[i]);
            husky::LOG_I << "gradient: " << std::to_string(gradient[i]);
        }
        mu = vector_operator::find_max_index(mu_set);

        // find descent direction
        for (i = 0; i < T; i++) {
            double grad = gradient[i] - gradient[mu];
            if (mu_set[i] == 0.0 && i != mu) {
                // if mu == 0.0 and grad >= 0.0 (want to move left to descent, truncate to 0)
                if (grad >= 0.0) {
                    descent_direction[i] = 0.0;
                // if mu == 0.0 and grad < 0.0 (want to move right to descent, keep)
                } else {
                    descent_direction[i] = -1 * grad;
                    descent_direction[mu] += grad;
                }
                reduced_gradient[i] = grad;
                reduced_gradient[mu] -= -1 * grad; 
            } else if(mu_set[i] > 0.0 && i != mu) {
                descent_direction[i] = -1 * grad;
                descent_direction[mu] += grad;
                reduced_gradient[i] = grad;
                reduced_gradient[mu] -= -1 * grad; 
            }
        }
        assert(vector_operator::sum_equal_to(descent_direction, 0.0) && "descent_direction does not sum to 0");
        assert(vector_operator::sum_equal_to(mu_set, 1.0) && "sum of mu_set not equal to 1");
        obj = 0;
        max_step_size = 0.0;
        // used for line search and update of mu_set after descent direction has been found
        const auto backup_obj = last_obj;
        const auto backup_gradient = gradient;
        const auto backup_mu_set = problem_->mu_set;
        // descent direction update
        while (obj < last_obj) {
            last_mu_set = mu_set;
            last_descent_direction = descent_direction;
            if ((v = vector_operator::arg_min_dm_over_Dm(last_mu_set, last_descent_direction) == -1)) {
                husky::LOG_I << "v equal to -1, no descent direction can be found";
                break;
            }
            // why is max_step_size smaller than 0???????????????????????????????????????
            max_step_size = -1 * last_mu_set[v] / last_descent_direction[v];
            husky::LOG_I << "max_step_size: " + std::to_string(max_step_size);
            for (int i = 0; i < T; i++) {
                // mu_set is a reference to the mu_set inside problem_
                mu_set[i] += max_step_size * last_descent_direction[i];
            }
            descent_direction[mu] = last_descent_direction[mu] + last_descent_direction[v];
            descent_direction[v] = 0;
            assert(vector_operator::sum_equal_to(mu_set, 1.0) && "sum of mu_set not equal to 1");
            obj = dcd_svm(problem_, QD, index, true);
            vector_operator::show(mu_set, "mu_set");
            husky::LOG_I << "new Objective: " << std::to_string(obj);
        }
        // if max_step_size == 0.0 => v = -1 => no descent direction can be found => already at optimum => go to test if global optimum is reached
        // note when this return, this may be the case that we have changed mu_set => continue searching
        // else we have not been able to change mu_set => algorithm fails to converge
        if (max_step_size == 0.0) {
            continue;
        }
        // gradient or reduced gradient? need discussion
        step_size = line_search(problem_, max_step_size, descent_direction, backup_obj, reduced_gradient, backup_mu_set, QD, index);
        for (i = 0; i < T; i++) {
            mu_set[i] = backup_mu_set[i] + step_size * descent_direction[i];
        }
        break;
    }
}

void initialize(problem* problem_) {
    auto& train_set_sw = husky::ObjListStore::create_objlist<ObjT>("train_sw");
    auto& train_set_fw = husky::ObjListStore::create_objlist<ObjT>("train_fw");
    auto& test_set = husky::ObjListStore::create_objlist<ObjT>("test_set");
    problem_->train_set_sw = &train_set_sw;
    problem_->test_set = &test_set;
    problem_->train_set_fw = &train_set_fw;

    auto format_str = husky::Context::get_param("format");
    husky::lib::ml::DataFormat format;
    if (format_str == "libsvm") {
        format = husky::lib::ml::kLIBSVMFormat;
    } else if (format_str == "tsv") {
        format = husky::lib::ml::kTSVFormat;
    }

    // load data
    int n = customize_load_data(husky::Context::get_param("train_sw"), train_set_sw, format);
    n = std::max(n, husky::lib::ml::load_data(husky::Context::get_param("test"), test_set, format));
    customize_load_data(husky::Context::get_param("train_fw"), train_set_fw, format, false);


    // get model config parameters
    problem_->B = std::stoi(husky::Context::get_param("B"));
    problem_->C = std::stod(husky::Context::get_param("C"));
    problem_->max_inn_iter = std::stoi(husky::Context::get_param("max_inn_iter"));

    auto& train_set_sw_data = train_set_sw.get_data();
    auto& train_set_fw_data = train_set_fw.get_data();

    problem_->label = DenseVector<double>(train_set_sw_data.size(), 0.0);
    for (auto& labeled_point : train_set_sw_data) {
        labeled_point.x.resize(n + 1);
        labeled_point.x.set(n, 1);
        problem_->label.set(labeled_point.id() - 1, labeled_point.y);
    }
    for (auto& labeled_point : test_set.get_data()) {
        labeled_point.x.resize(n + 1);
        labeled_point.x.set(n, 1);
    }

    n += 1;
    int l = train_set_sw_data.size();
    problem_->n = n;
    problem_->l = l;

    husky::LOG_I << "number of samples: " + std::to_string(l);
    husky::LOG_I << "number of features: " + std::to_string(n);
}

void evaluate(problem* problem_) {
    const auto& test_set_data = problem_->test_set->get_data();
    const auto& w = problem_->w_controlled;

    double error = 0;
    double indicator;
    for (auto& labeled_point : test_set_data) {
        indicator = w.dot(labeled_point.x);
        indicator *= labeled_point.y;
        if (indicator < 0) {
            error += 1;
        }
    }
    husky::LOG_I << "Classification accuracy on testing set with [B = " + std::to_string(problem_->B) + "], " +
                        "[C = " + std::to_string(problem_->C) + "], " +
                        "[max_inn_iter = " + std::to_string(problem_->max_inn_iter) + "], " +
                        "[test set size = " + std::to_string(test_set_data.size()) + "]: " +
                        std::to_string(1.0 - static_cast<double>(error / test_set_data.size()));
}

void job_runner() {
    problem* problem_ = new problem;
    initialize(problem_);
    problem_->alpha = DenseVector<double>(problem_->l, 1.0 / problem_->l);

    int out_iter = 0;
    while(1) {
        SparseVector<double> dt = find_most_violated(problem_);
        // auto start = std::chrono::steady_clock::now();
        // auto end = std::chrono::steady_clock::now();
        // husky::LOG_I << "time elapsed: " + std::to_string(std::chrono::duration_cast<std::chrono::duration<float>>(end - start).count());
        // evaluate(problem_);

        if (vector_operator::elem_at(dt, problem_->dt_set)) {
            husky::LOG_I << "FGM converged";
            evaluate(problem_);
            break;
        }
        problem_->dt_set.push_back(dt);
        problem_->wt_list.push_back(SparseVector<double>(problem_->n));
        problem_->mu_set.push_back(1.0); // dummy 1.0
        if (problem_->dt_set.size() == 1) {
            husky::LOG_I << "dt_set size equal to 1, directly call dcd_svm";
            dcd_svm(problem_);
        } else {
            husky::LOG_I << "dt_set size not equal to 1, call simpleMKL";
            simpleMKL(problem_);
        }
        vector_operator::show(problem_->mu_set, "mu_set");
    }

    delete problem_;
}

void init() {
    if (husky::Context::get_param("is_sparse") == "true") {
        job_runner();
    } else {
        husky::LOG_I << "Dense data format is not supported";
    }
}

int main(int argc, char** argv) {
    std::vector<std::string> args({"hdfs_namenode", "hdfs_namenode_port", "train_sw", "train_fw", "train", "test", "B", "C", "format", "is_sparse", "max_inn_iter"});
    if (husky::init_with_args(argc, argv, args)) {
        husky::run_job(init);
        return 0;
    }
    return 1;
}
