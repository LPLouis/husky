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
    max_iteration=200

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
#include <algorithm>
#include <cmath>
#include <string>

#include "boost/tokenizer.hpp"

#include "base/assert.hpp"
#include "core/executor.hpp"
#include "core/objlist.hpp"
#include "io/input/inputformat_store.hpp"
#include "lib/aggregator_factory.hpp"
#include "lib/ml/feature_label.hpp"
#include "lib/vector.hpp"

using husky::lib::Aggregator;
using husky::lib::AggregatorFactory;
using husky::lib::DenseVector;
using husky::lib::SparseVector;
using ObjT = husky::lib::ml::LabeledPointHObj<double, double, true>;

#define INF std::numeric_limits<double>::max()
#define Malloc(type, n) (type *) malloc( n * sizeof(type))

enum DataFormat { kLIBSVMFormat, kTSVFormat };

struct feature_node
{
    int index;
    float value;
};

struct sp_weight
{
    int index1;
    int index2;
    double value;

    bool operator < (sp_weight& b)
    {
        return abs(value) < abs(b.value);
    }

    bool operator <= (sp_weight& b)
    {
        return abs(value) <= abs(b.value);
    }

    bool operator > (sp_weight& b)
    {
        return abs(value) > abs(b.value);
    }

    sp_weight &operator = (sp_weight& b)
    {
        index1 = b.index1;
        index2 = b.index2;
        value = b.value;
        return *this;
    }
}

struct problem
{
    int l, n, n_kernel;
    int *y;

    husky::ObjList<ObjT>* train_set;
    husky::ObjList<ObjT>* test_set;
    struct feature_node ***xsp;
    struct feature_node ***xsp;
    int w_size;
    long int elements;
    int B;
};

struct parameter
{
    double eps;
    double C;
    int nr_weight;
    double* weight;
    int initial_type; // 0 for average initialization; 1 for trainning initialization
    int max_iteration;
    int t;
    int K;
    int B;  // number of features
    int fCRS;
    int z;
    int Ks;
}

struct model
{
    struct parameter param;
    int nr_feature;
    double *w;
    int *label;
    double bias;
    double *alpha;
    int l;

    double* sigma;
    int n_kernel;

    double* solution;   // alias for w
    int *count;
    int feature_pair;
    int B;
    double mkl_obj;
    double run_time;
}

struct parameter param;
struct problem prob;
struct model *model_;
struct feature_node *x_space;
double bias;

class FGM {
public:
    FGM();
    FGM(problem *& prob_, model *& model_, const parameter *svm_param_, int max_iteration_) 
    {
        max_iteration    = max_iteration_;
        param       = svm_param_;
        prob        = prob_;
        alpha       = model_->alpha;
        B           = svm_param_->B;
        // allocate memory for sub-features in advance
        elements    = prob_->l * (svm_param_->B + 1);
        svm_model   = model_;

        FGM_allocate();
    }

    void FGM_allocate() 
    {
        sub_x_space             = Malloc(feature_node *, max_iteration);
        prob->xsp               = Malloc(feature_node **, max_iteration);
        w_lin                   = Malloc(double, prob->n);
        QD                      = Malloc(float, prob->n);
        QD_count                = Malloc(int, prob->n);
        w2b_B                   = Malloc(sp_weight, 1 * param->B);
        w2b_temp                = Malloc(sp_weight, 3 * param->B);
        int i = 0;
        for (i = 0; i < max_iteration; i++)
        {
            prob->xsp[i] = Malloc(struct feature_node *, prob->l);
        }
    }
    ~FGM()
    {
        
        for (int i = 0; i < n_ITER-1; i++)
        {
            free(sub_x_space[i]);
            free(prob->xsp[i]);
        }
        
        free(sub_x_space);
        free(prob->xsp);
        free(w2b_temp);
        free(w2b_B);
        free(w_lin);
        free(QD);
        free(QD_count);
    }

    normalize()
    {
        for(int i = 0; i < prob->l; i++)
        {
            feature_node *xi = prob->x[i];
            {
                while (xi->index != -1)
                {
                    //QD_count[xi->index-1] += 1;
                    xi->value = xi->value / QD[xi->index-1];     
                    xi++;
                }
            }
        }
    }

    FGM_init()
    {
        int i = 0;
        for( i = 0; i < prob->l; i++)
        {
            alpha[i] = 1.0;
        }

        for(i = 0; i < param->B; i++)
        {
            // index1 is index of features locally [0, B)
            w2b_temp[i].index1 = i;
            // index2 is index of features globally [0, n)
            w2b_temp[i].index2 = -1;
            w2b_temp[i].value = 0;
        }
    }

    void heap_sort()

    most_violated(int iteration) 
    {
        int i;

        long int t_start = clock();
        long int t_finish;
        double runtime;
        double alphay;

        w_size_temp = param->B;
        for (i = 0; i < w_size_temp; i++)
        {
            w2b_temp[i].value = 0.0;
            w2b_temp[i].index1 = i;
            w2b_temp[i].index2 = -1;
        }
        for (i = 0; i < prob->n; i++)
        {
            w_lin[i] = 0.0;
        }
        const auto& train_set_data = prob->train_set->get_data();
        for (i = 0; i < prob-l; i++)
        {   
            if (alpha[i] != 0) {
                const auto& xi = train_set_data[i].x;
                alphay = alpha[i] * train_set_data[i].y;;
                for (auto it = xi.begin(); it != xi.end(); it++)
                {
                    w_lin[(*it).fea] += alphay * (*it).val;
                }
            }
        }

        for (i = 0; i < prob->n; i++)
        {
            if (fabs(QD[i]) > 0 && QD_count[i] > 0)
            {
                heap_sort(w2b_temp, fabs(w_lin[i]) / QD[i], w_size_temp, -1);
            }
        }

        sort_w2b(w2b_temp, w_size_temp);

        record_subfeature_sparse(w2b_temp, iteration);
    }

    int cutting_set_evolve()
    {
        most_violated(0);

    }

private:
    const parameter     *param;
    problem             *prob;
    double              *w_lin;     // for linear features
    float               *QD;        //for feature
    int                 *QD_count;
    int                 max_iteration;
    int                 n_ITER;
    int                 elements;
    feature_node        **sub_x_space;
    solution            solution;
    double              *alpha;
    int                 B;
    model               *svm_model;
    sp_weight           *w2b_B;
    sp_weight           *w2b_temp;
    int                 w_size_temp;   
};

void init() {
    if (husky::Context::get_param("is_sparse") == "true") {
        ;
    } else {
        husky::LOG_I << "Dense data format is not supported";
    }
}

int main(int argc, char** argv) {
    std::vector<std::string> args({"hdfs_namenode", "hdfs_namenode_port", "train", "test", "B", "C", "epsilon", "format", "is_sparse", "max_out_iter", "max_iteration", "max_inn_iter"});
    if (husky::init_with_args(argc, argv, args)) {
        husky::run_job(init);
        return 0;
    }
    return 1;
}
