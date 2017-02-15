#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include "boost/tokenizer.hpp"

#include "core/engine.hpp"
#include "core/utils.hpp"
#include "customize_data_loader.hpp"
#include "lib/ml/data_loader.hpp"
#include "lib/ml/feature_label.hpp"
#include "lib/ml/parameter.hpp"

using husky::lib::Aggregator;
using husky::lib::AggregatorFactory;
using husky::lib::DenseVector;
using husky::lib::SparseVector;
using ObjT = husky::lib::ml::LabeledPointHObj<double, double, true>;

void job_runner() {
    auto& train_set = husky::ObjListStore::create_objlist<ObjT>("train_set");
    auto& sample_set = husky::ObjListStore::create_objlist<ObjT>("sample_set");
    auto& feature_set = husky::ObjListStore::create_objlist<ObjT>("feature_set");

    int num_workers = husky::Context::get_num_workers();
    int tid = husky::Context::get_global_tid();

    std::string format_str = husky::Context::get_param("format");
    husky::lib::ml::DataFormat format;
    if (format_str == "libsvm") {
        format = husky::lib::ml::kLIBSVMFormat;
    } else if (format_str == "tsv") {
        format = husky::lib::ml::kTSVFormat;
    }

    // load data
    // int c = husky::lib::ml::load_data(husky::Context::get_param("input"), train_set, format);
    int n = customize_load_data(husky::Context::get_param("sample"), sample_set, format);
    int m = customize_load_data(husky::Context::get_param("feature"), feature_set, format, false);

    husky::LOG_I << "-----------------customize_load_data------------------";
    husky::LOG_I << std::to_string(tid) + ": local number of features: " +
                        std::to_string(feature_set.get_data().size());
    husky::LOG_I << std::to_string(tid) + ": local number of samples: " + std::to_string(sample_set.get_data().size());
    husky::LOG_I << std::to_string(tid) + ": total number of features: " + std::to_string(n);
    husky::LOG_I << std::to_string(tid) + ": total number of samples: " + std::to_string(m);

    /*
    husky::LOG_I << "---------------------load_data------------------------";
    husky::LOG_I << std::to_string(tid) + ": total number of features: " + std::to_string(c);
    husky::LOG_I << std::to_string(tid) + ": total number of samples: " + std::to_string(train_set.get_data().size());
    */
}

void init() {
    if (husky::Context::get_param("is_sparse") == "true") {
        job_runner();
    } else {
        husky::LOG_I << "Dense data format is not supported";
    }
}

int main(int argc, char** argv) {
    std::vector<std::string> args(
        {"hdfs_namenode", "hdfs_namenode_port", "input", "sample", "feature", "format", "is_sparse"});
    if (husky::init_with_args(argc, argv, args)) {
        husky::run_job(init);
        return 0;
    }
    return 1;
}
