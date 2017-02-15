/***
    Assumptions:
    input file fits in memory of one single machine
    input file always gets assigned to one single machine
    Input file can fit in one machine so that the order of the instances presented to data_proc_sample.cpp and data_proc_feature.cpp is the same

    input
    <sample_id> <label> <fea>:<val>
    output
    <fea_id> <sample>:<val>
***/
#include <algorithm>
#include <cmath>
#include <string>

#include <fstream>
#include <unordered_map>
#include "boost/tokenizer.hpp"

#include "core/executor.hpp"
#include "core/objlist.hpp"
#include "core/utils.hpp"
#include "io/input/inputformat_store.hpp"
#include "lib/aggregator_factory.hpp"
#include "lib/ml/data_loader.hpp"
#include "lib/ml/feature_label.hpp"
#include "lib/vector.hpp"

using husky::lib::Aggregator;
using husky::lib::AggregatorFactory;
using husky::lib::DenseVector;
using husky::lib::SparseVector;
using ObjT = husky::lib::ml::LabeledPointHObj<double, double, true>;

void init() {
    auto& infmt = husky::io::InputFormatStore::create_line_inputformat();
    infmt.set_input(husky::Context::get_param("input"));
    auto& dataset = husky::ObjListStore::create_objlist<ObjT>("dataset");

    std::string format_str = husky::Context::get_param("format");
    husky::lib::ml::DataFormat format;
    if (format_str == "libsvm") {
        format = husky::lib::ml::kLIBSVMFormat;
    } else if (format_str == "tsv") {
        format = husky::lib::ml::kTSVFormat;
    }
    int n = husky::lib::ml::load_data(husky::Context::get_param("input"), dataset, format);

    auto& data = dataset.get_data();
    int m = data.size();

    std::unordered_map<size_t, std::string> fea_map;
    for (int i = 1; i <= m; i++) {
        auto& x = data[i - 1].x;
        for (auto it = x.begin(); it != x.end(); it++) {
            // data loader set fea_num = idx - 1
            int fea_num = (*it).fea + 1;
            double fea_val = (*it).val;
            fea_map[fea_num] += std::to_string(i) + ":" + std::to_string(fea_val) + " ";
        }
    }

    std::vector<std::string> output_vec(n);
    for (auto it = fea_map.begin(); it != fea_map.end(); it++) {
        output_vec[it->first - 1] = it->second;
    }

    std::string output_filename = husky::Context::get_param("output") + "_pp_feature_num";
    std::ofstream outfile;
    outfile.open(output_filename);
    assert(outfile.is_open());
    for (int i = 0; i < n; i++) {
        outfile << std::to_string(i + 1) << " " << output_vec[i] << "\n";
    }
    outfile << std::to_string(n + 1);
    for (int i = 0; i < m; i++) {
        outfile << " " << std::to_string(i + 1) << ":" << std::to_string(1.0);
    }
    outfile.close();
}

int main(int argc, char** argv) {
    std::vector<std::string> args({"hdfs_namenode", "hdfs_namenode_port", "input", "output", "format"});
    if (husky::init_with_args(argc, argv, args)) {
        husky::run_job(init);
        return 0;
    }
    return 1;
}
