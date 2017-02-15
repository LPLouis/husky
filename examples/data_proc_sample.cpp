/***
    Assumptions:
    input file fits in memory of one single machine
    input file always gets assigned to one single machine

    input
    <label> <fea>:<val>
    output
    <sample_id> <label> <fea>:<val>
***/
#include <algorithm>
#include <cmath>
#include <string>

#include <fstream>
#include "boost/tokenizer.hpp"

#include "core/executor.hpp"
#include "core/objlist.hpp"
#include "core/utils.hpp"
#include "io/input/inputformat_store.hpp"
#include "lib/aggregator_factory.hpp"
#include "lib/ml/feature_label.hpp"
#include "lib/vector.hpp"

void init() {
    auto& infmt = husky::io::InputFormatStore::create_line_inputformat();
    infmt.set_input(husky::Context::get_param("input"));
    std::vector<std::string> buf;
    auto parser = [&](boost::string_ref chunk) { buf.push_back(chunk.to_string()); };
    husky::load(infmt, parser);
    std::string output_filename = husky::Context::get_param("output") + "_pp_sample_num";
    std::ofstream outfile;
    outfile.open(output_filename);
    assert(outfile.is_open());
    for (int i = 0; i < buf.size(); i++) {
        outfile << std::to_string(i + 1) << " " << buf[i] << "\n";
    }
    outfile.close();
}

int main(int argc, char** argv) {
    std::vector<std::string> args({"hdfs_namenode", "hdfs_namenode_port", "input", "output"});
    if (husky::init_with_args(argc, argv, args)) {
        husky::run_job(init);
        return 0;
    }
    return 1;
}
