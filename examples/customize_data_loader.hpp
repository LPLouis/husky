/***
    data format: kLIBSVMFormat
    pp_sample: sample_id label <fea>:<fea_val>
    pp_feature: feature_id <sample>:<sample_val>

    if pp_sample = true
        customize_load_data accepts pp_sample format
    else
        customize_load_data accepts pp_feature format
        for pp_feature, label is dummy 0
***/
#pragma once

#include <algorithm>
#include <cmath>
#include <string>

#include "boost/tokenizer.hpp"

#include "core/executor.hpp"
#include "core/objlist.hpp"
#include "core/utils.hpp"
#include "io/input/inputformat_store.hpp"
#include "lib/aggregator_factory.hpp"
#include "lib/ml/data_loader.hpp"
#include "lib/ml/feature_label.hpp"
#include "lib/vector.hpp"

// load data without knowing the number of features
template <typename FeatureT, typename LabelT, bool is_sparse>
int customize_load_data(std::string url,
                        husky::ObjList<husky::lib::ml::LabeledPointHObj<FeatureT, LabelT, is_sparse>>& data,
                        husky::lib::ml::DataFormat format, bool pp_sample = true) {
    using DataObj = husky::lib::ml::LabeledPointHObj<FeatureT, LabelT, is_sparse>;
    auto& infmt = husky::io::InputFormatStore::create_line_inputformat();
    infmt.set_input(url);

    husky::lib::Aggregator<int> num_features_agg(0, [](int& a, const int& b) { a = std::max(a, b); });
    auto& ac = husky::lib::AggregatorFactory::get_channel();

    std::function<void(boost::string_ref)> parser;
    if (pp_sample) {
        parser = [&](boost::string_ref chunk) {
            if (chunk.empty())
                return;
            boost::char_separator<char> sep(" \t");
            boost::tokenizer<boost::char_separator<char>> tok(chunk, sep);

            // get the largest index of features for this record
            int sz = 0;
            if (!is_sparse) {
                auto last_colon = chunk.find_last_of(':');
                if (last_colon != -1) {
                    auto last_space = chunk.substr(0, last_colon).find_last_of(' ');
                    sz = std::stoi(chunk.substr(last_space + 1, last_colon).data());
                }
                ASSERT_MSG(sz > 0, "The input file does not conform to LibSVM format.");
            }
            DataObj this_obj(sz);  // create a data object

            int flag = 0;
            for (auto& w : tok) {
                if (flag > 1) {
                    boost::char_separator<char> sep2(":");
                    boost::tokenizer<boost::char_separator<char>> tok2(w, sep2);
                    auto it = tok2.begin();
                    int idx = std::stoi(*it++);
                    double val = std::stod(*it++);
                    num_features_agg.update(idx);
                    this_obj.x.set(idx - 1, val);
                } else {
                    if (flag == 0) {
                        this_obj.key = std::stoi(w);
                        flag++;
                    } else {
                        this_obj.y = std::stod(w);
                        flag++;
                    }
                }
            }
            data.add_object(this_obj);
        };
    } else {
        parser = [&](boost::string_ref chunk) {
            if (chunk.empty())
                return;
            boost::char_separator<char> sep(" \t");
            boost::tokenizer<boost::char_separator<char>> tok(chunk, sep);

            // get the largest index of features for this record
            int sz = 0;
            if (!is_sparse) {
                auto last_colon = chunk.find_last_of(':');
                if (last_colon != -1) {
                    auto last_space = chunk.substr(0, last_colon).find_last_of(' ');
                    sz = std::stoi(chunk.substr(last_space + 1, last_colon).data());
                }
                ASSERT_MSG(sz > 0, "The input file does not conform to LibSVM format.");
            }
            DataObj this_obj(sz);  // create a data object

            int flag = 0;
            for (auto& w : tok) {
                if (flag > 0) {
                    boost::char_separator<char> sep2(":");
                    boost::tokenizer<boost::char_separator<char>> tok2(w, sep2);
                    auto it = tok2.begin();
                    int idx = std::stoi(*it++);
                    double val = std::stod(*it++);
                    num_features_agg.update(idx);
                    this_obj.x.set(idx - 1, val);
                } else {
                    // dummy label
                    this_obj.y = 0.0;
                    this_obj.key = std::stoi(w);
                    flag++;
                }
            }
            data.add_object(this_obj);
        };
    }

    husky::load(infmt, {&ac}, parser);
    int num_features = num_features_agg.get_value();
    list_execute(data, [&](DataObj& this_obj) {
        if (this_obj.x.get_feature_num() != num_features) {
            this_obj.x.resize(num_features);
        }
    });
    return num_features;
}
