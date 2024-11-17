#ifndef INCLUDE_HYDRA_UTILS_REASONING_IO_H_
#define INCLUDE_HYDRA_UTILS_REASONING_IO_H_
#endif  // INCLUDE_HYDRA_UTILS_REASONING_IO_H_

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <vector>

namespace hydra {

using json = nlohmann::json;
struct ReasoningOutput {
  std::vector<std::string> nodes;
  std::unordered_map<int, std::unordered_map<int, std::string>> edges;
  std::vector<std::vector<double>> node_probs;
  std::vector<std::vector<double>> edge_probs;
  std::vector<std::vector<double>> feature_vectors;
};

void parseReasoningJson(const std::string& filename,
                        ReasoningOutput& data,
                        const std::string& room_name);

bool readLines(const std::string& filename, std::vector<std::string>& lines);
}  // namespace hydra
