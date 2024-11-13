#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <nlohmann/json.hpp>

namespace hydra{

using json = nlohmann::json;
struct ReasoningJson {
    std::vector<std::string> nodes;
    std::unordered_map<int, std::unordered_map<int, std::string>> edges;
    std::vector<std::vector<double>> node_probs;
    std::vector<std::vector<double>> edge_probs;
};

void parseReasoningJson(const std::string& filename, ReasoningJson& data);  

bool readLines(const std::string& filename, std::vector<std::string>& lines);
} // namespace hydra