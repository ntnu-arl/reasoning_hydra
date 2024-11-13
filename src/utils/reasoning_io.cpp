#include "hydra/utils/reasoning_io.h"

namespace hydra{

void parseReasoningJson(const std::string& filename, ReasoningJson& data) {
    // Open and parse the JSON file
    std::ifstream file(filename);
    json jsonData;
    file >> jsonData;
    
    // Parse "nodes"
    if (jsonData.contains("R(0)") && jsonData["R(0)"].contains("nodes")) {
        data.nodes = jsonData["R(0)"]["nodes"].get<std::vector<std::string>>();
    }

    // Parse "edges"
    if (jsonData["R(0)"].contains("edges")) {
        for (auto& [key, value] : jsonData["R(0)"]["edges"].items()) {
            int from = std::stoi(key);
            for (auto& [to_str, relation] : value.items()) {
                int to = std::stoi(to_str);
                data.edges[from][to] = relation.get<std::string>();
            }
        }
    }

    // Parse "node_probs"
    if (jsonData["R(0)"].contains("node_probs")) {
        data.node_probs = jsonData["R(0)"]["node_probs"].get<std::vector<std::vector<double>>>();
    }

    // Parse "edge_probs"
    if (jsonData["R(0)"].contains("edge_probs")) {
        data.edge_probs = jsonData["R(0)"]["edge_probs"].get<std::vector<std::vector<double>>>();
    }

    file.close();
}

bool readLines(const std::string& filename, std::vector<std::string>& lines) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Failed to open the file: " << filename << std::endl;
        return false;
    }

    std::string line;
    while (std::getline(file, line)) {
        lines.push_back(line);
    }

    file.close();
    return true;
}
    
} // namespace hydra