#include "hydra/utils/reasoning_io.h"

namespace hydra{

void parseReasoningJson(const std::string& filename, ReasoningJson& data, const std::string& room_name) {
    // Open and parse the JSON file
    std::ifstream file(filename);
    json jsonData;
    file >> jsonData;
    
    // Parse "nodes"
    if (jsonData.contains(room_name) && jsonData[room_name].contains("nodes")) {
        data.nodes = jsonData[room_name]["nodes"].get<std::vector<std::string>>();
    }

    // Parse "edges"
    if (jsonData[room_name].contains("edges")) {
        for (auto& [key, value] : jsonData[room_name]["edges"].items()) {
            int from = std::stoi(key);
            for (auto& [to_str, relation] : value.items()) {
                int to = std::stoi(to_str);
                data.edges[from][to] = relation.get<std::string>();
            }
        }
    }

    // Parse "node_probs"
    if (jsonData[room_name].contains("node_probs")) {
        data.node_probs = jsonData[room_name]["node_probs"].get<std::vector<std::vector<double>>>();
    }

    // Parse "edge_probs"
    if (jsonData[room_name].contains("edge_probs")) {
        data.edge_probs = jsonData[room_name]["edge_probs"].get<std::vector<std::vector<double>>>();
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
        if (line == "none") {
            continue;
        }
        lines.push_back(line);
    }

    file.close();
    return true;
}
    
} // namespace hydra