#include "hydra/utils/reasoning_io.h"

namespace hydra {

void parseReasoningJson(const std::string& filename,
                        ReasoningOutput& data,
                        const std::string& room_name) {
  // Open and parse the JSON file
  std::ifstream file(filename);
  json jsonData;
  file >> jsonData;

  // Parse "nodes"
  if (jsonData.contains(room_name) && jsonData[room_name].contains("nodes")) {
    data.nodes = jsonData[room_name]["nodes"].get<Nodes>();
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
    data.node_probs = jsonData[room_name]["node_probs"].get<NodeProbs>();
  }

  // Parse "edge_probs"
  if (jsonData[room_name].contains("edge_probs")) {
    data.edge_probs = jsonData[room_name]["edge_probs"].get<EdgeProbs>();
  }

  // Parse "features"
  if (jsonData[room_name].contains("features")) {
    const auto& features_vector =
        jsonData[room_name]["features"]
            .get<std::vector<std::vector<std::vector<float>>>>();
    if (features_vector.empty()) {
      return;
    }
    const size_t num_features = features_vector.size();
    const size_t num_rows = features_vector[0].size();
    const size_t num_cols = features_vector[0][0].size();
    data.features =
        std::vector<Eigen::MatrixXf>(num_features, Eigen::MatrixXf(num_rows, num_cols));
    for (size_t i = 0; i < num_features; ++i) {
      data.features[i] = Eigen::Map<const Eigen::MatrixXf>(
          features_vector[0][i].data(), num_rows, num_cols);
    }
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

}  // namespace hydra
