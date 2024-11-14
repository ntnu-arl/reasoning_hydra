#include "hydra/reasoning/reasoning.h"

namespace hydra {

Reasoning::Reasoning(const ReasoningConfig& config) : config_(config) {
  // Check if the relationships file exists
  CHECK(std::filesystem::exists(config_.method_inference_script_dir))
      << "Inference script directory does not exist: "
      << config_.method_inference_script_dir;

  // Read the relationships file
  CHECK(hydra::readLines(config_.relations_file, relationships_))
      << "Failed to read relationships file";
  // Sort the relationships
  std::sort(relationships_.begin(), relationships_.end());
}

bool Reasoning::run(
    const std::vector<NodeId>& object_ids,
    std::map<std::pair<NodeId, NodeId>, std::vector<std::string>>& reasoning_edges,
    const std::string& room_name) const {
  return runReasoningScript(object_ids, reasoning_edges, room_name);
}

bool Reasoning::runReasoningScript(
    const std::vector<NodeId>& object_ids,
    std::map<std::pair<NodeId, NodeId>, std::vector<std::string>>& reasoning_edges,
    const std::string& room_name) const {
  std::string command = "bash " + config_.inference_script + " " +
                        config_.input_folder + " " + config_.output_folder + " " +
                        config_.method_inference_script_dir;
  int ret = system(command.c_str());
  if (ret != 0) {
    LOG(ERROR) << "Failed to run reasoning script";
    return false;
  }

  // Read the reasoning output json
  std::filesystem::path output_path =
      std::filesystem::path(config_.output_folder) / (room_name + ".json");
  if (!std::filesystem::exists(output_path)) {
    LOG(ERROR) << "Reasoning output file does not exist: " << output_path.string();
    return false;
  }

  // Parse the reasoning output json
  ReasoningJson reasoning_data;
  hydra::parseReasoningJson(output_path.string(), reasoning_data, room_name);

  // Iterate through reasoning edges probabilities
  for (size_t i = 0; i < reasoning_data.edge_probs.size(); ++i) {
    size_t from = static_cast<size_t>(std::floor(i / (object_ids.size() - 1)));
    size_t to = i % (object_ids.size() - 1);
    if (to >= from) {
      to += 1;
    }
    for (size_t j = 0; j < reasoning_data.edge_probs[i].size(); ++j) {
      if (reasoning_data.edge_probs[i][j] > config_.edge_prob_threshold) {
        reasoning_edges[{object_ids[from], object_ids[to]}].push_back(
            relationships_[j]);
      }
    }
  }
  return true;
}

}  // namespace hydra
