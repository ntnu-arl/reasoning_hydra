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

  // Create a color map for the relationships
  // Todo (albertgassol1): Read the colors from a file if provided
  for (size_t i = 0; i < relationships_.size(); ++i) {
    relationship_colors_[relationships_[i]] = spark_dsg::Color::random(true);
  }
}

bool Reasoning::run(ReasoningOutput& reasoning_data,
                    const std::string& room_name) const {
  bool success = runReasoningScript(reasoning_data, room_name);
  clearReasoning();
  return success;
}

void Reasoning::clearReasoning() const {
  // Remove all files and folders in input and output folders
  auto clearDirectory = [](const std::string& directory_path_string) {
    std::filesystem::path directory_path(directory_path_string);
    if (!std::filesystem::exists(directory_path) ||
        !std::filesystem::is_directory(directory_path)) {
      std::cerr << "The specified path is not a valid directory.\n";
      return;
    }

    try {
      for (const auto& entry : std::filesystem::directory_iterator(directory_path)) {
        std::filesystem::remove_all(
            entry.path());  // Recursively remove files and directories
      }
    } catch (const std::filesystem::filesystem_error& e) {
      std::cerr << "Error: " << e.what() << '\n';
    }
  };

  clearDirectory(config_.input_folder);
  clearDirectory(config_.output_folder);
}

bool Reasoning::runReasoningScript(ReasoningOutput& reasoning_data,
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
  hydra::parseReasoningJson(output_path.string(), reasoning_data, room_name);
  return true;
}

std::string Reasoning::getRelationship(size_t index) const {
  if (index >= relationships_.size()) {
    LOG(ERROR) << "Relationship index out of bounds";
    return "";
  }
  return relationships_[index];
}

spark_dsg::Color Reasoning::getRelationshipColor(
    const std::string& relationship) const {
  auto it = relationship_colors_.find(relationship);
  if (it != relationship_colors_.end()) {
    return it->second;
  }
  return spark_dsg::Color();
}

spark_dsg::Color Reasoning::getRelationshipColor(size_t index) const {
  if (index >= relationships_.size()) {
    LOG(ERROR) << "Relationship index out of bounds";
    return spark_dsg::Color();
  }
  return getRelationshipColor(relationships_[index]);
}

}  // namespace hydra
