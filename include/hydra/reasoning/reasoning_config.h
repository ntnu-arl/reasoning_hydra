#pragma once
#include <string>

namespace hydra {

struct ReasoningConfig {
  std::string inference_script;
  std::string input_folder;
  std::string output_folder;
  std::string method_inference_script_dir;
  std::string relations_file;
  bool save_objects;
};

struct ThreeDSSGConfig {
  ReasoningConfig reasoning;
  double normal_estimation_radius;
  size_t num_sampling_points;
  double edge_prob_threshold;
  float edges_max_radius;
};

void declare_config(ReasoningConfig& config);
void declare_config(ThreeDSSGConfig& config);

}  // namespace hydra
