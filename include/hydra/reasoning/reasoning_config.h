#pragma once
#include <string>

namespace hydra {

struct ThreeDSSGConfig {
  std::string inference_script;
  std::string input_folder;
  std::string output_folder;
  double normal_estimation_radius;
  size_t num_sampling_points;
};

void declare_config(ThreeDSSGConfig& config);

}  // namespace hydra
