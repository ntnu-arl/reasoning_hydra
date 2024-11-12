#pragma once
#include <string>

namespace hydra {

struct ThreeDSSGConfig {
  std::string inference_script;
  std::string input_folder;
  std::string output_folder;
};

void declare_config(ThreeDSSGConfig& config);

}  // namespace hydra
