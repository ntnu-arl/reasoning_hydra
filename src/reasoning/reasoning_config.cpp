#include "hydra/reasoning/reasoning_config.h"

#include <config_utilities/config.h>

namespace hydra {

void declare_config(ReasoningConfig& conf) {
  using namespace config;
  name("ReasoningConfig");
  field(conf.inference_script, "inference_script");
  field(conf.input_folder, "input_folder");
  field(conf.output_folder, "output_folder");
  field(conf.method_inference_script_dir, "method_inference_script_dir");
  field(conf.relations_file, "relations_file");
  field(conf.save_objects, "save_objects");
}

void declare_config(ThreeDSSGConfig& conf) {
  using namespace config;
  name("ThreeDSSGConfig");
  field(conf.reasoning, "reasoning");
  field(conf.normal_estimation_radius, "normal_estimation_radius", "m");
  field(conf.num_sampling_points, "num_sampling_points");
  field(conf.edge_prob_threshold, "edge_prob_threshold");
}

}  // namespace hydra
