#include "hydra/reasoning/reasoning_config.h"

#include <config_utilities/config.h>

namespace hydra {

void declare_config(ThreeDSSGConfig& conf) {
  using namespace config;
  name("ThreeDSSGConfig");
  field(conf.inference_script, "inference_script");
  field(conf.input_folder, "input_folder");
  field(conf.output_folder, "output_folder");
  field(conf.normal_estimation_radius, "normal_estimation_radius", "m");
  field(conf.num_sampling_points, "num_sampling_points");
}

}  // namespace hydra
