#pragma once

#include <glog/logging.h>
#include <spark_dsg/color.h>

#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "hydra/common/shared_module_state.h"
#include "hydra/reasoning/reasoning_config.h"
#include "hydra/reasoning/reasoning_output.h"
#include "hydra/utils/reasoning_io.h"

namespace hydra {

class Reasoning {
 public:
  using Ptr = std::unique_ptr<Reasoning>;
  explicit Reasoning(const ReasoningConfig& config);
  bool run(ReasoningOutput& reasoning_data, const std::string& room_name) const;

  std::string getRelationship(size_t index) const;
  spark_dsg::Color getRelationshipColor(const std::string& relationship) const;
  spark_dsg::Color getRelationshipColor(size_t index) const;

 private:
  bool runReasoningScript(ReasoningOutput& reasoning_data,
                          const std::string& room_name) const;
  void clearReasoning() const;

  ReasoningConfig config_;
  std::vector<std::string> relationships_;
  std::unordered_map<std::string, spark_dsg::Color> relationship_colors_;
};

}  // namespace hydra
