#pragma once

#include <glog/logging.h>

#include <filesystem>

#include "hydra/common/shared_module_state.h"
#include "hydra/reasoning/reasoning_config.h"
#include "hydra/utils/reasoning_io.h"

namespace hydra {

class Reasoning {
 public:
  Reasoning(const ReasoningConfig& config);
  bool run(
      const std::vector<NodeId>& object_ids,
      std::map<std::pair<NodeId, NodeId>, std::vector<std::string>>& reasoning_edges,
      const std::string& room_name) const;

 private:
  bool runReasoningScript(
      const std::vector<NodeId>& object_ids,
      std::map<std::pair<NodeId, NodeId>, std::vector<std::string>>& reasoning_edges,
      const std::string& room_name) const;

  ReasoningConfig config_;
  std::vector<std::string> relationships_;
};

}  // namespace hydra
