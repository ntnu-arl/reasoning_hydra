#pragma once
#include <filesystem>

#include <kimera_pgmo/kimera_pgmo_interface.h>
#include "hydra/backend/update_functions.h"
#include "hydra/reasoning/reasoning_config.h"
#include "hydra/common/shared_module_state.h"
#include "hydra/utils/nearest_neighbor_utilities.h"

namespace hydra {

struct UpdateReasoningFunctor {
  UpdateReasoningFunctor(const ThreeDSSGConfig& config, SharedModuleState::Ptr& state);
  MergeList call(const DynamicSceneGraph& unmerged,
                 SharedDsgInfo& dsg,
                 const UpdateInfo::ConstPtr& info);

  private:
    bool detectRoomChange(NodeId& room_to_reason, const SharedDsgInfo& dsg);

    ThreeDSSGConfig config_;
    SharedModuleState::Ptr state_;
    NodeId prev_room_node_id_;
    bool initialized_{false};
    std::unique_ptr<PointNeighborSearch> neighbor_search_;
};

}  // namespace hydra
