#pragma once
#include <filesystem>

#include <glog/logging.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>

#include <kimera_pgmo/kimera_pgmo_interface.h>
#include "hydra/backend/update_functions.h"
#include "hydra/reasoning/reasoning_config.h"
#include "hydra/common/shared_module_state.h"
#include "hydra/utils/nearest_neighbor_utilities.h"
#include "hydra/utils/timing_utilities.h"

namespace hydra {

struct UpdateReasoningFunctor {
  UpdateReasoningFunctor(const ThreeDSSGConfig& config, SharedModuleState::Ptr& state);
  MergeList call(const DynamicSceneGraph& unmerged,
                 SharedDsgInfo& dsg,
                 const UpdateInfo::ConstPtr& info);

  private:
  bool detectRoomChange(NodeId& room_to_reason, const SharedDsgInfo& dsg);
  void getObjectPointcloud(const SharedDsgInfo& dsg, const SceneGraphNode& room, pcl::PointCloud<pcl::PointXYZRGBL>::Ptr object_cloud) const;
  void savePointCloud(const pcl::PointCloud<pcl::PointXYZRGBL>::Ptr& cloud, const std::string& cloud_name, const std::string& format) const;

    ThreeDSSGConfig config_;
    SharedModuleState::Ptr state_;
    NodeId prev_room_node_id_;
    bool initialized_{false};
    std::unique_ptr<PointNeighborSearch> neighbor_search_;
};

}  // namespace hydra
