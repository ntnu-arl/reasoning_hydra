#pragma once
#include <glog/logging.h>
#include <kimera_pgmo/kimera_pgmo_interface.h>
#include <pcl/PolygonMesh.h>

#include <filesystem>

#include "hydra/backend/update_functions.h"
#include "hydra/common/shared_module_state.h"
#include "hydra/reasoning/reasoning.h"
#include "hydra/reasoning/reasoning_config.h"
#include "hydra/utils/nearest_neighbor_utilities.h"
#include "hydra/utils/pointcloud_utilities.h"
#include "hydra/utils/timing_utilities.h"

namespace hydra {

struct UpdateReasoningFunctor {
  UpdateReasoningFunctor(const ThreeDSSGConfig& config, SharedModuleState::Ptr& state);
  MergeList call(const DynamicSceneGraph& unmerged,
                 SharedDsgInfo& dsg,
                 const UpdateInfo::ConstPtr& info);

 private:
  bool detectRoomChange(NodeId& room_to_reason, const SharedDsgInfo& dsg);
  void getObjectPointcloud(const SharedDsgInfo& dsg,
                           const SceneGraphNode& room,
                           pcl::PointCloud<pcl::PointXYZRGB>::Ptr object_cloud,
                           std::vector<uint32_t>& instance_ids) const;

  bool areElementsInSet(const std::array<size_t, 3>& arr,
                        const std::set<size_t>& set) const;
  void getObjectMeshes(const SharedDsgInfo& dsg,
                       const SceneGraphNode& room,
                       std::vector<NodeId>& object_ids,
                       std::vector<pcl::PolygonMesh::Ptr>& object_meshes,
                       std::vector<uint32_t>& mesh_labels) const;

  ThreeDSSGConfig config_;
  SharedModuleState::Ptr state_;
  NodeId prev_room_node_id_;
  bool initialized_{false};
  std::unique_ptr<PointNeighborSearch> neighbor_search_;
  std::unique_ptr<Reasoning> reasoning_;
};

}  // namespace hydra
