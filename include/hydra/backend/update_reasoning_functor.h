#pragma once
#include <glog/logging.h>
#include <kimera_pgmo/kimera_pgmo_interface.h>
#include <pcl/PolygonMesh.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <filesystem>
#include <memory>
#include <mutex>
#include <set>
#include <vector>

#include "hydra/backend/backend_utilities.h"
#include "hydra/backend/update_functions.h"
#include "hydra/common/global_info.h"
#include "hydra/common/shared_module_state.h"
#include "hydra/reasoning/reasoning.h"
#include "hydra/reasoning/reasoning_config.h"
#include "hydra/reasoning/reasoning_output.h"
#include "hydra/utils/nearest_neighbor_utilities.h"
#include "hydra/utils/pointcloud_utilities.h"
#include "hydra/utils/timing_utilities.h"

namespace hydra {

class UpdateReasoningFunctor {
 public:
  using Ptr = std::shared_ptr<UpdateReasoningFunctor>;

  UpdateReasoningFunctor(const ThreeDSSGConfig& config,
                         SharedModuleState::Ptr& state,
                         SharedDsgInfo::Ptr& dsg);

  void spin(std::mutex& mutex);
  void call(const UpdateInfo::ConstPtr& info,
            ObjectsAttributes::Ptr& objects_attributes);

  void setShutdown(bool should_shutdown);

 private:
  void spinOnce(const BackendReasoningInput& input, std::mutex& mutex);
  bool detectRoomChange(NodeId& room_to_reason);
  void getObjectPointcloud(const SceneGraphNode& room,
                           pcl::PointCloud<pcl::PointXYZRGB>::Ptr object_cloud,
                           std::vector<uint32_t>& instance_ids) const;

  bool areElementsInSet(const std::array<size_t, 3>& arr,
                        const std::set<size_t>& set) const;
  void getObjectMeshes(const SceneGraphNode& room,
                       ObjectsAttributes::Ptr& objects_attributes) const;

  void updateGraph(const ReasoningOutput& reasoning_data,
                   const std::vector<NodeId>& object_ids) const;

  void getEdgeIndices(ObjectsAttributes::Ptr& objects_attributes) const;

  std::atomic<bool> should_shutdown_{false};
  ThreeDSSGConfig config_;
  SharedModuleState::Ptr state_;
  NodeId prev_room_node_id_;
  bool initialized_{false};
  SharedDsgInfo::Ptr dsg_;
  Reasoning::Ptr reasoning_;
  std::unique_ptr<PointNeighborSearch> neighbor_search_;
  std::unique_ptr<pcl::KdTreeFLANN<pcl::PointXYZ>> object_centroids_tree_;
};

}  // namespace hydra
