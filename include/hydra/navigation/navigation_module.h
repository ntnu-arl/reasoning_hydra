#pragma once

#include <config_utilities/config.h>
#include <config_utilities/factory.h>
#include <config_utilities/printing.h>
#include <config_utilities/types/eigen_matrix.h>
#include <config_utilities/types/enum.h>
#include <config_utilities/validation.h>
#include <glog/logging.h>
#include <spark_dsg/node_attributes.h>

#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "hydra/common/global_info.h"
#include "hydra/common/module.h"
#include "hydra/common/shared_module_state.h"
#include "hydra/navigation/dijkstra.h"
#include "hydra/utils/log_utilities.h"

namespace hydra {

struct NavigationPath {
  NodeId object_id;
  NodeId target_id;
  std::string object_label;
  std::string target_label;
  std::string explanation;
  std::vector<Eigen::Vector3d> agent_to_target;
  std::vector<Eigen::Vector3d> target_to_object;
};

struct NavigationInput {
  using Ptr = std::shared_ptr<NavigationInput>;
  std::vector<std::pair<NodeId, NodeId>> object_ids;
  std::vector<std::string> explanation;
  std::string method;
  bool object_search = false;
};

struct NavigationOutput {
  std::vector<NavigationPath> paths;
  bool object_search = false;
};

class NavigationModule : public Module {
 public:
  using Ptr = std::shared_ptr<NavigationModule>;
  using PathMethodVariant =
      std::variant<std::function<void(const std::map<NodeId, SceneGraphNode::Ptr>&,
                                      const std::set<EdgeKey>&,
                                      const NodeId&,
                                      const NodeId&,
                                      std::vector<NodeId>&,
                                      std::vector<Eigen::Vector3d>&)>>;

  struct Config {
  } const config;

  explicit NavigationModule(const Config& config);
  ~NavigationModule();

  void start() override;

  void stop() override;

  void save(const LogSetup& log_setup) override;

  std::string printInfo() const override;

  void spin();

  void spinOnce(const NavigationInput::Ptr& input);

  void setGraph(const DynamicSceneGraph::Ptr& scene_graph);

  InputQueue<NavigationInput::Ptr>::Ptr inputQueue() const { return input_queue_; }

  InputQueue<NavigationOutput>::Ptr outputQueue() const { return output_queue_; }

 protected:
  void stopImpl();

  bool findNavigation(const NodeId& obj1,
                      const NodeId& obj2,
                      const std::string& method,
                      const SceneGraphLayer::Nodes& place_nodes,
                      const SceneGraphLayer& object_layer,
                      const SceneGraphNode& agent_node,
                      const std::set<EdgeKey>& edges,
                      NavigationPath& output) const;

  bool findObjectNavigation(const NodeId& obj1,
                            const std::string& method,
                            const SceneGraphLayer::Nodes& place_nodes,
                            const SceneGraphLayer& object_layer,
                            const SceneGraphNode& agent_node,
                            const std::set<EdgeKey>& edges,
                            NavigationPath& output) const;

  std::unique_ptr<std::thread> spin_thread_;
  std::mutex mutex_;
  std::atomic<bool> should_shutdown_{false};

  DynamicSceneGraph::Ptr scene_graph_;
  InputQueue<NavigationInput::Ptr>::Ptr input_queue_;
  InputQueue<NavigationOutput>::Ptr output_queue_;
  std::unordered_map<std::string, PathMethodVariant> shortest_path_methods_;
};

void declare_config(NavigationModule::Config& conf);

}  // namespace hydra
