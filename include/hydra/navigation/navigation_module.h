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
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "hydra/common/global_info.h"
#include "hydra/common/module.h"
#include "hydra/common/output_sink.h"
#include "hydra/common/shared_module_state.h"
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
};

using NavigationOutput = std::vector<NavigationPath>;

class NavigationModule : public Module {
 public:
  using Ptr = std::shared_ptr<NavigationModule>;
  using Sink = OutputSink<const NavigationOutput&>;

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

 protected:
  void stopImpl();

  std::unique_ptr<std::thread> spin_thread_;
  std::atomic<bool> should_shutdown_{false};
  DynamicSceneGraph::Ptr scene_graph_;

  InputQueue<NavigationInput::Ptr>::Ptr navigation_queue_;

  std::mutex mutex_;
};

void declare_config(NavigationModule::Config& conf);

}  // namespace hydra
