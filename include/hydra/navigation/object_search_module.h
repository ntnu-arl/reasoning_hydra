#pragma once

#include <config_utilities/config.h>
#include <config_utilities/factory.h>
#include <config_utilities/printing.h>
#include <config_utilities/types/eigen_matrix.h>
#include <config_utilities/types/enum.h>
#include <config_utilities/validation.h>
#include <glog/logging.h>
#include <spark_dsg/edge_attributes.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "hydra/common/global_info.h"
#include "hydra/common/module.h"
#include "hydra/common/shared_module_state.h"
#include "hydra/navigation/cos_sim_search.h"
#include "hydra/navigation/search.h"
#include "hydra/utils/log_utilities.h"

namespace hydra {

struct ObjectSearchInput {
  using Ptr = std::shared_ptr<ObjectSearchInput>;
  struct ObjectFeature {
    Eigen::MatrixXf data;
    int cols;
    int rows;
  };
  std::vector<ObjectFeature> text_object_embedding;
  ObjectFeature text_room_embedding;
  std::string room;
  std::string prompt;
};

struct ObjectSearchOutput {
  using Ptr = std::shared_ptr<ObjectSearchOutput>;
  struct ObjectRelationship {
    spark_dsg::NodeId id;
    struct ObjectFeature {
      spark_dsg::NodeId object1;
      spark_dsg::NodeId object2;
      Eigen::MatrixXf feature;
    };
    std::vector<ObjectFeature> relationships;
  };
  std::vector<ObjectRelationship> objects;
  std::string room;
  std::string prompt;
};

class ObjectSearchModule : public Module {
 public:
  using Ptr = std::shared_ptr<ObjectSearchModule>;

  struct Config {
    config::VirtualConfig<Search> search;
  } const config;

  explicit ObjectSearchModule(const Config& config);
  ~ObjectSearchModule() override;

  void start() override;

  void stop() override;

  void save(const LogSetup& log_setup) override;

  std::string printInfo() const override;

  void spin();

  void spinOnce(const ObjectSearchInput::Ptr& input);

  void setGraph(const DynamicSceneGraph::Ptr& scene_graph);

  InputQueue<ObjectSearchInput::Ptr>::Ptr inputQueue() const { return input_queue_; }

  InputQueue<ObjectSearchOutput::Ptr>::Ptr outputQueue() const { return output_queue_; }

 protected:
  NodeId findRoom(const ObjectSearchInput::Ptr& input,
                  ObjectSearchOutput::Ptr& output) const;

  bool findObjects(const ObjectSearchInput::Ptr& input,
                   ObjectSearchOutput::Ptr& output,
                   const NodeId& chosen_room_id) const;

  std::unique_ptr<std::thread> spin_thread_;
  std::mutex mutex_;
  std::atomic<bool> should_shutdown_{false};

  DynamicSceneGraph::Ptr scene_graph_;
  Search::Ptr cos_sim_search_;
  InputQueue<ObjectSearchInput::Ptr>::Ptr input_queue_;
  InputQueue<ObjectSearchOutput::Ptr>::Ptr output_queue_;

 private:
  void stopImpl();
  inline static const auto registration_ =
      config::RegistrationWithConfig<ObjectSearchModule, ObjectSearchModule, Config>(
          "ObjectSearchModule");
};

void declare_config(ObjectSearchModule::Config& config);

}  // namespace hydra
