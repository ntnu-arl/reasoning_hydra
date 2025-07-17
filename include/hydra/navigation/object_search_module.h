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
  std::string room;
  std::string prompt;
  bool object_search;
  struct ObjectFeature {
    Eigen::MatrixXf data;
    int cols;
    int rows;
  };
  struct ObjectsPtomptsPair {
    size_t object_label_index;
    size_t subject_label_index;
    std::string prompt;
  };
  std::vector<ObjectFeature> text_object_embedding;
  ObjectFeature text_room_embedding;
  std::vector<ObjectsPtomptsPair> objects_prompt_pairs;
};

struct ObjectSearchOutput {
  using Ptr = std::shared_ptr<ObjectSearchOutput>;
  struct ObjectRelationship {
    spark_dsg::NodeId id;
    struct ObjectFeature {
      spark_dsg::NodeId object1;
      spark_dsg::NodeId object2;
      std::string object1_label;
      std::string object2_label;
      Eigen::MatrixXf feature;
      std::string prompt;
    };
    std::vector<ObjectFeature> relationships;
  };
  std::vector<ObjectRelationship> objects;
  std::string room;
  std::string general_prompt;
  bool object_search = false;
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
  std::optional<NodeId> findRoom(const ObjectSearchInput::Ptr& input,
                                 ObjectSearchOutput::Ptr& output) const;

  bool findObjects(const ObjectSearchInput::Ptr& input,
                   ObjectSearchOutput::Ptr& output,
                   const std::optional<NodeId>& chosen_room_id) const;

  bool basicObjectRelationshipsSearch(
      const ObjectSearchInput::Ptr& input,
      ObjectSearchOutput::Ptr& output,
      const std::vector<NodeId>& objects_in_room,
      const std::vector<Eigen::VectorXf>& object_embeddings,
      const std::unordered_map<NodeId, std::vector<NodeId>>& edges_in_room) const;
  bool pairBasedObjectRelationshipsSearch(
      const ObjectSearchInput::Ptr& input,
      ObjectSearchOutput::Ptr& output,
      const std::vector<NodeId>& objects_in_room,
      const std::vector<Eigen::VectorXf>& object_embeddings,
      const std::unordered_map<NodeId, std::vector<NodeId>>& edges_in_room) const;

  bool basicObjectSearch(const ObjectSearchInput::Ptr& input,
                         ObjectSearchOutput::Ptr& output,
                         const std::vector<NodeId>& objects_in_room,
                         const std::vector<Eigen::VectorXf>& object_embeddings) const;

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
