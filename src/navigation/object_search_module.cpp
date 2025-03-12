#include "hydra/navigation/object_search_module.h"

namespace hydra {

void declare_config(ObjectSearchModule::Config& config) {
  using namespace config;
  name("ObjectSearchConfig::Config");
  config.search.setOptional();
  field(config.search, "search");
}

ObjectSearchModule::ObjectSearchModule(const Config& config)
    : config(config), cos_sim_search_(config.search.create()) {
  input_queue_.reset(new InputQueue<ObjectSearchInput::Ptr>());
  output_queue_.reset(new InputQueue<ObjectSearchOutput::Ptr>());
}

ObjectSearchModule::~ObjectSearchModule() { stopImpl(); }

void ObjectSearchModule::start() {
  spin_thread_.reset(new std::thread(&ObjectSearchModule::spin, this));
  LOG(INFO) << "[Hydra ObjectSearch] started!";
}

void ObjectSearchModule::stopImpl() {
  should_shutdown_ = true;
  if (spin_thread_) {
    VLOG(2) << "[Hydra ObjectSearch] joining object search thread and stopping";
    spin_thread_->join();
    spin_thread_.reset();
    VLOG(2) << "[Hydra ObjectSearch] stopped!";
  }
}

void ObjectSearchModule::stop() { stopImpl(); }

void ObjectSearchModule::save(const LogSetup& log_setup) {}

std::string ObjectSearchModule::printInfo() const {
  std::stringstream ss;
  ss << config::toString(config);
  return ss.str();
}

void ObjectSearchModule::spin() {
  bool should_shutdown = false;
  while (!should_shutdown) {
    bool has_data = input_queue_->poll();
    if (GlobalInfo::instance().force_shutdown() || !has_data) {
      // copy over shutdown request
      should_shutdown = should_shutdown_;
    }

    if (!has_data) {
      continue;
    }

    spinOnce(input_queue_->front());
    input_queue_->pop();
  }
}

void ObjectSearchModule::spinOnce(const ObjectSearchInput::Ptr& input) {
  std::lock_guard<std::mutex> lock(mutex_);
  ObjectSearchOutput::Ptr output = std::make_shared<ObjectSearchOutput>();
  output->prompt = input->prompt;
  NodeId room_id = findRoom(input, output);
  if (output->room.empty()) {
    LOG(ERROR) << "Room not found!";
    return;
  }
  if (!findObjects(input, output, room_id)) {
    LOG(ERROR) << "Objects not found!";
    return;
  }
  if (!output_queue_->push(output)) {
    LOG(ERROR) << "Failed to push object search output!";
  }
}

NodeId ObjectSearchModule::findRoom(const ObjectSearchInput::Ptr& input,
                                    ObjectSearchOutput::Ptr& output) const {
  const auto& room_nodes = scene_graph_->getLayer(DsgLayers::ROOMS).nodes();
  std::vector<std::vector<Eigen::VectorXf>> room_embeddings;
  std::vector<std::string> room_names;
  std::vector<NodeId> room_ids;
  for (const auto& [room_id, room_node] : room_nodes) {
    if (!input->room.empty()) {
      if (room_node->attributes<SemanticNodeAttributes>().name == input->room) {
        output->room = input->room;
        return room_id;
      }
    } else {
      room_embeddings.push_back(
          room_node->attributes<RoomNodeAttributes>().feature_vectors);
      room_names.push_back(room_node->attributes<SemanticNodeAttributes>().name);
      room_ids.push_back(room_id);
    }
  }
  if (input->room.empty()) {
    size_t result;
    if (cos_sim_search_->searchRoom(Eigen::VectorXf(input->text_room_embedding.data),
                                    room_embeddings,
                                    result)) {
      output->room = room_names[result];
      return room_ids[result];
    }
  }
  return 0;
}

bool ObjectSearchModule::findObjects(const ObjectSearchInput::Ptr& input,
                                     ObjectSearchOutput::Ptr& output,
                                     const NodeId& chosen_room_id) const {
  std::vector<NodeId> objects_in_room;
  const auto& object_nodes = scene_graph_->getLayer(DsgLayers::OBJECTS).nodes();
  std::vector<Eigen::VectorXf> object_embeddings;

  for (const auto& [object_id, object_node] : object_nodes) {
    if (!object_node->attributes<ObjectNodeAttributes>().validFeatures()) {
      continue;
    }
    const auto& place_id = object_node->getParent();
    if (!place_id) {
      continue;
    }
    const auto& room_id = scene_graph_->getNode(*place_id).getParent();
    if (!room_id) {
      continue;
    }
    if (*room_id == chosen_room_id) {
      objects_in_room.push_back(object_id);
      object_embeddings.push_back(
          object_node->attributes<ObjectNodeAttributes>().semantic_feature);
    }
  }

  if (objects_in_room.empty()) {
    LOG(ERROR) << "No objects found in room!";
    return false;
  }

  std::unordered_map<NodeId, std::vector<NodeId>> edges_in_room;
  for (size_t i = 0; i < objects_in_room.size(); ++i) {
    for (size_t j = 0; j < objects_in_room.size(); ++j) {
      if (i == j) {
        continue;
      }
      if (scene_graph_->hasEdge(objects_in_room[i], objects_in_room[j])) {
        if (edges_in_room.count(objects_in_room[i]) == 0) {
          edges_in_room[objects_in_room[i]] = std::vector<NodeId>();
        }
        edges_in_room[objects_in_room[i]].push_back(objects_in_room[j]);
      }
    }
  }

  if (edges_in_room.empty()) {
    LOG(ERROR) << "No edges found in room!";
    return false;
  }

  bool any_edges = false;
  for (const auto& object_text_feature : input->text_object_embedding) {
    std::vector<size_t> results;
    if (cos_sim_search_->searchObject(
            Eigen::VectorXf(object_text_feature.data), object_embeddings, results)) {
      for (const auto& result : results) {
        ObjectSearchOutput::ObjectRelationship object_relationship;
        object_relationship.id = objects_in_room[result];
        for (const auto& edge : edges_in_room[object_relationship.id]) {
          any_edges = true;
          ObjectSearchOutput::ObjectRelationship::ObjectFeature object_feature;
          object_feature.object1 = object_relationship.id;
          object_feature.object2 = edge;
          object_feature.object1_label = scene_graph_->getNode(object_relationship.id)
                                             .attributes<SemanticNodeAttributes>()
                                             .name;
          object_feature.object2_label =
              scene_graph_->getNode(edge).attributes<SemanticNodeAttributes>().name;
          object_feature.feature = scene_graph_->getEdge(object_relationship.id, edge)
                                       .attributes<EdgeAttributes>()
                                       .feature(object_relationship.id);
          object_relationship.relationships.push_back(object_feature);
        }
        output->objects.push_back(object_relationship);
      }
    }
  }
  return any_edges;
}

void ObjectSearchModule::setGraph(const DynamicSceneGraph::Ptr& scene_graph) {
  std::lock_guard<std::mutex> lock(mutex_);
  scene_graph_ = scene_graph;
}

}  // namespace hydra
