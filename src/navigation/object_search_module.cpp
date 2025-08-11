#include "hydra/navigation/object_search_module.h"

namespace hydra {

void declare_config(ObjectSearchModule::Config& config) {
  using namespace config;
  name("ObjectSearchConfig::Config");
  config.search.setOptional();
  field(config.search, "search");
  field(config.verbose, "verbose");
  field(config.room_prob_threshold, "room_prob_threshold");
  field(config.object_prob_threshold, "object_prob_threshold");
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
  output->general_prompt = input->prompt;
  output->object_search = input->object_search;
  const auto room_ids = findRoom(input, output);
  if (output->room.empty()) {
    LOG(WARNING) << "Room not found!";
    return;
  }
  if (!findObjects(input, output, room_ids)) {
    LOG(WARNING) << "Objects not found!";
    return;
  }
  sortObjects(output);
  if (!output_queue_->push(output)) {
    LOG(ERROR) << "Failed to push object search output!";
  }
}

std::optional<std::vector<NodeId>> ObjectSearchModule::findRoom(
    const ObjectSearchInput::Ptr& input, ObjectSearchOutput::Ptr& output) const {

  if (input->room == "all") {
    output->room = "all";
    return std::nullopt;  // Special case to use all rooms
  }
  const auto& room_nodes = scene_graph_->getLayer(DsgLayers::ROOMS).nodes();
  std::vector<std::vector<Eigen::VectorXf>> room_embeddings;
  std::vector<std::string> room_names;
  std::vector<NodeId> room_ids;
  for (const auto& [room_id, room_node] : room_nodes) {
    if (!input->room.empty() && input->room != "find") {
      if (room_node->attributes<SemanticNodeAttributes>().name == input->room) {
        output->room = input->room;
        return std::vector<NodeId>{room_id};  // Return the room ID if it matches
      }
    } else {
      room_embeddings.push_back(
          room_node->attributes<RoomNodeAttributes>().feature_vectors);
      room_names.push_back(room_node->attributes<SemanticNodeAttributes>().name);
      room_ids.push_back(room_id);
    }
  }
  std::vector<float> probs;
  if (input->room == "find") {
    std::vector<Eigen::VectorXf> object_embeddings;
    for (const auto& object_feature : input->text_object_embedding) {
      object_embeddings.push_back(Eigen::VectorXf(object_feature.data));
    }
    std::vector<size_t> results;
    if (cos_sim_search_->searchRooms(object_embeddings, room_embeddings,
                                     config.room_prob_threshold, results, probs)) {
      std::vector<NodeId> found_room_ids;
      for (const auto& result : results) {
        if (result < room_ids.size()) {
          output->room += room_names[result] + ", ";
          found_room_ids.push_back(room_ids[result]);
        }
      }
      if (config.verbose) {
        // Log the probabilities of the rooms and the threshold
        std::string log_message = "Room probabilities: ";
        for (size_t i = 0; i < probs.size(); ++i) {
          log_message += room_names[i] + ": " + std::to_string(probs[i]) + ", ";
        }
        LOG(INFO) << log_message;
      }          
      return found_room_ids;
    }
    if (config.verbose) {
      // Log the probabilities of the rooms and the threshold
      std::string log_message = "Room probabilities: ";
      for (size_t i = 0; i < probs.size(); ++i) {
        log_message += room_names[i] + ": " + std::to_string(probs[i]) + ", ";
      }
      LOG(INFO) << log_message;
    }    
  } else if (input->room.empty()) {
    size_t result;
    if (cos_sim_search_->searchRoom(Eigen::VectorXf(input->text_room_embedding.data),
                                    room_embeddings,
                                    config.room_prob_threshold,
                                    result,
                                    probs)) {
      output->room = room_names[result];
      return std::vector<NodeId>{room_ids[result]};
    }
  }
  return std::nullopt;  // No room found or specified
}

bool ObjectSearchModule::findObjects(
    const ObjectSearchInput::Ptr& input,
    ObjectSearchOutput::Ptr& output,
    const std::optional<std::vector<NodeId>>& chosen_room_ids) const {
  std::vector<NodeId> objects_in_room;
  const auto& object_nodes = scene_graph_->getLayer(DsgLayers::OBJECTS).nodes();
  std::vector<Eigen::VectorXf> object_embeddings;

  for (const auto& [object_id, object_node] : object_nodes) {
    const auto& attrs = object_node->attributes<ObjectNodeAttributes>();
    if (!attrs.validFeatures() || attrs.mesh_connections.size() < config.min_object_vertices) {
      continue;
    }
    if (!chosen_room_ids) {
      objects_in_room.push_back(object_id);
      object_embeddings.push_back(
          attrs.semantic_feature);
      continue;  // If no specific room is chosen, include all objects
    }
    const auto& place_id = object_node->getParent();
    if (!place_id) {
      continue;
    }
    const auto& room_id = scene_graph_->getNode(*place_id).getParent();
    if (!room_id) {
      continue;
    }
    if (std::find(chosen_room_ids.value().begin(),
                  chosen_room_ids.value().end(),
                  *room_id) != chosen_room_ids.value().end()) {
      objects_in_room.push_back(object_id);
      object_embeddings.push_back(
          attrs.semantic_feature);
    }
  }

  if (objects_in_room.empty()) {
    LOG(WARNING) << "No objects found in room!";
    return false;
  }

  if (input->object_search) {
    return basicObjectSearch(input, output, objects_in_room, object_embeddings);
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
    LOG(WARNING) << "No edges found in room!";
    return false;
  }

  // Basic object search, where pairs of objects (for relationships) are not specified
  if (input->objects_prompt_pairs.empty()) {
    return basicObjectRelationshipsSearch(
        input, output, objects_in_room, object_embeddings, edges_in_room);
  }

  // Pair-based object search, where pairs of objects are specified
  return pairBasedObjectRelationshipsSearch(
      input, output, objects_in_room, object_embeddings, edges_in_room);
}

bool ObjectSearchModule::basicObjectRelationshipsSearch(
    const ObjectSearchInput::Ptr& input,
    ObjectSearchOutput::Ptr& output,
    const std::vector<NodeId>& objects_in_room,
    const std::vector<Eigen::VectorXf>& object_embeddings,
    const std::unordered_map<NodeId, std::vector<NodeId>>& edges_in_room) const {
  bool any_edges = false;
  for (const auto& object_text_feature : input->text_object_embedding) {
    std::vector<size_t> results;
    std::vector<float> probs;
    if (cos_sim_search_->searchObject(
            Eigen::VectorXf(object_text_feature.data), object_embeddings,
            config.object_prob_threshold, results, probs)) {
      for (const auto& result : results) {
        if (edges_in_room.count(objects_in_room[result]) == 0) {
          continue;
        }
        ObjectSearchOutput::ObjectRelationship object_relationship;
        object_relationship.id = objects_in_room[result];
        for (const auto& edge : edges_in_room.at(object_relationship.id)) {
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

bool ObjectSearchModule::pairBasedObjectRelationshipsSearch(
    const ObjectSearchInput::Ptr& input,
    ObjectSearchOutput::Ptr& output,
    const std::vector<NodeId>& objects_in_room,
    const std::vector<Eigen::VectorXf>& object_embeddings,
    const std::unordered_map<NodeId, std::vector<NodeId>>& edges_in_room) const {
  bool any_edges = false;
  // First, we find the objects with the highest cosine similairty compared to the list
  // of objects in the room.
  std::unordered_map<size_t, std::vector<NodeId>> found_objects;
  for (std::size_t i = 0; i < input->text_object_embedding.size(); ++i) {
    const auto& object_text_feature = input->text_object_embedding[i];
    std::vector<size_t> results;
    std::vector<float> probs;
    if (cos_sim_search_->searchObject(
            Eigen::VectorXf(object_text_feature.data), object_embeddings,
            config.object_prob_threshold, results, probs)) {
      for (const auto& result : results) {
        if (result < objects_in_room.size()) {
          found_objects[i].push_back(objects_in_room[result]);
        }
      }
    }
  }
  // Now we iterate over the pairs of objects and their corresponding prompt.
  // We look at the found objects and their relationships.
  std::unordered_set<std::pair<spark_dsg::NodeId, spark_dsg::NodeId>, PairHashNodeId> seen_pairs;
  for (const auto& pair : input->objects_prompt_pairs) {
    if (found_objects.count(pair.object_label_index) == 0 ||
        found_objects.count(pair.subject_label_index) == 0) {
      continue;  // Skip if no objects found for this pair
    }
    for (const auto& object : found_objects.at(pair.object_label_index)) {
      ObjectSearchOutput::ObjectRelationship object_relationship;
      object_relationship.id = object;
      if (edges_in_room.count(object_relationship.id) == 0) {
        continue;  // Skip if no edges for this object
      }
      for (const auto& subject : edges_in_room.at(object_relationship.id)) {
        // Check if the subject is in the found objects for the subject label index
        if (std::find(found_objects.at(pair.subject_label_index).begin(),
                      found_objects.at(pair.subject_label_index).end(),
                      subject) == found_objects.at(pair.subject_label_index).end()) {
          continue;  // Skip if subject is not found
        }
        std::pair<spark_dsg::NodeId, spark_dsg::NodeId> key = std::minmax(object_relationship.id, subject);
        if (seen_pairs.count(key) > 0) {
          continue;  // Skip if this pair has already been seen
        }
        seen_pairs.insert(key);
        if (!scene_graph_->hasNode(object_relationship.id) ||
            !scene_graph_->hasNode(subject)) {
          continue;  // Skip if nodes do not exist
        }
        const std::string& object_name = scene_graph_->getNode(object_relationship.id)
                                             .attributes<SemanticNodeAttributes>()
                                             .name;
        const std::string& subject_name =
            scene_graph_->getNode(subject).attributes<SemanticNodeAttributes>().name;
        if (object_name == subject_name) {
          continue;  // Skip if object and subject are the same
        }

        any_edges = true;
        ObjectSearchOutput::ObjectRelationship::ObjectFeature object_feature;
        object_feature.object1 = object_relationship.id;
        object_feature.object2 = subject;
        
        object_feature.object1_label = object_name;
        object_feature.object2_label = subject_name;
        const auto edge = scene_graph_->getEdge(object_relationship.id, subject).attributes<EdgeAttributes>();
        object_feature.feature = edge.feature(object_relationship.id);
        VLOG(1) << "[Object Search] Found relationship between: "
                 << object_name << " and " << subject_name
                 << " with num_observations: " << edge.numObservations(object_relationship.id);
        object_feature.num_observations = edge.numObservations(object_relationship.id);
        object_feature.prompt = pair.prompt;
        object_relationship.relationships.push_back(object_feature);
      }
      if (!object_relationship.relationships.empty()) {
        output->objects.push_back(object_relationship);
      }
    }
  }
  return any_edges;
}

bool ObjectSearchModule::basicObjectSearch(
    const ObjectSearchInput::Ptr& input,
    ObjectSearchOutput::Ptr& output,
    const std::vector<NodeId>& objects_in_room,
    const std::vector<Eigen::VectorXf>& object_embeddings) const {
  bool any_objects = false;
  for (const auto& object_text_feature : input->text_object_embedding) {
    std::vector<size_t> results;
    std::vector<float> probs;
    if (cos_sim_search_->searchObject(
            Eigen::VectorXf(object_text_feature.data), object_embeddings,
            config.object_prob_threshold, results, probs)) {
      any_objects = true;
      for (const auto& result : results) {
        ObjectSearchOutput::ObjectRelationship object_relationship;
        object_relationship.id = objects_in_room[result];
        output->objects.push_back(object_relationship);
        if (config.verbose) {
          LOG(INFO) << "Found object: " << scene_graph_->getNode(object_relationship.id)
                    .attributes<SemanticNodeAttributes>().name << " with probability "
                     << probs[result];
        }
      }
    }
  }
  return any_objects;
}

void ObjectSearchModule::sortObjects(ObjectSearchOutput::Ptr& output) const {
  if (output->objects.empty()) {
    return;  // No objects to sort
  }
  const auto& agent_layer =
      scene_graph_->dynamicLayersOfType(DsgLayers::AGENTS).begin()->second;
  if (agent_layer->numNodes() == 0) {
    return;
  }
  const auto& current_pose = agent_layer->getNodeByIndex(agent_layer->numNodes() - 1).attributes<AgentNodeAttributes>().position;

  std::sort(output->objects.begin(), output->objects.end(),
            [this, &current_pose](const ObjectSearchOutput::ObjectRelationship& a,
                            const ObjectSearchOutput::ObjectRelationship& b) {
              if (!scene_graph_->hasNode(a.id) ||
                  !scene_graph_->hasNode(b.id)) {
                return false;  // Keep order if nodes do not exist
              }
              const auto& object_a_pose = scene_graph_->getNode(a.id)
                    .attributes<SemanticNodeAttributes>().position;
              const auto& object_b_pose = scene_graph_->getNode(b.id)
                    .attributes<SemanticNodeAttributes>().position;

              double dist_a = (object_a_pose - current_pose).norm();
              double dist_b = (object_b_pose - current_pose).norm();
              return dist_a < dist_b;
            });
}


void ObjectSearchModule::setGraph(const DynamicSceneGraph::Ptr& scene_graph) {
  std::lock_guard<std::mutex> lock(mutex_);
  scene_graph_ = scene_graph;
}

}  // namespace hydra
