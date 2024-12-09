#include "hydra/backend/update_reasoning_functor.h"

namespace hydra {

using timing::ScopedTimer;

UpdateReasoningFunctor::UpdateReasoningFunctor(const ThreeDSSGConfig& config,
                                               SharedModuleState::Ptr& state,
                                               SharedDsgInfo::Ptr& dsg)
    : config_(config),
      state_(state),
      dsg_(dsg),
      reasoning_(std::make_unique<Reasoning>(config.reasoning)),
      object_centroids_tree_(std::make_unique<pcl::KdTreeFLANN<pcl::PointXYZ>>()) {}

void UpdateReasoningFunctor::setShutdown(bool should_shutdown) {
  should_shutdown_ = should_shutdown;
}

void UpdateReasoningFunctor::spin(std::mutex& mutex) {
  bool should_shutdown = false;
  while (!should_shutdown) {
    bool has_data = state_->reasoning_queue.poll();
    if (GlobalInfo::instance().force_shutdown() || !has_data) {
      // copy over shutdown request
      should_shutdown = should_shutdown_;
    }

    if (!has_data) {
      continue;
    }

    spinOnce(*state_->reasoning_queue.front(), mutex);
    state_->reasoning_queue.pop();
  }
}

void UpdateReasoningFunctor::spinOnce(const BackendReasoningInput& input,
                                      std::mutex& mutex) {
  std::lock_guard<std::mutex> lock(mutex);
  if (!input.reasoning_output) {
    return;
  }
  // Update the graph with the reasoning output
  updateGraph(*(input.reasoning_output), input.node_ids);
}

void UpdateReasoningFunctor::call(const UpdateInfo::ConstPtr& info,
                                  ObjectsAttributes::Ptr& objects_attributes) {
  if (dsg_->graph->getLayer(DsgLayers::ROOMS).numNodes() == 0 ||
      state_->latest_places.empty()) {
    return;
  }

  // Initialize previous room node id to the first room that we have seen
  if (!initialized_) {
    const auto& latest_place_id = *state_->latest_places.begin();
    assert(dsg_->graph->hasNode(latest_place_id));
    if (!dsg_->graph->getNode(latest_place_id).hasParent()) {
      return;
    }
    prev_room_node_id_ = *(dsg_->graph->getNode(latest_place_id).parents().begin());
    initialized_ = true;
    return;
  }
  // Detect if the room has changed
  NodeId room_to_reason_id;
  {
    ScopedTimer timer_room_change("backend/update_reasoning/detect_room_change",
                                  info->timestamp_ns);
    if (!detectRoomChange(room_to_reason_id)) {
      return;
    }
  }

  // Get the room to reason about
  if (!dsg_->graph->getLayer(DsgLayers::ROOMS).hasNode(room_to_reason_id)) {
    return;
  }
  const auto& room_to_reason =
      dsg_->graph->getLayer(DsgLayers::ROOMS).getNode(room_to_reason_id);

  // Get the object meshes and labels
  {
    ScopedTimer timer_get_object_meshes("backend/update_reasoning/get_object_meshes",
                                        info->timestamp_ns);
    getObjectMeshes(room_to_reason, objects_attributes);
  }
  // Compute the edges to reason about
  {
    ScopedTimer timer_get_edge_indices("backend/update_reasoning/get_edge_indices",
                                       info->timestamp_ns);
    getEdgeIndices(objects_attributes);
  }

  if (config_.reasoning.save_objects) {
    ScopedTimer timer_save("backend/update_reasoning/save", info->timestamp_ns);
    // Save meshes and labels to file
    std::filesystem::path object_mesh_path =
        std::filesystem::path(config_.reasoning.input_folder) /
        room_to_reason.attributes<SemanticNodeAttributes>().name;
    std::filesystem::create_directories(object_mesh_path);
    for (size_t i = 0; i < objects_attributes->meshes.size(); ++i) {
      std::string filename = (object_mesh_path / (std::to_string(i) + ".ply")).string();
      pcl::io::savePLYFile(filename, *objects_attributes->meshes[i]);
    }
    saveVectorToBinary(objects_attributes->labels,
                       (object_mesh_path / "labels").string());
  }
  // Run the reasoning script
  // ReasoningOutput reasoning_data;
  // {
  //   ScopedTimer timer_run_script("backend/update_reasoning/run_script",
  //                                info->timestamp_ns);
  //   if (!reasoning_->run(reasoning_data,
  //                        room_to_reason.attributes<SemanticNodeAttributes>().name)) {
  //     return;
  //   }
  // }

  // // Add reasoning edges to the graph
  // {
  //   ScopedTimer timer_update_graph("backend/update_reasoning/update_graph",
  //                                  info->timestamp_ns);
  //   updateGraph(reasoning_data, object_ids);
  // }
}

void UpdateReasoningFunctor::updateGraph(const ReasoningOutput& reasoning_data,
                                         const std::vector<NodeId>& object_ids) const {
  auto& graph = dsg_->graph;

  for (size_t i = 0; i < reasoning_data.edge_indices.size(); ++i) {
    size_t from = reasoning_data.edge_indices[i].first;
    size_t to = reasoning_data.edge_indices[i].second;

    EdgeAttributes::Ptr edge = std::make_unique<EdgeAttributes>(1.0);
    bool edge_exists = graph->hasEdge(object_ids[from], object_ids[to]);
    if (edge_exists) {
      edge = graph->getEdge(object_ids[from], object_ids[to]).info->clone();
      if (edge->source_id == object_ids[from] &&
          !reasoning_data.feature_vectors.empty()) {
        if (!reasoning_data.feature_vectors[i].empty()) {
          edge->relationship_source_target.feature_vector =
              reasoning_data.feature_vectors[i];
        }
      } else if (edge->source_id == object_ids[to] &&
                 !reasoning_data.feature_vectors.empty()) {
        if (!reasoning_data.feature_vectors[i].empty()) {
          edge->relationship_target_source.feature_vector =
              reasoning_data.feature_vectors[i];
        }
      }
    } else {
      edge->source_id = object_ids[from];
      edge->target_id = object_ids[to];
      if (!reasoning_data.feature_vectors.empty()) {
        if (!reasoning_data.feature_vectors[i].empty()) {
          edge->relationship_source_target.feature_vector =
              reasoning_data.feature_vectors[i];
        }
      }
    }
    edge->min_prob = config_.edge_prob_threshold;
    edge->setRelationshipProperty(object_ids[from],
                                  reasoning_->getRelationships(),
                                  reasoning_data.edge_probs[i],
                                  reasoning_->getRelationshipColors());

    if (edge_exists) {
      graph->setEdgeAttributes(object_ids[from], object_ids[to], std::move(edge));
    } else {
      graph->insertEdge(object_ids[from], object_ids[to], std::move(edge));
    }
  }
}

bool UpdateReasoningFunctor::detectRoomChange(NodeId& room_to_reason_id) {
  // Get place nodes that have parents (i.e. are in rooms)
  room_to_reason_id = prev_room_node_id_;
  double closest_distance = std::numeric_limits<double>::max();
  double closest_current_room_distance = std::numeric_limits<double>::max();
  const auto& places = dsg_->graph->getLayer(DsgLayers::PLACES).nodes();
  const auto current_pose =
      dsg_->graph->dynamicLayersOfType(DsgLayers::AGENTS)
          .begin()
          ->second
          ->getPositionByIndex(dsg_->graph->dynamicLayersOfType(DsgLayers::AGENTS)
                                   .begin()
                                   ->second->numNodes() -
                               1)
          .cast<float>();
  for (const auto& [place_id, place] : places) {
    if (NodeSymbol(place_id).category() == 'p' && place->hasParent()) {
      const auto& place_pose =
          place->attributes<NodeAttributes>().position.cast<float>();
      const auto distance_to_pose = (place_pose - current_pose).norm();
      if (distance_to_pose < closest_distance) {
        closest_distance = distance_to_pose;
        room_to_reason_id = *(place->parents().begin());
      }
      if (place->parents().find(prev_room_node_id_) != place->parents().end() &&
          distance_to_pose < closest_current_room_distance) {
        closest_current_room_distance = distance_to_pose;
      }
    }
  }

  if (closest_current_room_distance < config_.room_change_threshold ||
      room_to_reason_id == prev_room_node_id_) {
    return false;
  }

  NodeId aux_room_to_reason_id = room_to_reason_id;
  room_to_reason_id = prev_room_node_id_;
  prev_room_node_id_ = aux_room_to_reason_id;

  return true;
}

bool UpdateReasoningFunctor::areElementsInSet(const std::array<size_t, 3>& arr,
                                              const std::set<size_t>& set) const {
  // Iterate through all elements in the array
  for (size_t i = 0; i < arr.size(); ++i) {
    // Check if the element is found in the set
    if (set.find(arr[i]) == set.end()) {
      return false;  // If any element is not found, return false
    }
  }
  return true;  // All elements are found in the set
}

void UpdateReasoningFunctor::getObjectMeshes(
    const SceneGraphNode& room, ObjectsAttributes::Ptr& objects_attributes) const {
  // Iterate over the objects in the room, store a pointcloud of the objects
  for (const auto& place_id : room.children()) {
    if (!dsg_->graph->getLayer(DsgLayers::PLACES).hasNode(place_id)) {
      continue;
    }
    const auto& place = dsg_->graph->getLayer(DsgLayers::PLACES).getNode(place_id);
    for (const auto& object_id : place.children()) {
      if (!dsg_->graph->getLayer(DsgLayers::OBJECTS).hasNode(object_id)) {
        continue;
      }
      const auto& object = dsg_->graph->getLayer(DsgLayers::OBJECTS).getNode(object_id);
      const auto& object_attrs = object.attributes<ObjectNodeAttributes>();
      const auto& object_mesh_conections = object_attrs.mesh_connections;
      if (object_mesh_conections.empty()) {
        continue;
      }
      pcl::PolygonMesh::Ptr object_mesh(new pcl::PolygonMesh);
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(
          new pcl::PointCloud<pcl::PointXYZRGB>);
      std::set<size_t> vertex_positions;
      std::map<size_t, size_t> vertex_map;
      bool saved_label = false;
      size_t j = 0;
      for (const auto& vertex_index : object_mesh_conections) {
        if (vertex_index >= dsg_->graph->mesh()->numVertices()) {
          continue;
        }
        const auto vertex_pos = dsg_->graph->mesh()->pos(vertex_index).cast<double>();
        vertex_positions.insert(vertex_index);
        vertex_map[vertex_index] = j;
        j++;
        const auto vertex_color = dsg_->graph->mesh()->color(vertex_index);

        pcl::PointXYZRGB point;
        point.x = vertex_pos.x();
        point.y = vertex_pos.y();
        point.z = vertex_pos.z();
        point.r = vertex_color.r;
        point.g = vertex_color.g;
        point.b = vertex_color.b;
        cloud->push_back(point);
        if (!saved_label) {
          objects_attributes->ids.push_back(object_id);
          objects_attributes->labels.push_back(object_attrs.semantic_label);
          objects_attributes->features.push_back(object_attrs.semantic_feature);
          saved_label = true;
        }
      }
      cloud->width = cloud->size();
      cloud->height = 1;
      cloud->is_dense = true;

      for (size_t i = 0; i < dsg_->graph->mesh()->numFaces(); ++i) {
        const auto& face = dsg_->graph->mesh()->face(i);
        if (areElementsInSet(face, vertex_positions)) {
          pcl::Vertices vertices;
          vertices.vertices = {static_cast<uint32_t>(vertex_map[face[0]]),
                               static_cast<uint32_t>(vertex_map[face[1]]),
                               static_cast<uint32_t>(vertex_map[face[2]])};
          object_mesh->polygons.push_back(vertices);
        }
      }
      if (cloud->empty() || object_mesh->polygons.empty()) {
        objects_attributes->ids.pop_back();
        objects_attributes->meshes.pop_back();
        continue;
      }
      pcl::toPCLPointCloud2(*cloud, object_mesh->cloud);
      objects_attributes->meshes.push_back(object_mesh);
    }
  }
}

void UpdateReasoningFunctor::getObjectPointcloud(
    const SceneGraphNode& room,
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr object_cloud,
    std::vector<uint32_t>& instance_ids) const {
  // Iterate over the objects in the room, store a pointcloud of the objects
  uint32_t instance_id = 0;
  for (const auto& place_id : room.children()) {
    if (!dsg_->graph->getLayer(DsgLayers::PLACES).hasNode(place_id)) {
      continue;
    }
    const auto& place = dsg_->graph->getLayer(DsgLayers::PLACES).getNode(place_id);
    for (const auto& object_id : place.children()) {
      if (!dsg_->graph->getLayer(DsgLayers::OBJECTS).hasNode(object_id)) {
        continue;
      }
      const auto& object = dsg_->graph->getLayer(DsgLayers::OBJECTS).getNode(object_id);
      const auto& object_attrs = object.attributes<ObjectNodeAttributes>();
      const auto& object_mesh_conections = object_attrs.mesh_connections;
      if (object_mesh_conections.empty()) {
        continue;
      }
      for (const auto& vertex_index : object_mesh_conections) {
        if (vertex_index >= dsg_->graph->mesh()->numVertices()) {
          continue;
        }
        const auto vertex_pos = dsg_->graph->mesh()->pos(vertex_index).cast<double>();
        const auto vertex_color = dsg_->graph->mesh()->color(vertex_index);

        pcl::PointXYZRGB point;
        point.x = vertex_pos.x();
        point.y = vertex_pos.y();
        point.z = vertex_pos.z();
        point.r = vertex_color.r;
        point.g = vertex_color.g;
        point.b = vertex_color.b;
        object_cloud->push_back(point);
        instance_ids.emplace_back(instance_id);
      }
      ++instance_id;
    }
  }
}

void UpdateReasoningFunctor::getEdgeIndices(
    ObjectsAttributes::Ptr& objects_attributes) const {
  if (objects_attributes->meshes.empty()) {
    return;
  }
  // Get meshes centroids
  pcl::PointCloud<pcl::PointXYZ>::Ptr mesh_centroids(
      new pcl::PointCloud<pcl::PointXYZ>);
  for (const auto& mesh : objects_attributes->meshes) {
    pcl::PointCloud<pcl::PointXYZ> mesh_cloud;
    pcl::fromPCLPointCloud2(mesh->cloud, mesh_cloud);
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(mesh_cloud, centroid);
    pcl::PointXYZ point;
    point.x = centroid[0];
    point.y = centroid[1];
    point.z = centroid[2];
    mesh_centroids->push_back(point);
  }

  // Build a kdtree
  object_centroids_tree_->setInputCloud(mesh_centroids);
  for (size_t i = 0; i < objects_attributes->meshes.size(); ++i) {
    std::vector<int> indices;
    std::vector<float> distances;
    object_centroids_tree_->radiusSearch(
        mesh_centroids->at(i), config_.edges_max_radius, indices, distances);
    if (indices.size() < 2) {
      continue;
    }
    for (size_t j = 1; j < indices.size(); ++j) {
      objects_attributes->edge_indices.push_back(
          std::make_pair(i, static_cast<size_t>(indices[j])));
    }
  }
}
}  // namespace hydra
