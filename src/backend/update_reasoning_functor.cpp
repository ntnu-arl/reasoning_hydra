#include "hydra/backend/update_reasoning_functor.h"

namespace hydra {

UpdateReasoningFunctor::UpdateReasoningFunctor(const ThreeDSSGConfig& config,
                                               SharedModuleState::Ptr& state)
    : config_(config),
      state_(state),
      reasoning_(std::make_unique<Reasoning>(config.reasoning)) {
  // Make input and output directories if they don't exist
  std::filesystem::create_directories(
      std::filesystem::path(config_.reasoning.input_folder));
  std::filesystem::create_directories(
      std::filesystem::path(config_.reasoning.output_folder));
  // Assert that the inference script directory exists
  CHECK(std::filesystem::exists(config_.reasoning.method_inference_script_dir))
      << "Inference script directory does not exist: "
      << config_.reasoning.method_inference_script_dir;
}

MergeList UpdateReasoningFunctor::call(const DynamicSceneGraph&,
                                       SharedDsgInfo& dsg,
                                       const UpdateInfo::ConstPtr&) {
  if (dsg.graph->getLayer(DsgLayers::ROOMS).numNodes() == 0 ||
      state_->latest_places.empty()) {
    return {};
  }

  // Initialize previous room node id to the first room that we have seen
  if (!initialized_) {
    const auto& latest_place_id = *state_->latest_places.begin();
    assert(dsg.graph->hasNode(latest_place_id));
    if (!dsg.graph->getNode(latest_place_id).hasParent()) {
      return {};
    }
    prev_room_node_id_ = *(dsg.graph->getNode(latest_place_id).parents().begin());
    initialized_ = true;
    return {};
  }
  // Detect if the room has changed
  NodeId room_to_reason_id;
  if (!detectRoomChange(room_to_reason_id, dsg)) {
    return {};
  }

  // Get the room to reason about
  if (!dsg.graph->getLayer(DsgLayers::ROOMS).hasNode(room_to_reason_id)) {
    return {};
  }
  const auto& room_to_reason =
      dsg.graph->getLayer(DsgLayers::ROOMS).getNode(room_to_reason_id);

  // Get the object meshes and labels
  std::vector<pcl::PolygonMesh::Ptr> object_meshes;
  std::vector<uint32_t> mesh_labels;
  std::vector<NodeId> object_ids;
  getObjectMeshes(dsg, room_to_reason, object_ids, object_meshes, mesh_labels);

  // Save meshes and labels to file
  std::filesystem::path object_mesh_path =
      std::filesystem::path(config_.reasoning.input_folder) /
      room_to_reason.attributes<SemanticNodeAttributes>().name;
  std::filesystem::create_directories(object_mesh_path);
  for (size_t i = 0; i < object_meshes.size(); ++i) {
    std::string filename = (object_mesh_path / (std::to_string(i) + ".ply")).string();
    pcl::io::savePLYFile(filename, *object_meshes[i]);
  }
  saveVectorToBinary(mesh_labels, (object_mesh_path / "labels").string());

  // Run the reasoning script
  std::map<std::pair<NodeId, NodeId>, std::vector<std::string>> reasoning_edges;
  if (!reasoning_->run(object_ids,
                       reasoning_edges,
                       room_to_reason.attributes<SemanticNodeAttributes>().name)) {
    return {};
  }

  return {};
}

bool UpdateReasoningFunctor::detectRoomChange(NodeId& room_to_reason_id,
                                              const SharedDsgInfo& dsg) {
  // Get place nodes that have parents (i.e. are in rooms)
  std::vector<NodeId> latest_places_vec;
  std::vector<Eigen::Vector3f> place_centroids;
  const auto& places = dsg.graph->getLayer(DsgLayers::PLACES).nodes();

  for (const auto& [place_id, place] : places) {
    if (NodeSymbol(place_id).category() == 'p' && place->hasParent()) {
      place_centroids.emplace_back(
          place->attributes<NodeAttributes>().position.cast<float>());
      latest_places_vec.emplace_back(place_id);
    }
  }

  if (place_centroids.empty()) {
    return false;
  }

  if (!neighbor_search_) {
    neighbor_search_ = std::make_unique<PointNeighborSearch>(place_centroids);
  } else {
    neighbor_search_.reset(new PointNeighborSearch(place_centroids));
  }

  // Find the closest place to the current pose
  size_t closest_place_id;
  float distance_squared;
  const auto current_pose =
      dsg.graph->dynamicLayersOfType(DsgLayers::AGENTS)
          .begin()
          ->second
          ->getPositionByIndex(dsg.graph->dynamicLayersOfType(DsgLayers::AGENTS)
                                   .begin()
                                   ->second->numNodes() -
                               1)
          .cast<float>();
  bool nn_success =
      neighbor_search_->search(current_pose, distance_squared, closest_place_id);

  if (!nn_success) {
    return false;
  }

  const auto& closest_place_node_id = latest_places_vec[closest_place_id];
  NodeId closest_room_node_id = *(places.at(closest_place_node_id)->parents().begin());

  // If the closest place is in the same room as we were previously, then we haven't
  // changed rooms
  if (closest_room_node_id == prev_room_node_id_) {
    return false;
  }

  room_to_reason_id = prev_room_node_id_;
  prev_room_node_id_ = closest_room_node_id;

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
    const SharedDsgInfo& dsg,
    const SceneGraphNode& room,
    std::vector<NodeId>& object_ids,
    std::vector<pcl::PolygonMesh::Ptr>& object_meshes,
    std::vector<uint32_t>& mesh_labels) const {
  // Iterate over the objects in the room, store a pointcloud of the objects
  for (const auto& place_id : room.children()) {
    if (!dsg.graph->getLayer(DsgLayers::PLACES).hasNode(place_id)) {
      continue;
    }
    const auto& place = dsg.graph->getLayer(DsgLayers::PLACES).getNode(place_id);
    for (const auto& object_id : place.children()) {
      if (!dsg.graph->getLayer(DsgLayers::OBJECTS).hasNode(object_id)) {
        continue;
      }
      const auto& object = dsg.graph->getLayer(DsgLayers::OBJECTS).getNode(object_id);
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
        if (vertex_index >= dsg.graph->mesh()->numVertices()) {
          continue;
        }
        const auto vertex_pos = dsg.graph->mesh()->pos(vertex_index).cast<double>();
        vertex_positions.insert(vertex_index);
        vertex_map[vertex_index] = j;
        j++;
        const auto vertex_color = dsg.graph->mesh()->color(vertex_index);

        pcl::PointXYZRGB point;
        point.x = vertex_pos.x();
        point.y = vertex_pos.y();
        point.z = vertex_pos.z();
        point.r = vertex_color.r;
        point.g = vertex_color.g;
        point.b = vertex_color.b;
        cloud->push_back(point);
        if (!saved_label) {
          object_ids.emplace_back(object_id);
          mesh_labels.emplace_back(object_attrs.semantic_label);
          saved_label = true;
        }
      }
      cloud->width = cloud->size();
      cloud->height = 1;
      cloud->is_dense = true;

      for (size_t i = 0; i < dsg.graph->mesh()->numFaces(); ++i) {
        const auto& face = dsg.graph->mesh()->face(i);
        if (areElementsInSet(face, vertex_positions)) {
          pcl::Vertices vertices;
          vertices.vertices = {
              vertex_map[face[0]], vertex_map[face[1]], vertex_map[face[2]]};
          object_mesh->polygons.push_back(vertices);
        }
      }
      pcl::toPCLPointCloud2(*cloud, object_mesh->cloud);
      object_meshes.emplace_back(object_mesh);
    }
  }
}

void UpdateReasoningFunctor::getObjectPointcloud(
    const SharedDsgInfo& dsg,
    const SceneGraphNode& room,
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr object_cloud,
    std::vector<uint32_t>& instance_ids) const {
  // Iterate over the objects in the room, store a pointcloud of the objects
  uint32_t instance_id = 0;
  for (const auto& place_id : room.children()) {
    if (!dsg.graph->getLayer(DsgLayers::PLACES).hasNode(place_id)) {
      continue;
    }
    const auto& place = dsg.graph->getLayer(DsgLayers::PLACES).getNode(place_id);
    for (const auto& object_id : place.children()) {
      if (!dsg.graph->getLayer(DsgLayers::OBJECTS).hasNode(object_id)) {
        continue;
      }
      const auto& object = dsg.graph->getLayer(DsgLayers::OBJECTS).getNode(object_id);
      const auto& object_attrs = object.attributes<ObjectNodeAttributes>();
      const auto& object_mesh_conections = object_attrs.mesh_connections;
      if (object_mesh_conections.empty()) {
        continue;
      }
      for (const auto& vertex_index : object_mesh_conections) {
        if (vertex_index >= dsg.graph->mesh()->numVertices()) {
          continue;
        }
        const auto vertex_pos = dsg.graph->mesh()->pos(vertex_index).cast<double>();
        const auto vertex_color = dsg.graph->mesh()->color(vertex_index);

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
}  // namespace hydra
