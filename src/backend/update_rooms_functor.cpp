/* -----------------------------------------------------------------------------
 * Copyright 2022 Massachusetts Institute of Technology.
 * All Rights Reserved
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Research was sponsored by the United States Air Force Research Laboratory and
 * the United States Air Force Artificial Intelligence Accelerator and was
 * accomplished under Cooperative Agreement Number FA8750-19-2-1000. The views
 * and conclusions contained in this document are those of the authors and should
 * not be interpreted as representing the official policies, either expressed or
 * implied, of the United States Air Force or the U.S. Government. The U.S.
 * Government is authorized to reproduce and distribute reprints for Government
 * purposes notwithstanding any copyright notation herein.
 * -------------------------------------------------------------------------- */
#include "hydra/backend/update_rooms_functor.h"

#include <config_utilities/config.h>
#include <config_utilities/validation.h>
#include <glog/logging.h>

#include "hydra/common/global_info.h"
#include "hydra/utils/timing_utilities.h"

namespace hydra {

using timing::ScopedTimer;
using SemanticLabel = SemanticNodeAttributes::Label;

void declare_config(UpdateRoomsFunctor::Config& config) {
  using namespace config;
  name("UpdateRoomsFunctor::Config");
  field(config.room_finder, "room_finder");
  field(config.kmeans, "kmeans");
  field(config.places_layer, "places_layer");
}

UpdateRoomsFunctor::UpdateRoomsFunctor(const Config& config)
    : config(config::checkValid(config)),
      room_finder(new RoomFinder(config.room_finder)),
      kmeans(new KMeans<float>(config.kmeans)) {}

void UpdateRoomsFunctor::rewriteRooms(const SceneGraphLayer* new_rooms,
                                      DynamicSceneGraph& graph) const {
  std::vector<NodeId> to_remove;
  std::unordered_map<NodeId, std::string> old_room_labels;
  const auto& prev_rooms = graph.getLayer(DsgLayers::ROOMS);
  for (const auto& id_node_pair : prev_rooms.nodes()) {
    to_remove.push_back(id_node_pair.first);
    old_room_labels.emplace(
        id_node_pair.first,
        id_node_pair.second->attributes<RoomNodeAttributes>().label);
  }

  for (const auto node_id : to_remove) {
    graph.removeNode(node_id);
  }

  if (!new_rooms) {
    return;
  }

  for (auto&& [id, node] : new_rooms->nodes()) {
    // Preserve old labels if they exist
    if (old_room_labels.count(id) > 0) {
      auto& room_attrs = node->attributes<RoomNodeAttributes>();
      room_attrs.label = old_room_labels.at(id);
    }
    graph.emplaceNode(DsgLayers::ROOMS, id, node->attributes().clone());
  }

  for (const auto& id_edge_pair : new_rooms->edges()) {
    const auto& edge = id_edge_pair.second;
    graph.insertEdge(edge.source, edge.target, edge.info->clone());
  }
}

void UpdateRoomsFunctor::call(const DynamicSceneGraph&,
                              SharedDsgInfo& dsg,
                              const UpdateInfo::ConstPtr& info) {
  if (!room_finder) {
    return;
  }

  const auto places_layer = dsg.graph->findLayer(config.places_layer);
  if (!places_layer) {
    return;
  }

  ScopedTimer timer("backend/room_detection", info->timestamp_ns, true, 1, false);
  auto places_clone = places_layer->clone(
      [](const auto& node) { return NodeSymbol(node.id).category() == 'p'; });

  // TODO(nathan) layer view
  // TODO(nathan) pass in timestamp?
  auto rooms = room_finder->findRooms(*places_clone);
  rewriteRooms(rooms.get(), *dsg.graph);
  room_finder->addRoomPlaceEdges(*dsg.graph);

  connectRoomsToTraversability(dsg.graph);
  computeRoomFeatures(dsg.graph, rooms.get(), info->feature);

  return;
}

void UpdateRoomsFunctor::computeRoomFeatures(
    DynamicSceneGraph::Ptr& graph,
    const SceneGraphLayer* new_rooms,
    const std::optional<FeatureVector>& feature) const {
  if (!kmeans || !feature) {
    return;
  }

  const auto layer_id = graph->getLayerKey(spark_dsg::DsgLayers::AGENTS);
  if (!layer_id) {
    return;
  }
  const auto& prefix = GlobalInfo::instance().getRobotPrefix();
  const auto layer = graph->findLayer(layer_id->layer, prefix.key);
  if (!layer) {
    LOG(WARNING) << "Missing layer '" << spark_dsg::DsgLayers::AGENTS
                 << "' and partition " << prefix.key << " in DSG!";
    return;
  }
  if (layer->numNodes() == 0) {
    LOG(WARNING) << "No nodes in layer '" << spark_dsg::DsgLayers::AGENTS
                 << "' and partition " << prefix.key << " in DSG!";
    return;
  }

  // Get last inserted agent node --> last iterator of the map
  NodeSymbol node_id(prefix.key, layer->numNodes() - 1);
  auto& current_agent_node = layer->getNode(node_id);
  auto& current_agent_attrs = current_agent_node.attributes<AgentNodeAttributes>();
  current_agent_attrs.image_feature = *feature;

  if (!new_rooms) {
    return;
  }
  std::unordered_map<NodeId, std::vector<FeatureVector>> room_features;
  room_features.reserve(new_rooms->numNodes());

  for (const auto& [agent_id, agent] : layer->nodes()) {
    const auto& agent_attrs = agent->attributes<AgentNodeAttributes>();
    if (agent_attrs.image_feature.size() == 0) {
      VLOG(2) << "Agent node " << agent_id
              << " has no image feature, cannot compute room features!";
      continue;
    }
    const auto& place_id = agent->getParent();
    if (!place_id) {
      VLOG(2) << "Agent node " << agent_id
              << " has no parent place, cannot compute room features!";
      continue;
    }
    const auto& place = graph->getNode(*place_id);
    const auto& room_id = place.getParent();
    if (!room_id) {
      VLOG(2) << "Place node " << *place_id
              << " has no parent room, cannot compute room features!";
      continue;
    }
    room_features[*room_id].push_back(agent_attrs.image_feature);
  }

  for (auto map_iter = room_features.begin(); map_iter != room_features.end();
       ++map_iter) {
    const auto& room_id = map_iter->first;
    const auto& features = map_iter->second;
    if (features.empty()) {
      continue;
    }
    auto& room_attrs = graph->getNode(room_id).attributes<RoomNodeAttributes>();
    if (features.size() <= kmeans->config.num_clusters) {
      room_attrs.features = features;
    } else {
      kmeans->cluster(features, room_attrs.features);
    }
  }
}

void UpdateRoomsFunctor::connectRoomsToTraversability(
    DynamicSceneGraph::Ptr& graph) const {
  if (!graph->hasLayer(DsgLayers::TRAVERSABILITY) ||
      !graph->hasLayer(DsgLayers::ROOMS) || !graph->hasLayer(config.places_layer)) {
    VLOG(1) << "Missing layer '" << DsgLayers::TRAVERSABILITY << "' or '"
            << DsgLayers::ROOMS << "' or '" << config.places_layer
            << "' in DSG, cannot connect rooms to traversability!";
    return;
  }
  const auto& places_layer = graph->getLayer(config.places_layer);
  if (places_layer.numNodes() == 0) {
    VLOG(1) << "No nodes in layer '" << config.places_layer
            << "' in DSG, cannot connect rooms to traversability!";
    return;
  }
  std::vector<NodeId> places_ids;
  places_ids.reserve(places_layer.numNodes());
  const auto& places_nodes = places_layer.nodes();
  for (const auto& [pid, _] : places_nodes) {
    places_ids.push_back(pid);
  }

  nn_finder_.reset(new NearestNodeFinder(places_layer, places_ids));

  const auto& tr_layer = graph->getLayer(DsgLayers::TRAVERSABILITY);
  for (const auto& [tr_node_id, tr_node] : tr_layer.nodes()) {
    const auto& tr_attrs = tr_node->attributes<TraversabilityNodeAttributes>();
    nn_finder_->find(tr_attrs.position, 1, false, [&](NodeId pid, size_t, double) {
      const auto& place_node = graph->getNode(pid);
      const auto room_id = place_node.getParent();
      if (room_id.has_value()) {
        graph->insertEdge(room_id.value(), tr_node_id, nullptr, true);
      }
    });
  }
}

}  // namespace hydra
