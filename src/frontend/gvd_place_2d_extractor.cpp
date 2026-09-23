/* -----------------------------------------------------------------------------
 * BSD 3-Clause License
 *
 * Copyright (c) 2026, NTNU Autonomous Robots Lab
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * -------------------------------------------------------------------------- */
#include "hydra/frontend/gvd_place_2d_extractor.h"

#include <config_utilities/config.h>
#include <config_utilities/validation.h>
#include <spark_dsg/node_attributes.h>
#include <spark_dsg/node_symbol.h>

using namespace spark_dsg;

namespace hydra {

inline std::pair<NodeId, NodeId> makeEdge(NodeId a, NodeId b) {
  return (a < b) ? std::make_pair(a, b) : std::make_pair(b, a);
}

void declare_config(GvdPlace2DExtractor::Config& config) {
  using namespace config;
  name("GvdPlace2DExtractorConfig");
  field(config.layer, "layer");
  field(config.extractor_config, "extractor_config");
  field(config.sinks, "sinks");
}

GvdPlace2DExtractor::GvdPlace2DExtractor(const Config& config)
    : config(config::checkValid(config)),
      graph_extractor_(
          std::make_unique<voronoi::GraphExtractor>(config.extractor_config)),
      sinks_(Sink::instantiate(config.sinks)) {}

void GvdPlace2DExtractor::detect(const ActiveWindowOutput& msg) {
  graph_extractor_->extract(msg.occupancy_grid);
  world_T_body_ = msg.world_T_body();
  if (sinks_.empty()) {
    return;
  }
  std::vector<Eigen::Vector3d> vertices;
  std::vector<std::pair<size_t, size_t>> edges;
  const auto& origin = graph_extractor_->getOrigin();
  const auto& resolution = msg.occupancy_grid->resolution;
  for (const auto& segment : graph_extractor_->getSegments()) {
    const auto& path = segment.getPath();
    const auto& source = path.front() * resolution + origin;
    const auto& target = path.back() * resolution + origin;
    vertices.push_back(Eigen::Vector3d(source.x(), source.y(), 0.0));
    vertices.push_back(Eigen::Vector3d(target.x(), target.y(), 0.0));
    edges.emplace_back(vertices.size() - 2, vertices.size() - 1);
  }
  Sink::callAll(sinks_, msg.timestamp_ns, vertices, edges);
}

void GvdPlace2DExtractor::updateGraph(const ActiveWindowOutput& msg,
                                      DynamicSceneGraph& graph) {
  if (!graph.hasLayer(config.layer)) {
    return;
  }

  const auto& places_graph = graph_extractor_->getGraph();

  std::set<NodeId> new_nodes;
  for (const auto& [pose, node_id] : places_graph->nodes) {
    auto attrs = std::make_unique<TraversabilityNodeAttributes>();
    attrs->position =
        Eigen::Vector3d(pose.x(), pose.y(), world_T_body_.translation().z());
    attrs->is_active = true;
    attrs->is_predicted = false;
    attrs->last_update_time_ns = msg.timestamp_ns;

    graph.addOrUpdateNode(config.layer, node_id.id, std::move(attrs));
    new_nodes.insert(node_id.id);
  }

  for (const auto& old_node : old_nodes_) {
    if (new_nodes.count(old_node) == 0) {
      graph.removeNode(old_node);
    }
  }
  old_nodes_ = std::move(new_nodes);

  std::set<std::pair<NodeId, NodeId>> new_edges;
  for (const auto& edge : places_graph->edges) {
    auto edge_attrs = std::make_unique<EdgeAttributes>(edge.weight);
    edge_attrs->weighted = true;
    graph.addOrUpdateEdge(edge.source, edge.target, std::move(edge_attrs));
    new_edges.insert(makeEdge(edge.source, edge.target));
  }

  for (const auto& old_edge : old_edges_) {
    if (new_edges.count(old_edge) == 0) {
      graph.removeEdge(old_edge.first, old_edge.second);
      graph.removeEdge(old_edge.second, old_edge.first);
    }
  }
  old_edges_ = std::move(new_edges);
}

}  // namespace hydra
