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
#include "hydra/frontend/traversability_place_extractor.h"

#include <config_utilities/config.h>
#include <config_utilities/types/conversions.h>
#include <config_utilities/types/enum.h>
#include <config_utilities/validation.h>
#include <glog/logging.h>

#include <memory>

#include "hydra/common/global_info.h"
#include "hydra/utils/timing_utilities.h"

using Timer = hydra::timing::ScopedTimer;

namespace hydra {

void declare_config(TraversabilityPlaceExtractor::Config& config) {
  using namespace config;
  name("TraversabilityPlaceExtractor::Config");
  field(config.estimator, "estimator");
  field(config.clustering, "clustering");
  field(config.postprocessing, "postprocessing");
  field(config.traversability_sinks, "traversability_sinks");
  field(config.sinks, "sinks");
}

TraversabilityPlaceExtractor::TraversabilityPlaceExtractor(const Config& config)
    : config(config::checkValid(config)),
      estimator_(config.estimator.create()),
      clustering_(config.clustering.create()),
      postprocessing_(config.postprocessing),
      traversability_sinks_(
          TraversabilitySink::instantiate(config.traversability_sinks)),
      sinks_(Sink::instantiate(config.sinks)) {}

void TraversabilityPlaceExtractor::detect(const ActiveWindowOutput& msg) {
  Timer timer("traversability/estimate", msg.timestamp_ns);
  estimator_->updateTraversability(msg);
}

void TraversabilityPlaceExtractor::updateGraph(const ActiveWindowOutput& msg,
                                               DynamicSceneGraph& graph) {
  // TODO(lschmid): Find a nicer way than copying the layer here. Should not be too
  // expensive though.
  auto timer = Timer("traversability/postprocessing", msg.timestamp_ns);
  places::TraversabilityLayer layer = estimator_->getTraversabilityLayer();
  postprocessing_.apply(layer);

  timer.reset("traversability/clustering");
  clustering_->updateGraph(layer, msg, graph);

  timer.reset("traversability/sinks");
  TraversabilitySink::callAll(
      traversability_sinks_, msg.timestamp_ns, msg.world_t_body, layer);
  if (sinks_.empty()) {
    return;
  }
  std::vector<Eigen::Vector3d> vertices;
  std::vector<std::pair<size_t, size_t>> edges;
  const auto& trav_layer = graph.getLayer(DsgLayers::TRAVERSABILITY);
  for (const auto& [edge_key, _] : trav_layer.edges()) {
    const auto& source_node_position =
        trav_layer.getNode(edge_key.k1).attributes<NodeAttributes>().position;
    const auto& target_node_position =
        trav_layer.getNode(edge_key.k2).attributes<NodeAttributes>().position;
    vertices.push_back(source_node_position);
    vertices.push_back(target_node_position);
    edges.emplace_back(vertices.size() - 2, vertices.size() - 1);
  }
  Sink::callAll(sinks_, msg.timestamp_ns, vertices, edges);
}

}  // namespace hydra
