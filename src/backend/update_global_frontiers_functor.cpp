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
#include "hydra/backend/update_global_frontiers_functor.h"

#include <config_utilities/config.h>
#include <config_utilities/validation.h>
#include <glog/logging.h>

namespace hydra {

void declare_config(UpdateGlobalFrontiersFunctor::Config& config) {
  using namespace config;
  name("UpdateGlobalFrontiersFunctor::Config");
  field(config.object_connection_max_distance, "object_connection_max_distance");
  field(config.nav_layer, "nav_layer");
}

UpdateGlobalFrontiersFunctor::UpdateGlobalFrontiersFunctor(const Config& config)
    : config(config::checkValid(config)),
      object_connection_max_distance_sq_(
          std::pow(config.object_connection_max_distance, 2)) {}

void UpdateGlobalFrontiersFunctor::call(const DynamicSceneGraph&,
                                        SharedDsgInfo& dsg,
                                        const UpdateInfo::ConstPtr&) {
  auto& graph = dsg.graph;
  if (!graph->hasLayer(DsgLayers::FRONTIERS) || !graph->hasLayer(DsgLayers::OBJECTS)) {
    VLOG(1) << "Missing required layers, skipping.";
    return;
  }
  const auto& frontiers_layer = graph->getLayer(DsgLayers::FRONTIERS);

  for (const auto& [fid, frontier_node] : frontiers_layer.nodes()) {
    // Find nearest nav node
    auto& attrs = frontier_node->attributes<GlobalFrontierNodeAttributes>();
    if (!graph->hasLayer(attrs.nav_layer)) {
      VLOG(1) << "Missing navigation layer " << attrs.nav_layer << ", skipping";
      return;
    }
    attrs.connected_objects.clear();
    std::vector<NodeId> nav_ids;
    const auto& nav_layer = graph->getLayer(attrs.nav_layer);
    for (const auto& [pid, _] : nav_layer.nodes()) {
      nav_ids.push_back(pid);
    }
    nn_finder_.reset(new NearestNodeFinder(nav_layer, nav_ids));
    nn_finder_->find(frontier_node->attributes().position,
                     1,
                     false,
                     [&](NodeId pid, size_t, double) {
                       attrs.connected_nav = pid;
                       if (attrs.use_nav_as_centroid) {
                         const auto& nav_node = nav_layer.getNode(pid);
                         const auto& nav_attrs =
                             nav_node.attributes<spark_dsg::NodeAttributes>();
                         attrs.position.x() = nav_attrs.position.x();
                         attrs.position.y() = nav_attrs.position.y();
                       }
                     });

    // Find nearby objects
    std::vector<NodeId> object_ids;
    const auto& objects_layer = graph->getLayer(DsgLayers::OBJECTS);
    for (const auto& [oid, _] : objects_layer.nodes()) {
      object_ids.push_back(oid);
    }
    nn_finder_.reset(new NearestNodeFinder(objects_layer, object_ids));
    nn_finder_->findRadius(
        frontier_node->attributes().position,
        object_connection_max_distance_sq_,
        false,
        [&](NodeId oid, size_t, double) { attrs.connected_objects.push_back(oid); });
  }
}

}  // namespace hydra
