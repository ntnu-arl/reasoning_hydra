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
#include "hydra/backend/update_agents_functor.h"

#include <config_utilities/config.h>
#include <glog/logging.h>
#include <gtsam/geometry/Pose3.h>
#include <kimera_pgmo/utils/common_functions.h>
#include <spark_dsg/printing.h>

#include <iomanip>

#include "hydra/common/global_info.h"
#include "hydra/utils/printing.h"
#include "hydra/utils/timing_utilities.h"

namespace hydra {

using timing::ScopedTimer;

namespace {

inline std::string toString(const Eigen::Quaterniond& q, const Eigen::Vector3d& p) {
  const auto fmt = getDefaultFormat(3);
  std::stringstream ss;
  ss << std::setprecision(3) << "q: {w=" << q.w() << ", x=" << q.x() << ", y=" << q.y()
     << ", z=" << q.z() << "}, t: " << p.format(fmt);
  return ss.str();
}

}  // namespace

void declare_config(UpdateAgentsFunctor::Config& config) {
  using namespace config;
  name("UpdateAgentsFunctor::Config");
  field(config.enable_agent_keyframes, "enable_agent_keyframes");
  field(config.cos_sim_thresh, "cos_sim_thresh");
  field(config.min_translation_m, "min_translation_m");
  field(config.min_rotation_deg, "min_rotation_deg");
}

UpdateAgentsFunctor::UpdateAgentsFunctor(const Config& config) : config(config) {}

void UpdateAgentsFunctor::call(const DynamicSceneGraph&,
                               SharedDsgInfo& dsg,
                               const UpdateInfo::ConstPtr& info) {
  updateAgentKeyframes(dsg, info);
  if (!info->pgmo_values || info->pgmo_values->size() == 0) {
    return;
  }

  ScopedTimer timer("backend/agent_update", info->timestamp_ns, true, 1, false);
  auto& graph = *dsg.graph;
  const auto desired_layer = graph.getLayerKey(DsgLayers::AGENTS)->layer;
  for (const auto& [prefix, layer] : graph.layer_partition(desired_layer)) {
    std::set<NodeId> missing_nodes;
    for (const auto& [node_id, node] : layer->nodes()) {
      auto& attrs = node->attributes<AgentNodeAttributes>();
      if (!info->pgmo_values->exists(attrs.external_key)) {
        missing_nodes.insert(node->id);
        continue;
      }

      const auto p_prev = attrs.position;
      const auto q_prev = attrs.world_R_body;
      const gtsam::Pose3 prev_pose(gtsam::Rot3(q_prev), p_prev);
      auto pose = info->pgmo_values->at<gtsam::Pose3>(attrs.external_key);
      attrs.position = pose.translation();
      attrs.world_R_body = Eigen::Quaterniond(pose.rotation().matrix());

      const auto diff = prev_pose.between(pose);
      const auto q_diff = Eigen::Quaterniond(diff.rotation().matrix());
      const auto p_diff = diff.translation();
      VLOG(10) << "Updating agent " << NodeSymbol(node->id).str() << " pose from "
               << NodeSymbol(attrs.external_key).str() << ":"
               << "\n - original: " << toString(q_prev, p_prev)
               << "\n - new:      " << toString(attrs.world_R_body, attrs.position)
               << "\n - diff:     " << toString(q_diff, p_diff);
    }

    if (!missing_nodes.empty()) {
      LOG(WARNING) << "Layer " << DsgLayers::AGENTS << "(" << prefix
                   << "): could not update "
                   << displayNodeSymbolContainer(missing_nodes);
    }
  }
}

void UpdateAgentsFunctor::updateAgentKeyframes(SharedDsgInfo& dsg,
                                               const UpdateInfo::ConstPtr& info) {
  if (!config.enable_agent_keyframes) {
    return;
  }
  auto& graph = *dsg.graph;
  const auto desired_layer = graph.getLayerKey(DsgLayers::AGENTS);
  if (!desired_layer) {
    LOG(WARNING) << "[UpdateAgentsFunctor] No AGENTS layer found in DSG!";
    return;
  }
  const auto& prefix = GlobalInfo::instance().getRobotPrefix();
  if (!graph.hasLayer(desired_layer.value().layer, prefix.key)) {
    LOG(WARNING) << "[UpdateAgentsFunctor] No agent partition '" << prefix.key
                 << "' found in DSG!";
    return;
  }
  const auto& agent_layer = graph.getLayer(desired_layer.value().layer, prefix.key);
  if (agent_layer.numNodes() < 2) {
    LOG(WARNING) << "[UpdateAgentsFunctor] No agent nodes found in DSG!";
    return;
  }
  NodeSymbol latest_node_id(prefix.key, agent_layer.numNodes() - 1);
  auto& latest_attrs = graph.getNode(latest_node_id).attributes<AgentNodeAttributes>();
  latest_attrs.image_feature = info->feature.value_or(FeatureVector());

  if (!latest_attrs.validFeatures()) {
    LOG(WARNING) << "[UpdateAgentsFunctor] Latest agent node has empty feature vector!";
    return;
  }

  // Reset last keyframe if the latest node has a different parent room node than the
  // last keyframe
  if (last_keyframe_id_) {
    const auto& last_keyframe_node = graph.getNode(last_keyframe_id_.value());
    const auto& latest_node = graph.getNode(latest_node_id);
    if (last_keyframe_node.hasParent() && latest_node.hasParent()) {
      const auto& last_nav_node = graph.getNode(last_keyframe_node.getParent().value());
      const auto& latest_nav_node = graph.getNode(latest_node.getParent().value());
      if (last_nav_node.hasParent() && latest_nav_node.hasParent()) {
        if (last_nav_node.getParent().value() != latest_nav_node.getParent().value()) {
          last_keyframe_id_ = std::nullopt;
        }
      }
    }
  }

  if (!last_keyframe_id_) {
    // First keyframe --> last agent node if it exists
    latest_attrs.image = info->input_image.clone();
    last_keyframe_id_ = latest_node_id;
    VLOG(1) << "[UpdateAgentsFunctor] Initialized last keyframe to "
            << last_keyframe_id_.value().str();
  } else {
    // Check if we need to add a new keyframe
    const auto& last_keyframe_node = graph.getNode(last_keyframe_id_.value());
    const auto& last_attrs = last_keyframe_node.attributes<AgentNodeAttributes>();
    const gtsam::Pose3 last_pose(gtsam::Rot3(last_attrs.world_R_body),
                                 last_attrs.position);

    const gtsam::Pose3 latest_pose(gtsam::Rot3(latest_attrs.world_R_body),
                                   latest_attrs.position);

    const auto diff = last_pose.between(latest_pose);
    const double translation_m = diff.translation().norm();
    const double rotation_deg = diff.rotation().axisAngle().second * (180.0 / M_PI);

    VLOG(1) << "[UpdateAgentsFunctor] Agent keyframe check: translation = "
            << translation_m << " m, rotation = " << rotation_deg << " deg";

    if (translation_m < config.min_translation_m &&
        rotation_deg < config.min_rotation_deg) {
      return;
    }

    if (!latest_attrs.validFeatures()) {
      LOG(WARNING) << "[UpdateAgentsFunctor] Latest agent node missing valid features, "
                      "skipping keyframe check";
      return;
    }
    const auto cos_sim = latest_attrs.featureDistance(last_attrs.image_feature);
    VLOG(1) << "[UpdateAgentsFunctor] Agent keyframe check: cosine similarity = "
            << cos_sim;
    if (cos_sim >= config.cos_sim_thresh) {
      return;
    }

    // Add new keyframe
    latest_attrs.image = info->input_image.clone();
    last_keyframe_id_ = latest_node_id;
    VLOG(1) << "[UpdateAgentsFunctor] Added new agent keyframe at "
            << last_keyframe_id_.value().str();
  }
}
}  // namespace hydra
