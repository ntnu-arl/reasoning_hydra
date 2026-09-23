// Portions of the following code and their modifications are originally from
// https://github.com/MIT-SPARK/Hydra/tree/main and are licensed under the following
// license:
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

// Copyright (c) 2025, Autonomous Robots Lab, Norwegian University of Science and
// Technology All rights reserved.

// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
#include "hydra/active_window/reconstruction_module.h"

#include <config_utilities/config.h>
#include <config_utilities/printing.h>
#include <config_utilities/validation.h>

#include <chrono>
#include <iomanip>

#include "hydra/common/global_info.h"
#include "hydra/input/input_conversion.h"
#include "hydra/places/robot_footprint_integrator.h"
#include "hydra/reconstruction/integration_masking.h"
#include "hydra/reconstruction/mesh_integrator.h"
#include "hydra/reconstruction/projective_integrator.h"
#include "hydra/utils/printing.h"
#include "hydra/utils/timing_utilities.h"

namespace hydra {
namespace {

static const auto registration =
    config::RegistrationWithConfig<ActiveWindowModule,
                                   ReconstructionModule,
                                   ReconstructionModule::Config,
                                   ActiveWindowModule::OutputQueue::Ptr>(
        "ReconstructionModule");

double diffInSeconds(uint64_t lhs, uint64_t rhs) {
  return std::chrono::duration_cast<std::chrono::duration<double>>(
             std::chrono::nanoseconds(lhs) - std::chrono::nanoseconds(rhs))
      .count();
}

std::string printRotation(const Eigen::Matrix3d& rot) {
  const Eigen::Quaterniond q(rot);
  std::stringstream ss;
  ss << std::setprecision(3) << "{w: " << q.w() << ", x: " << q.x() << ", y: " << q.y()
     << ", z: " << q.z() << "}";
  return ss.str();
}

}  // namespace

using timing::ScopedTimer;

void declare_config(ReconstructionModule::Config& config) {
  using namespace config;
  name("ReconstructionModule::Config");
  base<ActiveWindowModule::Config>(config);
  field(config.full_update_separation_s, "full_update_separation_s", "s");
  field(config.max_input_queue_size, "max_input_queue_size");
  field(config.tsdf, "tsdf");
  field(config.mesh, "mesh");
  config.robot_footprint.setOptional();
  field(config.robot_footprint, "robot_footprint");
  field(config.compute_occupancy, "compute_occupancy");
  field(config.occupancy, "occupancy");
}

ReconstructionModule::ReconstructionModule(const Config& config,
                                           const OutputQueue::Ptr& queue)
    : ActiveWindowModule(config, queue),
      config(config::checkValid(config)),
      last_update_ns_(std::nullopt),
      tsdf_integrator_(std::make_unique<ProjectiveIntegrator>(config.tsdf)),
      mesh_integrator_(std::make_unique<MeshIntegrator>(config.mesh)),
      footprint_integrator_(config.robot_footprint.create()) {
  if (config.tsdf.semantic_integrator && !map_.config.with_semantics) {
    LOG(ERROR)
        << "Semantic integrator specified but map does not contain semantic layer!";
  }
  if (config.compute_occupancy) {
    occupancy_integrator_ = std::make_unique<OccupancyIntegrator>(config.occupancy);
  }
}

ReconstructionModule::~ReconstructionModule() {}

std::string ReconstructionModule::printInfo() const {
  return config::toString(config) + "\n" + Sink::printSinks(sinks_);
}

bool ReconstructionModule::shouldUpdate(uint64_t timestamp_ns) const {
  if (!last_update_ns_) {
    return true;
  }

  const auto diff_s = diffInSeconds(timestamp_ns, last_update_ns_.value());
  return diff_s >= config.full_update_separation_s;
}

ActiveWindowOutput::Ptr ReconstructionModule::spinOnce(const InputPacket& msg) {
  if (!msg.sensor_input) {
    LOG(ERROR) << "[Hydra Reconstruction] received invalid sensor data in input!";
    return nullptr;
  }

  const auto timestamp_ns = msg.timestamp_ns;
  const auto world_T_body = msg.world_T_body();
  const auto fmt = getDefaultFormat();
  VLOG(5) << "[Hydra Reconstruction] Got input @ " << timestamp_ns
          << " [ns] with pose: p=" << world_T_body.translation().format(fmt)
          << ", q=" << printRotation(world_T_body.rotation());

  const auto do_full_update = shouldUpdate(timestamp_ns);

  VLOG(2) << "[Hydra Reconstruction] starting " << (do_full_update ? "full" : "partial")
          << " update for message @ " << timestamp_ns << " (" << input_queue_->size()
          << " message(s) left)";

  ScopedTimer timer("reconstruction/spin", timestamp_ns);
  // force semantic normalization if volumetric map has semantic layer
  InputData::Ptr data = conversions::parseInputPacket(msg, false, map_.hasSemantics());
  if (!data) {
    return nullptr;
  }

  // TODO(nathan) cache somewhere
  const auto label_config = GlobalInfo::instance().getLabelSpaceConfig();
  std::set<int32_t> invalid_labels;
  invalid_labels.insert(label_config.invalid_labels.begin(),
                        label_config.invalid_labels.end());
  invalid_labels.insert(label_config.dynamic_labels.begin(),
                        label_config.dynamic_labels.end());

  cv::Mat integration_mask;
  maskInvalidSemantics(data->label_image, invalid_labels, integration_mask);

  BlockIndices updated_blocks;
  {  // timing scope
    ScopedTimer timer("reconstruction/tsdf", timestamp_ns);
    updated_blocks = tsdf_integrator_->updateMap(*data, map_, true, integration_mask);
    if (footprint_integrator_) {
      footprint_integrator_->markFreespace(world_T_body.cast<float>(), map_);
    }
  }  // timing scope

  updated_blocks_.insert(updated_blocks.begin(), updated_blocks.end());

  auto& tsdf = map_.getTsdfLayer();
  if (tsdf.numBlocks() == 0 || !do_full_update) {
    return nullptr;
  }

  last_update_ns_ = timestamp_ns;
  {  // timing scope
    ScopedTimer timer("reconstruction/mesh", timestamp_ns);
    mesh_integrator_->generateMesh(map_, true, true);
  }  // timing scope

  auto output = ActiveWindowOutput::fromInput(msg);
  output->sensor_data = data;

  // this comes before clearing the update flag as we don't archive updated blocks
  if (map_window_) {
    output->archived_mesh_indices =
        map_window_->archiveBlocks(timestamp_ns, world_T_body, map_);
    VLOG(2) << "[Hydra Reconstruction] archived "
            << output->archived_mesh_indices.size() << " @ " << timestamp_ns << " [ns]";
  }

  output->setMap(map_.cloneUpdated());
  for (const auto& block : tsdf) {
    block.clearUpdated();
  }

  // Remove semantic features from map and mesh
  clearSemanticFeatures();

  // Update occupancy grid if enabled
  if (occupancy_integrator_) {
    ScopedTimer timer("reconstruction/occupancy", timestamp_ns);
    const auto& map = output->map();
    output->occupancy_grid = occupancy_integrator_->integrate(output->timestamp_ns,
                                                              map.getTsdfLayer(),
                                                              map.getSemanticLayer(),
                                                              output->world_T_body());
  }
  return output;
}

void ReconstructionModule::clearSemanticFeatures() {
  // Start with semantic layer
  for (const auto& idx : updated_blocks_) {
    auto blocks = map_.getBlock(idx);
    if (blocks.tsdf) {
      for (size_t i = 0; i < blocks.tsdf->numVoxels(); ++i) {
        auto voxels = blocks.getVoxels(i);
        if (voxels.semantic) {
          voxels.semantic->semantic_feature = std::nullopt;
          voxels.semantic->panoptic_id = 0;
          voxels.semantic->pixelwise_feature = std::nullopt;
        }
      }
    }
    // Continue with mesh layer
    auto mesh = map_.getMeshLayer().getBlockPtr(idx);
    if (!mesh) {
      continue;
    }
    if (mesh->has_semantic_features) {
      for (auto& semantic_feature : mesh->semantic_features) {
        semantic_feature = std::nullopt;
      }
    }
    if (mesh->has_panoptic_ids) {
      for (auto& panoptic_id : mesh->panoptic_ids) {
        panoptic_id = std::nullopt;
      }
    }
  }
  updated_blocks_.clear();
}

}  // namespace hydra
