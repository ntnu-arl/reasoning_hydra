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
#include "hydra/reconstruction/reconstruction_module.h"

#include <config_utilities/config.h>
#include <config_utilities/printing.h>
#include <config_utilities/types/conversions.h>
#include <config_utilities/types/eigen_matrix.h>
#include <config_utilities/validation.h>
#include <kimera_pgmo/compression/block_compression.h>
#include <kimera_pgmo/compression/delta_compression.h>
#include <kimera_pgmo/utils/common_functions.h>
#include <kimera_pgmo/utils/mesh_io.h>

#include <opencv2/opencv.hpp>

#include "hydra/common/global_info.h"
#include "hydra/input/input_conversion.h"
#include "hydra/reconstruction/mesh_integrator.h"
#include "hydra/reconstruction/projective_integrator.h"
#include "hydra/utils/pgmo_mesh_traits.h"
#include "hydra/utils/timing_utilities.h"

namespace hydra {

using timing::ScopedTimer;

void declare_config(ReconstructionModule::Config& conf) {
  using namespace config;
  name("ReconstructionConfig");
  field(conf.show_stats, "show_stats");
  field(conf.stats_verbosity, "stats_verbosity");
  field(conf.clear_distant_blocks, "clear_distant_blocks");
  field(conf.dense_representation_radius_m, "dense_representation_radius_m");
  field(conf.num_poses_per_update, "num_poses_per_update");
  field(conf.max_input_queue_size, "max_input_queue_size");
  field(conf.semantic_measurement_probability, "semantic_measurement_probability");
  field(conf.tsdf, "tsdf");
  field(conf.mesh, "mesh");
  conf.robot_footprint.setOptional();
  field(conf.robot_footprint, "robot_footprint");
  field(conf.sinks, "sinks");
}

Mesh::Ptr activeMesh(const MeshLayer& mesh_layer, const BlockIndices& archived_blocks) {
  auto active_mesh =
      std::make_shared<spark_dsg::Mesh>(true, false, true, true, true, false);
  const BlockIndexSet archived_set(archived_blocks.begin(), archived_blocks.end());
  size_t num_points = 0;
  for (const auto& block : mesh_layer.updatedBlockIndices()) {
    if (archived_set.count(block)) {
      continue;
    }
    auto& block_data = mesh_layer.getBlock(block);
    active_mesh->points.insert(
        active_mesh->points.end(), block_data.points.begin(), block_data.points.end());
    active_mesh->colors.insert(
        active_mesh->colors.end(), block_data.colors.begin(), block_data.colors.end());
    active_mesh->stamps.insert(
        active_mesh->stamps.end(), block_data.stamps.begin(), block_data.stamps.end());
    active_mesh->first_seen_stamps.insert(active_mesh->first_seen_stamps.end(),
                                          block_data.first_seen_stamps.begin(),
                                          block_data.first_seen_stamps.end());
    active_mesh->labels.insert(
        active_mesh->labels.end(), block_data.labels.begin(), block_data.labels.end());
    active_mesh->semantic_features.insert(active_mesh->semantic_features.end(),
                                          block_data.semantic_features.begin(),
                                          block_data.semantic_features.end());
    active_mesh->panoptic_ids.insert(active_mesh->panoptic_ids.end(),
                                     block_data.panoptic_ids.begin(),
                                     block_data.panoptic_ids.end());
    // Remap face indices.
    for (auto face : block_data.faces) {
      face[0] += num_points;
      face[1] += num_points;
      face[2] += num_points;
      active_mesh->faces.push_back(face);
    }
    num_points += block_data.points.size();
  }
  return active_mesh;
}

ReconstructionModule::ReconstructionModule(const Config& config,
                                           const OutputQueue::Ptr& queue)
    : config(config::checkValid(config)),
      num_poses_received_(0),
      output_queue_(queue),
      sinks_(Sink::instantiate(config.sinks)) {
  queue_.reset(new InputPacketQueue());
  queue_->max_size = config.max_input_queue_size;

  map_.reset(new VolumetricMap(GlobalInfo::instance().getMapConfig(), true));
  tsdf_integrator_ = std::make_unique<ProjectiveIntegrator>(config.tsdf);
  mesh_integrator_ = std::make_unique<MeshIntegrator>(config.mesh);
  footprint_integrator_ = config.robot_footprint.create();
}

ReconstructionModule::~ReconstructionModule() { stop(); }

void ReconstructionModule::start() {
  spin_thread_.reset(new std::thread(&ReconstructionModule::spin, this));
  LOG(INFO) << "[Hydra Reconstruction] started!";
}

void ReconstructionModule::stop() {
  should_shutdown_ = true;

  if (spin_thread_) {
    VLOG(2) << "[Hydra Reconstruction] stopping reconstruction!";
    spin_thread_->join();
    spin_thread_.reset();
    VLOG(2) << "[Hydra Reconstruction] stopped!";
  }

  VLOG(2) << "[Hydra Reconstruction] input queue: " << queue_->size();
  if (output_queue_) {
    VLOG(2) << "[Hydra Reconstruction] output queue: " << output_queue_->size();
  } else {
    VLOG(2) << "[Hydra Reconstruction] output queue: n/a";
  }
}

void ReconstructionModule::save(const LogSetup&) {}

std::string ReconstructionModule::printInfo() const {
  std::stringstream ss;
  ss << std::endl << config::toString(config);
  return ss.str();
}

void ReconstructionModule::spin() {
  // TODO(nathan) fix shutdown logic
  while (!should_shutdown_) {
    bool has_data = queue_->poll();
    if (!has_data) {
      continue;
    }

    spinOnce(*queue_->front());
    queue_->pop();
  }
}

bool ReconstructionModule::spinOnce() {
  bool has_data = queue_->poll();
  if (!has_data) {
    return false;
  }

  const auto success = spinOnce(*queue_->front());
  queue_->pop();
  return success;
}

bool ReconstructionModule::spinOnce(const InputPacket& msg) {
  if (!msg.sensor_input) {
    LOG(ERROR) << "[Hydra Reconstruction] received invalid sensor data in input!";
    return false;
  }

  ScopedTimer timer("places/spin", msg.timestamp_ns);
  VLOG(2) << "[Hydra Reconstruction]: Processing msg @ " << msg.timestamp_ns;
  VLOG(2) << "[Hydra Reconstruction]: " << queue_->size() << " message(s) left";

  ++num_poses_received_;
  const bool do_full_update = (num_poses_received_ % config.num_poses_per_update == 0);
  update(msg, do_full_update);
  return do_full_update;
}

void ReconstructionModule::addSink(const Sink::Ptr& sink) {
  if (sink) {
    sinks_.push_back(sink);
  }
}

void ReconstructionModule::fillOutput(ReconstructionOutput& msg) {
  // TODO(nathan) figure out a better way to handle repeated timestamps
  size_t ts = msg.timestamp_ns;
  while (timestamp_cache_.count(ts)) {
    ++ts;
  }

  timestamp_cache_.insert(ts);
  msg.timestamp_ns = ts;
  msg.setMap(*map_);

  if (!config.clear_distant_blocks) {
    return;
  }

  const auto indices = findBlocksToArchive(msg.world_t_body.cast<float>());
  msg.archived_blocks.insert(msg.archived_blocks.end(), indices.begin(), indices.end());
  map_->removeBlocks(indices);

  // const auto writable_mesh = activeMesh(map_->getMeshLayer(), indices);
  // if (writable_mesh && !writable_mesh->empty()) {
  //   kimera_pgmo::WriteMesh("/home/albert/Desktop/meshes/mesh_full.ply",
  //                          *writable_mesh);
  //   if (msg.sensor_data->features_mask) {
  //     cv::imwrite("/home/albert/Desktop/panoptic_images/" +
  //                 std::to_string(msg.timestamp_ns) + ".png",
  //                 msg.sensor_data->features_mask.value());
  //   }
  // }
}

bool ReconstructionModule::update(const InputPacket& msg, bool full_update) {
  VLOG(2) << "[Hydra Reconstruction] starting " << ((full_update) ? "full" : "partial")
          << " update @ " << msg.timestamp_ns << " [ns]";

  InputData::Ptr data = conversions::parseInputPacket(msg);
  if (!data) {
    return false;
  }

  BlockIndices updated_blocks;
  {  // timing scope
    ScopedTimer timer("places/tsdf", msg.timestamp_ns);
    updated_blocks = tsdf_integrator_->updateMap(*data, *map_);
  }  // timing scope

  updated_blocks_.insert(updated_blocks.begin(), updated_blocks.end());

  if (footprint_integrator_) {
    footprint_integrator_->addFreespaceFootprint(msg.world_T_body().cast<float>(),
                                                 *map_);
  }

  auto& tsdf = map_->getTsdfLayer();
  if (tsdf.numBlocks() == 0) {
    return false;
  }

  {  // timing scope
    ScopedTimer timer("places/mesh", msg.timestamp_ns);
    mesh_integrator_->generateMesh(*map_, true, true);
  }  // timing scope

  if (!full_update) {
    return false;
  }

  if (config.show_stats) {
    VLOG(config.stats_verbosity) << "Memory used: {" << map_->printStats() << "}";
  }

  auto output = ReconstructionOutput::fromInput(msg);
  output->sensor_data = data;
  fillOutput(*output);

  // Remove semantic features from map and mesh
  clearSemanticFeatures();

  Sink::callAll(sinks_, msg.timestamp_ns, data->getSensorPose(), tsdf, *output);

  if (output_queue_) {
    output_queue_->push(output);
  }

  for (const auto block : tsdf.blocksWithCondition(TsdfBlock::esdfUpdated)) {
    block->esdf_updated = false;
    block->updated = false;
  }

  return true;
}

void ReconstructionModule::clearSemanticFeatures() {
  // Start with semantic layer
  for (const auto& idx : updated_blocks_) {
    auto blocks = map_->getBlock(idx);
    if (blocks.tsdf) {
      for (size_t i = 0; i < blocks.tsdf->numVoxels(); ++i) {
        auto voxels = blocks.getVoxels(i);
        if (voxels.semantic) {
          voxels.semantic->feature_vector = std::nullopt;
          voxels.semantic->panoptic_id = 0;
        }
      }
    }
    // Continue with mesh layer
    auto mesh = map_->getMeshLayer().getBlockPtr(idx);
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

// TODO(nathan) push to map?
BlockIndices ReconstructionModule::findBlocksToArchive(
    const Eigen::Vector3f& center) const {
  const auto& tsdf = map_->getTsdfLayer();
  BlockIndices to_archive;
  for (const auto& block : tsdf) {
    if ((center - block.position()).norm() < config.dense_representation_radius_m) {
      continue;
    }

    // TODO(nathan) filter by update flag?
    to_archive.push_back(block.index);
  }

  return to_archive;
}

}  // namespace hydra
