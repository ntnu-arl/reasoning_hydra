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
#pragma once
#include <config_utilities/config_utilities.h>

#include <Eigen/Core>

#include "hydra/common/output_sink.h"
#include "hydra/reconstruction/occupancy_grid.h"
#include "hydra/reconstruction/volumetric_map.h"

namespace hydra {

class OccupancyIntegrator {
 public:
  struct DebugInfo {
    std::vector<Eigen::Vector3f> column_positions;
    std::vector<double> column_weights;
    std::vector<double> column_distances;
    std::vector<std::optional<FeatureVector>> column_features;
    std::vector<SemanticLabel> column_labels;
    std::vector<CellState> states;
    CellState state;
    std::optional<FeatureVector> feature;
    std::map<SemanticLabel, size_t> semantic_class_labels;
  };

  using Sink = OutputSink<uint64_t, const Eigen::Isometry3d&, const OccupancyGrid&>;
  using DebugSink = OutputSink<const DebugInfo&>;
  enum class FeaturesCollapseMode { ALL, HEIGHT, SURFACES };

  struct Config {
    // TSDF interpretation
    double min_observation_weight = 1.0e-5;
    double occupied_distance = 0.1;
    bool clear_occupied_on_free = false;
    bool use_free_space_evidence = false;
    size_t min_free_space_voxels = 1;
    double min_free_space_fraction = 0.6;
    // Height filtering
    double min_height = -0.5;
    double max_height = 1.0;
    // Pose-relative integration filtering. Non-positive values disable this filter.
    double max_integration_radius_m = 0.0;
    // Unknown cells to add around observed cells for frontier detection.
    int unknown_padding_cells = 1;
    // Features config
    FeaturesCollapseMode features_collapse_mode = FeaturesCollapseMode::ALL;
    // Initial sphere config
    bool initial_sphere_free = true;
    float initial_sphere_radius_m = 0.0;
    // Sinks
    std::vector<Sink::Factory> sinks;
    std::vector<DebugSink::Factory> debug_sinks;
  } const config;

  explicit OccupancyIntegrator(const Config& config);

  virtual ~OccupancyIntegrator() = default;

  const OccupancyGrid::Ptr integrate(const uint64_t timestamp,
                                     const TsdfLayer& layer,
                                     const SemanticLayer* semantic_layer,
                                     const Eigen::Isometry3d& world_T_body);

 protected:
  void addFeatureToCell(const float& voxel_z,
                        const float& z_min,
                        const float& z_max,
                        const TsdfVoxel& voxel,
                        const std::optional<FeatureVector>& voxel_feature,
                        const SemanticLabel& voxel_label,
                        std::optional<FeatureVector>& cell_feature,
                        std::map<SemanticLabel, size_t>& cell_labels,
                        size_t& count) const;
  OccupancyGrid::Ptr global_grid_;
  Sink::List sinks_;
  DebugSink::List debug_sinks_;
  std::vector<GridIndex> initial_sphere_indices_;
};

void declare_config(OccupancyIntegrator::Config& config);

}  // namespace hydra
