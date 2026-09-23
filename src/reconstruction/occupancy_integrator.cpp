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
#include "hydra/reconstruction/occupancy_integrator.h"

#include <config_utilities/config.h>
#include <config_utilities/types/enum.h>

#include <vector>

namespace hydra {

namespace {

void addUnknownPadding(OccupancyGrid& grid, int padding_cells) {
  if (padding_cells <= 0 || grid.empty()) {
    return;
  }

  std::vector<GridIndex> padding_indices;
  for (const auto& [idx, data] : grid.grid) {
    if (data.state == CellState::Unknown) {
      continue;
    }

    for (int dr = -padding_cells; dr <= padding_cells; ++dr) {
      for (int dc = -padding_cells; dc <= padding_cells; ++dc) {
        if (dr == 0 && dc == 0) {
          continue;
        }

        const GridIndex padding_idx{idx.first + dr, idx.second + dc};
        if (grid.grid.count(padding_idx) == 0) {
          padding_indices.push_back(padding_idx);
        }
      }
    }
  }

  for (const auto& idx : padding_indices) {
    const auto [it, inserted] = grid.grid.emplace(idx, OccupancyGrid::GridData());
    if (inserted) {
      it->second.state = CellState::Unknown;
      grid.bounds.update(idx.first, idx.second);
    }
  }
}

}  // namespace

void declare_config(OccupancyIntegrator::Config& config) {
  using namespace config;
  name("OccupancyIntegratorConfig");
  field(config.min_observation_weight, "min_observation_weight");
  field(config.occupied_distance, "occupied_distance");
  field(config.clear_occupied_on_free, "clear_occupied_on_free");
  field(config.use_free_space_evidence, "use_free_space_evidence");
  field(config.min_free_space_voxels, "min_free_space_voxels");
  field(config.min_free_space_fraction, "min_free_space_fraction");
  field(config.min_height, "min_height");
  field(config.max_height, "max_height");
  field(config.max_integration_radius_m, "max_integration_radius_m");
  field(config.unknown_padding_cells, "unknown_padding_cells");
  enum_field(config.features_collapse_mode,
             "features_collapse_mode",
             {{OccupancyIntegrator::FeaturesCollapseMode::ALL, "ALL"},
              {OccupancyIntegrator::FeaturesCollapseMode::HEIGHT, "HEIGHT"},
              {OccupancyIntegrator::FeaturesCollapseMode::SURFACES, "SURFACES"}});
  field(config.initial_sphere_free, "initial_sphere_free");
  field(config.initial_sphere_radius_m, "initial_sphere_radius_m");
  field(config.sinks, "sinks");
  field(config.debug_sinks, "debug_sinks");
}

OccupancyIntegrator::OccupancyIntegrator(const Config& config)
    : config(config),
      global_grid_(std::make_shared<OccupancyGrid>()),
      sinks_(Sink::instantiate(config.sinks)),
      debug_sinks_(DebugSink::instantiate(config.debug_sinks)) {}

const OccupancyGrid::Ptr OccupancyIntegrator::integrate(
    const uint64_t timestamp,
    const TsdfLayer& layer,
    const SemanticLayer* semantic_layer,
    const Eigen::Isometry3d& world_T_body) {
  global_grid_->resolution = layer.voxel_size;

  const double z_min = world_T_body.translation().z() + config.min_height;
  const double z_max = world_T_body.translation().z() + config.max_height;
  const bool use_integration_radius = config.max_integration_radius_m > 0.0;
  const double max_integration_radius_sq =
      config.max_integration_radius_m * config.max_integration_radius_m;
  const Eigen::Vector2f body_xy = world_T_body.translation().head<2>().cast<float>();

  OccupancyGrid new_grid;
  new_grid.resolution = layer.voxel_size;
  std::unordered_map<GridIndex, bool, GridIndexHash> hard_unknown_cells;
  for (const auto& block : layer) {
    for (size_t x = 0; x < block.voxels_per_side; ++x) {
      for (size_t y = 0; y < block.voxels_per_side; ++y) {
        const Eigen::Vector3f pos =
            block.getVoxelPosition({static_cast<int>(x), static_cast<int>(y), 0});
        if (use_integration_radius &&
            (pos.head<2>() - body_xy).squaredNorm() > max_integration_radius_sq) {
          continue;
        }
        const auto cell_idx = new_grid.worldToGrid(pos.head<2>());
        bool free_cell = true;
        bool occupied_cell = false;
        bool out_of_bounds = true;
        bool known_cell = true;
        size_t free_space_voxels = 0;
        size_t occupied_voxels = 0;
        if (new_grid.grid.count(cell_idx) > 0) {
          if (new_grid.grid.at(cell_idx).state == CellState::ObservedOccupied) {
            occupied_cell = true;
          } else {
            free_cell = new_grid.grid.at(cell_idx).state == CellState::ObservedFree;
          }
        } else {
          new_grid.grid.emplace(cell_idx, OccupancyGrid::GridData());
          new_grid.bounds.update(cell_idx.first, cell_idx.second);
        }
        DebugInfo debug_info;

        for (size_t z = 0; z < block.voxels_per_side; ++z) {
          // Semantic update (if available)
          const Eigen::Vector3f pos_z = block.getVoxelPosition(
              {static_cast<int>(x), static_cast<int>(y), static_cast<int>(z)});
          const auto& voxel = block.getVoxel(
              {static_cast<int>(x), static_cast<int>(y), static_cast<int>(z)});
          if (semantic_layer && semantic_layer->hasBlock(block.index)) {
            const auto& semantic_voxel = semantic_layer->getBlock(block.index)
                                             .getVoxel({static_cast<int>(x),
                                                        static_cast<int>(y),
                                                        static_cast<int>(z)});
            addFeatureToCell(pos_z.z(),
                             z_min,
                             z_max,
                             voxel,
                             semantic_voxel.pixelwise_feature,
                             semantic_voxel.semantic_label,
                             new_grid.grid[cell_idx].features,
                             new_grid.grid[cell_idx].semantic_class_labels,
                             new_grid.grid[cell_idx].observation_count);
            if (!debug_sinks_.empty()) {
              debug_info.column_features.push_back(semantic_voxel.pixelwise_feature);
              debug_info.column_labels.push_back(semantic_voxel.semantic_label);
            }
          }

          if (!debug_sinks_.empty()) {
            debug_info.column_positions.push_back(pos_z);
          }
          // Occupancy update
          if (occupied_cell && !config.use_free_space_evidence &&
              debug_sinks_.empty()) {
            continue;
          }

          // Height filtering
          if (pos_z.z() < z_min || pos_z.z() > z_max) {
            if (!debug_sinks_.empty()) {
              debug_info.states.push_back(CellState::OutOfBounds);
              debug_info.column_distances.push_back(-100.0);
              debug_info.column_weights.push_back(voxel.weight);
            }
            continue;
          }
          out_of_bounds = false;

          if (hard_unknown_cells.count(cell_idx) == 0) {
            hard_unknown_cells[cell_idx] =
                occupancy::isHardUnknown(voxel, config.min_observation_weight);
          } else {
            hard_unknown_cells[cell_idx] =
                hard_unknown_cells[cell_idx] ||
                occupancy::isHardUnknown(voxel, config.min_observation_weight);
          }

          // Not observed
          if (!occupancy::isObserved(voxel, config.min_observation_weight)) {
            known_cell = false;
            if (!debug_sinks_.empty()) {
              debug_info.states.push_back(CellState::Unknown);
              debug_info.column_distances.push_back(voxel.distance);
              debug_info.column_weights.push_back(voxel.weight);
            }
            continue;
          }
          // Free space or occupied space
          if (occupancy::isOccupied(voxel, config.occupied_distance)) {
            occupied_cell = true;
            free_cell = false;
            ++occupied_voxels;
            if (!debug_sinks_.empty()) {
              debug_info.states.push_back(CellState::ObservedOccupied);
              debug_info.column_distances.push_back(voxel.distance);
              debug_info.column_weights.push_back(voxel.weight);
            }
            continue;
          }
          if (!debug_sinks_.empty()) {
            debug_info.states.push_back(CellState::ObservedFree);
            debug_info.column_distances.push_back(voxel.distance);
            debug_info.column_weights.push_back(voxel.weight);
          }
          ++free_space_voxels;
          free_cell = true;
        }

        if (config.use_free_space_evidence) {
          const size_t classified_voxels = free_space_voxels + occupied_voxels;
          const double free_space_fraction =
              classified_voxels == 0 ? 0.0
                                     : static_cast<double>(free_space_voxels) /
                                           static_cast<double>(classified_voxels);
          const bool enough_free_space =
              free_space_voxels >= config.min_free_space_voxels &&
              free_space_fraction >= config.min_free_space_fraction;

          if (!out_of_bounds && enough_free_space) {
            new_grid.grid[cell_idx].state = CellState::ObservedFree;
          } else if (occupied_cell) {
            new_grid.grid[cell_idx].state = CellState::ObservedOccupied;
          }
        } else {
          if (occupied_cell) {
            new_grid.grid[cell_idx].state = CellState::ObservedOccupied;
          } else if (free_cell && !out_of_bounds && known_cell) {
            new_grid.grid[cell_idx].state = CellState::ObservedFree;
          }
        }
        if (!debug_sinks_.empty()) {
          debug_info.state = new_grid.grid[cell_idx].state;
          debug_info.semantic_class_labels =
              new_grid.grid[cell_idx].semantic_class_labels;
          if (new_grid.grid[cell_idx].features) {
            debug_info.feature =
                new_grid.grid[cell_idx].features.value() /
                static_cast<float>(new_grid.grid[cell_idx].observation_count);
          }
          DebugSink::callAll(debug_sinks_, debug_info);
        }
      }
    }
  }

  for (auto& [idx, data] : new_grid.grid) {
    if (hard_unknown_cells.count(idx) > 0 && hard_unknown_cells[idx] &&
        data.state != CellState::ObservedOccupied &&
        !(config.use_free_space_evidence && data.state == CellState::ObservedFree)) {
      data.state = CellState::Unknown;
    }
    // Update cell state
    if (global_grid_->grid.count(idx) == 0) {
      global_grid_->grid[idx].state = data.state;
      global_grid_->bounds.update(idx.first, idx.second);
    } else {
      if (data.state == CellState::ObservedOccupied) {
        global_grid_->grid[idx].state = CellState::ObservedOccupied;
      } else if (data.state == CellState::ObservedFree &&
                 (config.clear_occupied_on_free ||
                  global_grid_->grid.at(idx).state != CellState::ObservedOccupied)) {
        global_grid_->grid[idx].state = CellState::ObservedFree;
      }
    }
    // if (data.state == CellState::Unknown) {
    //   continue;
    // }
    // global_grid_->grid[idx].state = data.state;
    // global_grid_->bounds.update(idx.first, idx.second);
    // Update features and observation count
    auto& existing_cell = global_grid_->grid[idx];
    if (data.observation_count > 0) {
      data.features.value() /= static_cast<float>(data.observation_count);
      if (existing_cell.observation_count == 0) {
        existing_cell.features = data.features;
      } else {
        existing_cell.features.value() =
            (existing_cell.features.value() *
                 static_cast<float>(existing_cell.observation_count) +
             data.features.value()) /
            static_cast<float>(existing_cell.observation_count + 1);
      }
      existing_cell.observation_count += 1;
    }
    // Merge semantic class labels
    for (const auto& [label, count] : data.semantic_class_labels) {
      existing_cell.semantic_class_labels[label] += count;
    }
  }

  // Initial sphere (if configured)
  if (config.initial_sphere_free) {
    if (initial_sphere_indices_.empty()) {
      const int radius_voxels = static_cast<int>(
          std::ceil(config.initial_sphere_radius_m / new_grid.resolution));
      for (int dx = -radius_voxels; dx <= radius_voxels; ++dx) {
        for (int dy = -radius_voxels; dy <= radius_voxels; ++dy) {
          if ((dx * new_grid.resolution) * (dx * new_grid.resolution) +
                  (dy * new_grid.resolution) * (dy * new_grid.resolution) <=
              config.initial_sphere_radius_m * config.initial_sphere_radius_m) {
            const GridIndex idx = global_grid_->worldToGrid(
                world_T_body.translation().head<2>().cast<float>() +
                Point2d(dx, dy) * new_grid.resolution);
            initial_sphere_indices_.push_back(idx);
          }
        }
      }
    }
    for (const auto& idx : initial_sphere_indices_) {
      auto& cell = global_grid_->grid[idx];
      cell.state = (cell.state != CellState::ObservedOccupied)
                       ? CellState::ObservedFree
                       : CellState::ObservedOccupied;
      global_grid_->bounds.update(idx.first, idx.second);
    }
  }

  addUnknownPadding(*global_grid_, config.unknown_padding_cells);

  // Create a copy to return
  OccupancyGrid::Ptr output_grid = std::make_shared<OccupancyGrid>();
  *output_grid = *global_grid_;
  Sink::callAll(sinks_, timestamp, world_T_body, *output_grid);
  return output_grid;
}

void OccupancyIntegrator::addFeatureToCell(
    const float& voxel_z,
    const float& z_min,
    const float& z_max,
    const TsdfVoxel& voxel,
    const std::optional<FeatureVector>& voxel_feature,
    const SemanticLabel& voxel_label,
    std::optional<FeatureVector>& cell_feature,
    std::map<SemanticLabel, size_t>& cell_labels,
    size_t& count) const {
  // Lambda to add feature
  auto addFeature = [&]() {
    if (voxel_feature.has_value()) {
      if (!cell_feature.has_value()) {
        cell_feature = FeatureVector::Zero(voxel_feature->size());
      }
      cell_feature.value() += voxel_feature.value();
      count += 1;
    }
  };

  switch (config.features_collapse_mode) {
    case FeaturesCollapseMode::ALL:
      addFeature();
      cell_labels[voxel_label] += 1;
      break;
    case FeaturesCollapseMode::HEIGHT:
      if (voxel_z >= z_min && voxel_z <= z_max) {
        addFeature();
        cell_labels[voxel_label] += 1;
      }
      break;
    case FeaturesCollapseMode::SURFACES:
      if (occupancy::isObserved(voxel, config.min_observation_weight) &&
          occupancy::isOccupied(voxel, config.occupied_distance)) {
        addFeature();
        cell_labels[voxel_label] += 1;
      }
      break;
  }
}

}  // namespace hydra
