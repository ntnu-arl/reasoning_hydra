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
#include "hydra/frontend/occupancy_frontier_extractor.h"

#include <config_utilities/config.h>
#include <config_utilities/types/enum.h>
#include <glog/logging.h>
#include <spark_dsg/node_attributes.h>
#include <spatial_hash/voxel_layer.h>

#include <functional>
#include <queue>

#include "hydra/common/global_info.h"

namespace hydra {

using namespace spark_dsg;

inline std::vector<size_t> nhood4(const size_t& idx, const DenseGrid& g) {
  std::vector<size_t> out;
  const auto size_x = static_cast<size_t>(g.width);
  const auto size_y = static_cast<size_t>(g.height);

  if (idx > size_x * size_y - 1) {
    LOG(WARNING) << "Searching neighbourhood for off-map point";
    return out;
  }

  if (idx % size_x > 0) {
    out.push_back(idx - 1);
  }
  if (idx % size_x < size_x - 1) {
    out.push_back(idx + 1);
  }
  if (idx >= size_x) {
    out.push_back(idx - size_x);
  }
  if (idx < size_x * (size_y - 1)) {
    out.push_back(idx + size_x);
  }
  return out;
}

inline std::vector<size_t> nhood8(const size_t& idx, const DenseGrid& costmap) {
  auto out = nhood4(idx, costmap);
  const auto size_x = static_cast<size_t>(costmap.width);
  const auto size_y = static_cast<size_t>(costmap.height);

  if (idx > size_x * size_y - 1) {
    return out;
  }

  if (idx % size_x > 0 && idx >= size_x) {
    out.push_back(idx - 1 - size_x);
  }
  if (idx % size_x > 0 && idx < size_x * (size_y - 1)) {
    out.push_back(idx - 1 + size_x);
  }
  if (idx % size_x < size_x - 1 && idx >= size_x) {
    out.push_back(idx + 1 - size_x);
  }
  if (idx % size_x < size_x - 1 && idx < size_x * (size_y - 1)) {
    out.push_back(idx + 1 + size_x);
  }

  return out;
}

inline bool nearestCell(size_t& out_result,
                        size_t start,
                        CellState val,
                        const DenseGrid& costmap) {
  const size_t size_x = static_cast<size_t>(costmap.width);
  const size_t size_y = static_cast<size_t>(costmap.height);

  if (start >= size_x * size_y) {
    return false;
  }

  // initialize breadth first search
  std::queue<size_t> bfs;
  std::vector<bool> visited(size_x * size_y, false);

  // push initial cell
  bfs.push(start);
  visited[start] = true;

  // search for neighbouring cell matching value
  while (!bfs.empty()) {
    size_t idx = bfs.front();
    bfs.pop();

    // return if cell of correct value is found
    if (costmap.cells[idx] == val) {
      out_result = idx;
      return true;
    }

    // iterate over all adjacent unvisited cells
    for (auto nbr : nhood8(idx, costmap)) {
      if (!visited[nbr]) {
        bfs.push(nbr);
        visited[nbr] = true;
      }
    }
  }
  return false;
}

inline bool isFree(CellState s) { return s == CellState::ObservedFree; }

inline bool isUnknown(CellState s) { return s == CellState::Unknown; }

inline bool isOccupied(CellState s) { return s == CellState::ObservedOccupied; }

inline bool isFrontierCellMatching(
    const size_t& idx,
    const DenseGrid& costmap,
    size_t& free_cell,
    const std::function<bool(size_t)>& is_valid_free_cell) {
  if (!isUnknown(costmap.cells[idx])) {
    return false;
  }

  auto n_hood = nhood4(idx, costmap);
  return std::any_of(n_hood.begin(), n_hood.end(), [&](auto n_idx) {
    if (is_valid_free_cell(n_idx)) {
      free_cell = n_idx;
      return true;
    }
    return false;
  });
}

inline bool isFrontierCell(const size_t& idx,
                           const DenseGrid& costmap,
                           size_t& free_cell) {
  return isFrontierCellMatching(idx, costmap, free_cell, [&](auto n_idx) {
    return isFree(costmap.cells[n_idx]);
  });
}

void declare_config(OccupancyFrontierExtractor::Config& config) {
  using namespace config;
  name("OccupancyFrontierExtractor::Config");
  field<CharConversion>(config.prefix, "prefix");
  field(config.nav_layer, "nav_layer");
  field(config.sinks, "sinks");
  field(config.object_connection_max_distance, "object_connection_max_distance");
  field(config.min_cluster_size, "min_cluster_size");
  field(config.use_initial_pose, "use_initial_pose");
  enum_field(config.detection_mode,
             "detection_mode",
             {{OccupancyFrontierExtractor::DetectionMode::WFD, "WFD"},
              {OccupancyFrontierExtractor::DetectionMode::GLOBAL, "GLOBAL"}});
  field(config.local_frontiers, "local_frontiers");
  field(config.max_radius_m, "max_radius_m");
  enum_field(
      config.features_collection_mode,
      "features_collection_mode",
      {{OccupancyFrontierExtractor::FeaturesCollectionMode::FRONTIER_POINTS,
        "FRONTIER_POINTS"},
       {OccupancyFrontierExtractor::FeaturesCollectionMode::CLUSTER_RADIUS,
        "CLUSTER_RADIUS"},
       {OccupancyFrontierExtractor::FeaturesCollectionMode::FRONTIER_POINTS_RADIUS,
        "FRONTIER_POINTS_RADIUS"}});
  field(config.features_radius, "features_radius");
  field((config.set_centroid_to_nearest_nav_node), "set_centroid_to_nearest_nav_node");
  field(config.inflation_radius_m, "inflation_radius_m");
}

size_t DenseGrid::idx(int r, int c) const { return (r - min_r) * width + (c - min_c); }

bool DenseGrid::inBounds(int r, int c) const {
  return r >= min_r && r < min_r + height && c >= min_c && c < min_c + width;
}

CellState DenseGrid::at(int r, int c) const { return cells[idx(r, c)]; }

DenseGrid DenseGrid::makeDense(const OccupancyGrid& g) {
  DenseGrid d;
  d.min_r = g.bounds.min_row;
  d.min_c = g.bounds.min_col;
  d.width = g.bounds.width;
  d.height = g.bounds.height;
  d.cells.resize(d.width * d.height, CellState::Unknown);
  d.features.resize(d.width * d.height, std::nullopt);
  for (auto& [idx, data] : g.grid) {
    int r = idx.first;
    int c = idx.second;
    d.cells[d.idx(r, c)] = data.state;
    d.features[d.idx(r, c)] = data.features;
  }
  return d;
}

OccupancyFrontierExtractor::OccupancyFrontierExtractor(const Config& config)
    : config(config::checkValid(config)), sinks_(Sink::instantiate(config.sinks)) {}

void OccupancyFrontierExtractor::extractFrontiersWFD(
    const OccupancyGrid::Ptr& occupancy_grid, const Eigen::Vector3d& robot_pos) {
  // 1. Remove old frontiers that are outside the active window (if using local
  // frontiers)
  if (config.local_frontiers) {
    std::vector<size_t> to_remove;

    for (size_t i = 0; i < clusters_.size(); ++i) {
      float dist = (clusters_[i].centroid - robot_pos.head<2>().cast<float>()).norm();

      if (dist <= config.max_radius_m) {
        to_remove.push_back(i);
      }

      clusters_[i].in_active_window = false;
    }

    for (int i = static_cast<int>(to_remove.size()) - 1; i >= 0; --i) {
      clusters_.erase(clusters_.begin() + to_remove[i]);
    }
  } else {
    clusters_.clear();
  }

  // 2. Precompute traversability mask and EDT
  DenseGrid costmap = DenseGrid::makeDense(*occupancy_grid);

  std::vector<float> edt;
  computeEDT(costmap, occupancy_grid->resolution, edt);

  auto isTraversable = [&](size_t idx) {
    return isFree(costmap.cells[idx]) && edt[idx] >= config.inflation_radius_m;
  };

  const int size = costmap.width * costmap.height;
  std::vector<DenseCellState> cell_states(size, DenseCellState::UNCHECKED);

  // 3. Get robot position in grid coordinates
  int rr = 0, rc = 0;

  if (config.use_initial_pose) {
    std::tie(rr, rc) =
        occupancy_grid->worldToGrid(initial_robot_position_.head<2>().cast<float>());
  } else {
    std::tie(rr, rc) = occupancy_grid->worldToGrid(robot_pos.head<2>().cast<float>());
  }

  if (!costmap.inBounds(rr, rc)) {
    return;
  }

  // 4. Initialize BFS with nearest free cell to robot position, respecting inflation
  // constraint
  std::queue<size_t> bfs;

  size_t clear;
  size_t start_idx = costmap.idx(rr, rc);
  clear = start_idx;
  // Find nearest free cell
  if (nearestCell(clear, start_idx, CellState::ObservedFree, costmap)) {
    if (isTraversable(clear)) {
      bfs.push(clear);
    } else {
      bfs.push(start_idx);
      LOG(WARNING) << "Start cell not traversable under inflation";
    }
  } else {
    bfs.push(start_idx);
    LOG(WARNING) << "Could not find nearby clear cell to start search";
  }
  cell_states[clear] = DenseCellState::MAP_OPEN;

  // 5. Perform BFS to find frontiers, expanding only into traversable cells or unknown
  // cells that border traversable free space
  auto is_frontier_cell = [&](size_t idx, size_t& free_cell) {
    return isFrontierCell(idx, costmap, free_cell);
  };

  while (!bfs.empty()) {
    size_t p = bfs.front();
    bfs.pop();

    if (cell_states[p] == DenseCellState::MAP_CLOSED) {
      continue;
    }

    size_t free_cell;
    if (isFrontierCell(p, costmap, free_cell)) {
      frontiers::FrontierCluster cluster = buildFrontier(
          occupancy_grid, p, cell_states, costmap, is_frontier_cell, robot_pos.z());

      float dist =
          (cluster.centroid / cluster.points.size() - robot_pos.head<2>().cast<float>())
              .norm();

      if (cluster.points.size() * occupancy_grid->resolution >=
          config.min_cluster_size) {
        if (config.local_frontiers && dist > config.max_radius_m) {
          continue;
        }

        clusters_.push_back(cluster);
      }
    }

    for (auto nbr : nhood8(p, costmap)) {
      if (cell_states[nbr] == DenseCellState::MAP_OPEN ||
          cell_states[nbr] == DenseCellState::MAP_CLOSED) {
        continue;
      }

      // Case 1: traversable free cell
      if (isTraversable(nbr)) {
        bfs.push(nbr);
        cell_states[nbr] = DenseCellState::MAP_OPEN;
        continue;
      }

      // Case 2: unknown cell that borders traversable free
      if (isUnknown(costmap.cells[nbr])) {
        auto n_hood = nhood8(nbr, costmap);

        bool has_traversable_free = std::any_of(
            n_hood.begin(), n_hood.end(), [&](auto idx) { return isTraversable(idx); });

        if (has_traversable_free) {
          bfs.push(nbr);
          cell_states[nbr] = DenseCellState::MAP_OPEN;
        }
      }
    }

    cell_states[p] = DenseCellState::MAP_CLOSED;
  }
}

void OccupancyFrontierExtractor::extractFrontiersGlobal(
    const OccupancyGrid::Ptr& occupancy_grid, const Eigen::Vector3d& robot_pos) {
  clusters_.clear();

  DenseGrid costmap = DenseGrid::makeDense(*occupancy_grid);

  std::vector<float> edt;
  computeEDT(costmap, occupancy_grid->resolution, edt);

  auto isTraversable = [&](size_t idx) {
    return isFree(costmap.cells[idx]) && edt[idx] >= config.inflation_radius_m;
  };

  auto is_frontier_cell = [&](size_t idx, size_t& free_cell) {
    return isFrontierCellMatching(
        idx, costmap, free_cell, [&](auto n_idx) { return isTraversable(n_idx); });
  };

  const size_t size = static_cast<size_t>(costmap.width * costmap.height);
  std::vector<DenseCellState> cell_states(size, DenseCellState::UNCHECKED);

  for (size_t idx = 0; idx < size; ++idx) {
    if (cell_states[idx] != DenseCellState::UNCHECKED) {
      continue;
    }

    size_t free_cell;
    if (!is_frontier_cell(idx, free_cell)) {
      continue;
    }

    frontiers::FrontierCluster cluster = buildFrontier(
        occupancy_grid, idx, cell_states, costmap, is_frontier_cell, robot_pos.z());

    if (cluster.points.size() * occupancy_grid->resolution >= config.min_cluster_size) {
      clusters_.push_back(cluster);
    }
  }
}

frontiers::FrontierCluster OccupancyFrontierExtractor::buildFrontier(
    const OccupancyGrid::Ptr& occupancy_grid,
    size_t initial_cell,
    std::vector<DenseCellState>& cell_states,
    const DenseGrid& costmap,
    const std::function<bool(size_t, size_t&)>& is_frontier_cell,
    const double& robot_z) const {
  frontiers::FrontierCluster output;
  std::vector<size_t> frontier_indices;
  std::queue<size_t> bfs;
  bfs.push(initial_cell);
  cell_states[initial_cell] = DenseCellState::FRONTIER_OPEN;

  while (!bfs.empty()) {
    auto p = bfs.front();
    bfs.pop();

    if (cell_states[p] == DenseCellState::FRONTIER_CLOSED ||
        cell_states[p] == DenseCellState::MAP_CLOSED) {
      continue;
    }
    size_t free_cell;
    if (is_frontier_cell(p, free_cell)) {
      int r = p / costmap.width, c = p % costmap.width;
      int r_free = free_cell / costmap.width, c_free = free_cell % costmap.width;
      Point2d pt = occupancy_grid->gridToWorld({r + costmap.min_r, c + costmap.min_c});
      output.points.push_back(pt);
      Point2d free_pt =
          occupancy_grid->gridToWorld({r_free + costmap.min_r, c_free + costmap.min_c});
      output.centroid += pt;
      output.free_points.push_back(free_pt);
      output.grid_indices.push_back({r + costmap.min_r, c + costmap.min_c});
      bfs.push(p);
      frontier_indices.push_back(p);
      for (auto nbr : nhood8(p, costmap)) {
        if (cell_states[nbr] == DenseCellState::MAP_OPEN ||
            cell_states[nbr] == DenseCellState::UNCHECKED) {
          bfs.push(nbr);
          cell_states[nbr] = DenseCellState::FRONTIER_OPEN;
        }
      }
    }
    cell_states[p] = DenseCellState::FRONTIER_CLOSED;
  }
  // mark all frontier points as processed
  std::for_each(
      frontier_indices.begin(), frontier_indices.end(), [&cell_states](auto idx) {
        cell_states[idx] = DenseCellState::MAP_CLOSED;
      });

  output.centroid /= output.points.size();
  selectFreeFrontier(output);
  computeClusterDirection(occupancy_grid, output);
  output.in_active_window = true;
  output.z_value = robot_z;

  return output;
}

void OccupancyFrontierExtractor::selectFreeFrontier(
    frontiers::FrontierCluster& cluster) const {
  size_t best = 0;
  double best_dist = std::numeric_limits<double>::max();

  for (size_t i = 0; i < cluster.free_points.size(); ++i) {
    double d = (cluster.free_points[i] - cluster.centroid).squaredNorm();
    if (d < best_dist) {
      best_dist = d;
      best = i;
    }
  }
  cluster.centroid = cluster.free_points[best];
}

void OccupancyFrontierExtractor::computeClusterDirection(
    const OccupancyGrid::Ptr& occupancy_grid,
    frontiers::FrontierCluster& cluster) const {
  Eigen::Vector2f dir_sum = Eigen::Vector2f::Zero();

  for (const auto& idx : cluster.grid_indices) {
    // world position of this frontier cell
    Eigen::Vector2f p = occupancy_grid->gridToWorld(idx);

    for (int k = 0; k < 8; ++k) {
      GridIndex nbr{idx.first + dr_[k], idx.second + dc_[k]};
      auto it = occupancy_grid->grid.find(nbr);
      // Neighbor is unknown or outside grid: frontier faces this way
      if (it == occupancy_grid->grid.end() || it->second.state == CellState::Unknown) {
        Eigen::Vector2f q = occupancy_grid->gridToWorld(nbr);
        Eigen::Vector2f v = q - p;

        float n = v.norm();
        if (n > 1e-4f) {
          dir_sum += v / n;  // accumulate normalized direction
        }
      }
    }
  }

  if (dir_sum.norm() > 1e-4f) {
    cluster.direction = dir_sum.normalized();
  } else {
    cluster.direction = Eigen::Vector2f::Zero();
  }
}

void OccupancyFrontierExtractor::computeClustersSemanticsFP(
    const OccupancyGrid::Ptr& occupancy_grid) {
  for (auto& cluster : clusters_) {
    if (!cluster.in_active_window) {
      continue;
    }
    FeatureVector cluster_features;
    size_t num_features = 0;
    for (const auto& idx : cluster.grid_indices) {
      if (occupancy_grid->grid.count(idx) > 0) {
        const auto& cell_data = occupancy_grid->grid.at(idx);
        if (cell_data.features.has_value()) {
          if (cluster_features.size() == 0) {
            cluster_features = FeatureVector::Zero(cell_data.features.value().size());
          }
          cluster_features += cell_data.features.value();
          num_features += 1;
          cluster.feature_points.push_back(occupancy_grid->gridToWorld(idx));
          cluster.feature_points_features.push_back(cell_data.features.value());
        }
        // Semantic labels
        for (const auto& [label, count] : cell_data.semantic_class_labels) {
          cluster.semantic_class_labels[label] += count;
        }
      }
    }
    if (num_features > 0) {
      cluster.feature = cluster_features / static_cast<float>(num_features);
    }
  }
}

void OccupancyFrontierExtractor::computeClustersSemanticsRadius(
    const OccupancyGrid::Ptr& occupancy_grid) {
  const int radius_cells =
      static_cast<int>(std::ceil(config.features_radius / occupancy_grid->resolution));

  for (auto& cluster : clusters_) {
    if (cluster.grid_indices.empty() || !cluster.in_active_window) {
      continue;
    }

    FeatureVector sum;
    size_t count = 0;

    // Use centroid as search center
    GridIndex center = occupancy_grid->worldToGrid(cluster.centroid);

    for (int dr = -radius_cells; dr <= radius_cells; ++dr) {
      for (int dc = -radius_cells; dc <= radius_cells; ++dc) {
        GridIndex idx{center.first + dr, center.second + dc};
        auto it = occupancy_grid->grid.find(idx);
        if (it == occupancy_grid->grid.end()) {
          continue;
        }
        if (!it->second.features.has_value()) {
          continue;
        }

        // Optional: precise circle check
        const float dist = std::hypot(dr, dc) * occupancy_grid->resolution;
        if (dist > config.features_radius) {
          continue;
        }

        if (count == 0) {
          sum = FeatureVector::Zero(it->second.features->size());
        }

        sum += *(it->second.features);
        ++count;
        cluster.feature_points.push_back(occupancy_grid->gridToWorld(idx));
        cluster.feature_points_features.push_back(*(it->second.features));
        // Semantic labels
        for (const auto& [label, label_count] : it->second.semantic_class_labels) {
          cluster.semantic_class_labels[label] += label_count;
        }
      }
    }

    if (count > 0) {
      sum /= static_cast<float>(count);
      cluster.feature = sum;
    }
  }
}

void OccupancyFrontierExtractor::computeClustersSemanticsFPRadius(
    const OccupancyGrid::Ptr& occupancy_grid) {
  if (occupancy_grid->empty()) return;

  const int radius_cells =
      static_cast<int>(std::ceil(config.features_radius / occupancy_grid->resolution));
  for (auto& cluster : clusters_) {
    if (cluster.grid_indices.empty() || !cluster.in_active_window) {
      continue;
    }
    FeatureVector sum;
    size_t count = 0;

    for (const auto& center_idx : cluster.grid_indices) {
      for (int dr = -radius_cells; dr <= radius_cells; ++dr) {
        for (int dc = -radius_cells; dc <= radius_cells; ++dc) {
          GridIndex idx{center_idx.first + dr, center_idx.second + dc};

          auto it = occupancy_grid->grid.find(idx);
          if (it == occupancy_grid->grid.end()) {
            continue;
          }
          if (!it->second.features.has_value()) {
            continue;
          }
          // Optional: circular mask
          const float dist = std::hypot(dr, dc) * occupancy_grid->resolution;
          if (dist > config.features_radius) {
            continue;
          }
          if (count == 0) {
            sum = FeatureVector::Zero(it->second.features->size());
          }
          sum += *(it->second.features);
          ++count;
          cluster.feature_points.push_back(occupancy_grid->gridToWorld(idx));
          cluster.feature_points_features.push_back(*(it->second.features));
          // Semantic labels
          for (const auto& [label, label_count] : it->second.semantic_class_labels) {
            cluster.semantic_class_labels[label] += label_count;
          }
        }
      }
    }

    if (count > 0) {
      sum /= static_cast<float>(count);
      cluster.feature = sum;
    }
  }
}

void OccupancyFrontierExtractor::detectRealFrontiers(const ActiveWindowOutput& input) {
  if (!initial_position_set_) {
    initial_robot_position_ = input.world_T_body().translation();
    initial_position_set_ = true;
  }
  switch (config.detection_mode) {
    case DetectionMode::WFD:
      extractFrontiersWFD(input.occupancy_grid, input.world_T_body().translation());
      break;
    case DetectionMode::GLOBAL:
      extractFrontiersGlobal(input.occupancy_grid, input.world_T_body().translation());
      break;
  }

  switch (config.features_collection_mode) {
    case FeaturesCollectionMode::FRONTIER_POINTS:
      computeClustersSemanticsFP(input.occupancy_grid);
      break;
    case FeaturesCollectionMode::CLUSTER_RADIUS:
      computeClustersSemanticsRadius(input.occupancy_grid);
      break;
    case FeaturesCollectionMode::FRONTIER_POINTS_RADIUS:
      computeClustersSemanticsFPRadius(input.occupancy_grid);
      break;
  }
}

void OccupancyFrontierExtractor::addRealFrontiers(
    uint64_t timestamp_ns,
    DynamicSceneGraph& graph,
    const Eigen::Isometry3d& world_T_body) {
  if (!graph.hasLayer(DsgLayers::FRONTIERS)) {
    return;
  }

  NodeSymbol next_id(config.prefix, 0);
  for (auto& cluster : clusters_) {
    auto attrs = std::make_unique<GlobalFrontierNodeAttributes>();
    attrs->position =
        Eigen::Vector3d(cluster.centroid.x(), cluster.centroid.y(), cluster.z_value);
    attrs->last_update_time_ns = timestamp_ns;
    attrs->is_active = true;
    attrs->semantic_feature = cluster.feature.value_or(Eigen::VectorXf());
    attrs->semantic_class_labels = cluster.semantic_class_labels;
    attrs->direction = cluster.direction;
    for (size_t i = 0; i < cluster.feature_points_features.size(); ++i) {
      attrs->features.push_back(cluster.feature_points_features[i]);
      attrs->feature_points.push_back(cluster.feature_points[i]);
    }

    for (const auto& free_point : cluster.free_points) {
      attrs->frontier_points.push_back(free_point);
    }

    graph.addOrUpdateNode(DsgLayers::FRONTIERS, next_id, std::move(attrs));
    addEdges(next_id, world_T_body, timestamp_ns, graph, cluster);
    ++next_id;
  }

  if (last_num_frontiers_ > clusters_.size()) {
    for (size_t i = clusters_.size(); i < last_num_frontiers_; ++i) {
      graph.removeNode(NodeSymbol(config.prefix, i));
    }
  }
  last_num_frontiers_ = clusters_.size();
}

void OccupancyFrontierExtractor::addEdges(const NodeSymbol& id,
                                          const Eigen::Isometry3d& world_T_body,
                                          uint64_t timestamp_ns,
                                          DynamicSceneGraph& graph,
                                          frontiers::FrontierCluster& cluster) {
  auto& frontier = graph.getLayer(DsgLayers::FRONTIERS).getNode(id);
  auto& attrs = frontier.attributes<GlobalFrontierNodeAttributes>();

  if (graph.hasLayer(config.nav_layer)) {
    const auto& nav_layer = graph.getLayer(config.nav_layer);
    if (!nav_layer.nodes().empty()) {
      std::vector<NodeId> nav_ids;
      nav_ids.reserve(nav_layer.nodes().size());

      for (const auto& [pid, _] : nav_layer.nodes()) {
        nav_ids.push_back(pid);
      }
      nn_finder_.reset(new NearestNodeFinder(nav_layer, nav_ids));
      nn_finder_->find(attrs.position, 1, false, [&](NodeId pid, size_t, double) {
        attrs.connected_nav = pid;
        attrs.nav_layer = config.nav_layer;
        if (config.set_centroid_to_nearest_nav_node) {
          const auto& nav_node = nav_layer.getNode(pid);
          const auto& nav_attrs = nav_node.attributes<spark_dsg::NodeAttributes>();
          attrs.position.x() = nav_attrs.position.x();
          attrs.position.y() = nav_attrs.position.y();
          attrs.use_nav_as_centroid = true;
          cluster.centroid.x() = nav_attrs.position.x();
          cluster.centroid.y() = nav_attrs.position.y();
        }
      });
    }
  }

  if (graph.hasLayer(DsgLayers::OBJECTS)) {
    const auto& objects_layer = graph.getLayer(DsgLayers::OBJECTS);
    if (!objects_layer.nodes().empty()) {
      std::vector<NodeId> object_ids;
      object_ids.reserve(objects_layer.nodes().size());
      for (const auto& [oid, _] : objects_layer.nodes()) {
        object_ids.push_back(oid);
      }
      nn_finder_.reset(new NearestNodeFinder(objects_layer, object_ids));
      nn_finder_->findRadius(
          attrs.position,
          config.object_connection_max_distance,
          false,
          [&](NodeId oid, size_t, double) { attrs.connected_objects.push_back(oid); });
    }
  }
  Sink::callAll(sinks_, timestamp_ns, world_T_body, clusters_);
}

void OccupancyFrontierExtractor::computeEDT(const DenseGrid& grid,
                                            const float& resolution,
                                            std::vector<float>& edt) const {
  const int W = grid.width;
  const int H = grid.height;
  const float INF = std::numeric_limits<float>::infinity();

  edt.clear();
  edt.resize(W * H, INF);

  // Initialize: occupied = 0, else INF
  for (int i = 0; i < W * H; ++i) {
    if (grid.cells[i] == CellState::ObservedOccupied) {
      edt[i] = 0.0f;
    }
  }

  // Forward pass
  for (int r = 0; r < H; ++r) {
    for (int c = 0; c < W; ++c) {
      int idx = r * W + c;
      if (edt[idx] == 0.0f) continue;

      if (r > 0) edt[idx] = std::min(edt[idx], edt[(r - 1) * W + c] + resolution);
      if (c > 0) edt[idx] = std::min(edt[idx], edt[r * W + (c - 1)] + resolution);
      if (r > 0 && c > 0)
        edt[idx] = std::min(edt[idx],
                            edt[(r - 1) * W + (c - 1)] + resolution * std::sqrt(2.0f));
    }
  }

  // Backward pass
  for (int r = H - 1; r >= 0; --r) {
    for (int c = W - 1; c >= 0; --c) {
      int idx = r * W + c;

      if (r < H - 1) edt[idx] = std::min(edt[idx], edt[(r + 1) * W + c] + resolution);
      if (c < W - 1) edt[idx] = std::min(edt[idx], edt[r * W + (c + 1)] + resolution);
      if (r < H - 1 && c < W - 1)
        edt[idx] = std::min(edt[idx],
                            edt[(r + 1) * W + (c + 1)] + resolution * std::sqrt(2.0f));
    }
  }
}
}  // namespace hydra
