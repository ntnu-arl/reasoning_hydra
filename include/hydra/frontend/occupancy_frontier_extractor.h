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
#include <config_utilities/factory.h>
#include <config_utilities/virtual_config.h>
#include <spark_dsg/dynamic_scene_graph.h>
#include <spark_dsg/node_symbol.h>

#include <Eigen/Core>
#include <functional>
#include <unordered_map>
#include <vector>

#include "hydra/active_window/active_window_output.h"
#include "hydra/common/output_sink.h"
#include "hydra/reconstruction/occupancy_grid.h"
#include "hydra/utils/nearest_neighbor_utilities.h"

namespace hydra {

using namespace spark_dsg;

namespace frontiers {

struct ColumnInfo {
  bool free = true;
  bool observed = true;
  bool has_ground = false;
  bool in_fov = false;
};

struct FrontierPoint {
  Point2d position;
  std::map<SemanticLabel, size_t> semantic_class_labels;
  std::optional<FeatureVector> feature;
};

struct FrontierCluster {
  std::vector<Point2d> points;
  std::vector<Point2d> free_points;
  std::vector<GridIndex> grid_indices;
  std::vector<Point2d> feature_points;
  std::vector<FeatureVector> feature_points_features;
  Point2d centroid = Point2d::Zero();
  float z_value = 0.0f;
  Eigen::Vector2f direction = Eigen::Vector2f::Zero();
  std::optional<FeatureVector> feature;
  std::map<SemanticLabel, size_t> semantic_class_labels;
  bool in_active_window = false;
};
}  // namespace frontiers

struct DenseGrid {
  int min_r, min_c;
  int width, height;
  std::vector<CellState> cells;                        // row-major
  std::vector<std::optional<FeatureVector>> features;  // row-major

  size_t idx(int r, int c) const;

  bool inBounds(int r, int c) const;

  CellState at(int r, int c) const;

  static DenseGrid makeDense(const OccupancyGrid& g);
};

class OccupancyFrontierExtractor {
 public:
  enum DenseCellState {
    UNCHECKED,
    MAP_OPEN,
    MAP_CLOSED,
    FRONTIER_OPEN,
    FRONTIER_CLOSED
  };
  enum class FeaturesCollectionMode {
    FRONTIER_POINTS,
    CLUSTER_RADIUS,
    FRONTIER_POINTS_RADIUS
  };
  enum class DetectionMode { WFD, GLOBAL };

  using Sink = OutputSink<uint64_t,
                          const Eigen::Isometry3d&,
                          const std::vector<frontiers::FrontierCluster>&>;
  using Ptr = std::unique_ptr<OccupancyFrontierExtractor>;
  struct Config {
    // General config
    char prefix = 'f';
    std::string nav_layer = spark_dsg::DsgLayers::PLACES;
    std::vector<Sink::Factory> sinks;
    double object_connection_max_distance = 2.0;
    // Frontier clustering
    double min_cluster_size = 0.5;
    // Wavefront detection config
    bool use_initial_pose = true;
    DetectionMode detection_mode = DetectionMode::WFD;
    bool local_frontiers = false;
    float max_radius_m = 9.0;
    // Features config
    FeaturesCollectionMode features_collection_mode =
        FeaturesCollectionMode::FRONTIER_POINTS;
    float features_radius = 1.0f;
    // Use nearest navigation node as centroid
    bool set_centroid_to_nearest_nav_node = false;
    // Robot inflation radius (for frontier point selection)
    float inflation_radius_m = 0.5f;
  } const config;

  explicit OccupancyFrontierExtractor(const Config& config);

  virtual ~OccupancyFrontierExtractor() = default;

  void detectRealFrontiers(const ActiveWindowOutput& input);

  void addRealFrontiers(uint64_t timestamp_ns,
                        DynamicSceneGraph& graph,
                        const Eigen::Isometry3d& world_T_body);

 protected:
  void addEdges(const NodeSymbol& id,
                const Eigen::Isometry3d& world_T_body,
                uint64_t timestamp_ns,
                DynamicSceneGraph& graph,
                frontiers::FrontierCluster& cluster);
  void extractFrontiersWFD(const OccupancyGrid::Ptr& occupancy_grid,
                           const Eigen::Vector3d& robot_pos);
  void extractFrontiersGlobal(const OccupancyGrid::Ptr& occupancy_grid,
                              const Eigen::Vector3d& robot_pos);
  void selectFreeFrontier(frontiers::FrontierCluster& cluster) const;
  void computeClusterDirection(const OccupancyGrid::Ptr& occupancy_grid,
                               frontiers::FrontierCluster& cluster) const;

  frontiers::FrontierCluster buildFrontier(
      const OccupancyGrid::Ptr& occupancy_grid,
      size_t initial_cell,
      std::vector<DenseCellState>& cell_states,
      const DenseGrid& costmap,
      const std::function<bool(size_t, size_t&)>& is_frontier_cell,
      const double& robot_z) const;
  void computeClustersSemanticsFP(const OccupancyGrid::Ptr& occupancy_grid);
  void computeClustersSemanticsRadius(const OccupancyGrid::Ptr& occupancy_grid);
  void computeClustersSemanticsFPRadius(const OccupancyGrid::Ptr& occupancy_grid);
  void computeEDT(const DenseGrid& grid,
                  const float& resolution,
                  std::vector<float>& edt) const;

  Eigen::Vector3d initial_robot_position_;
  bool initial_position_set_ = false;

  Sink::List sinks_;
  std::vector<frontiers::FrontierCluster> clusters_;
  std::vector<frontiers::FrontierPoint> frontier_points_;
  std::unique_ptr<NearestNodeFinder> nn_finder_;
  size_t last_num_frontiers_ = 0;

  const int dr_[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
  const int dc_[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
};

void declare_config(OccupancyFrontierExtractor::Config& config);

}  // namespace hydra
