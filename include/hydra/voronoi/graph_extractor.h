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

#include <spark_dsg/node_symbol.h>

#include <memory>
#include <opencv2/core/mat.hpp>

#include "hydra/reconstruction/occupancy_grid.h"
#include "hydra/voronoi/segment.h"

using namespace spark_dsg;

namespace hydra {
namespace voronoi {

struct Vector2dHash {
  size_t operator()(const Eigen::Vector2d& v) const {
    auto hx = std::hash<double>()(v.x());
    auto hy = std::hash<double>()(v.y());
    return hx ^ (hy << 1);
  }
};

struct Vector2dEqual {
  bool operator()(const Eigen::Vector2d& a, const Eigen::Vector2d& b) const {
    return a.isApprox(b, 1e-6);
  }
};

struct Node {
  NodeId id;
  float width;
};

struct Edge {
  NodeId source;
  NodeId target;
  double weight;
};

struct Graph {
  using Ptr = std::shared_ptr<Graph>;
  std::unordered_map<Eigen::Vector2d, Node, Vector2dHash, Vector2dEqual> nodes;
  std::vector<Edge> edges;

  Graph() = default;
};

class GraphExtractor {
 public:
  using Ptr = std::unique_ptr<GraphExtractor>;

  struct Config {
    char prefix = 't';
    double map_inflation = 0.1;
    float segment_length = 0.5;
    float crossing_optimization = 0.2;
    float end_segmentation_optimization = 0.2;
  } const config;

  explicit GraphExtractor(const Config& config);

  virtual ~GraphExtractor() = default;

  void extract(const OccupancyGrid::Ptr& grid);

  const Graph::Ptr& getGraph() const { return graph_; }

  const Eigen::Vector2d& getOrigin() const { return origin_; }

  const std::vector<Segment>& getSegments() const { return segments_; }

 protected:
  void prepareMap(const cv::Mat& grid_map, cv::Mat& map, const float& resolution) const;

  void computeVoronoiMap(const cv::Mat& distance_map, cv::Mat& voronoi_map) const;

  void generateSegments(cv::Mat& map,
                        cv::Mat& distance_map,
                        cv::Mat& voronoi_map,
                        const float& resolution,
                        std::vector<Segment>& _segments) const;

 private:
  Graph::Ptr graph_;
  std::vector<Segment> segments_;
  Eigen::Vector2d origin_;
};

void declare_config(GraphExtractor::Config& config);

}  // namespace voronoi
}  // namespace hydra
