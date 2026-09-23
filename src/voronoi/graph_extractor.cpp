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
#include "hydra/voronoi/graph_extractor.h"

#include <config_utilities/config.h>
#include <config_utilities/validation.h>
#include <glog/logging.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "hydra/voronoi/segment_expander.h"
#include "hydra/voronoi/thinning.h"

namespace hydra {
namespace voronoi {

void declare_config(GraphExtractor::Config& config) {
  using namespace config;
  name("GraphExtractorConfig");
  field(config.prefix, "prefix");
  field(config.map_inflation, "map_inflation");
  field(config.segment_length, "segment_length");
  field(config.crossing_optimization, "crossing_optimization");
  field(config.end_segmentation_optimization, "end_segmentation_optimization");
}

GraphExtractor::GraphExtractor(const Config& config)
    : config(config::checkValid(config)), graph_(std::make_shared<Graph>()) {}

void GraphExtractor::extract(const OccupancyGrid::Ptr& grid) {
  graph_.reset(new Graph());
  // Prepare the map for processing
  cv::Mat grid_map = grid->toCvMat();
  cv::Mat map;
  prepareMap(grid_map, map, grid->resolution);

  // Distance field
  cv::Mat distance_map;
  cv::distanceTransform(map, distance_map, cv::DIST_L2, 3);

  // Compute the Voronoi map
  cv::Mat voronoi_map;
  computeVoronoiMap(distance_map, voronoi_map);

  // Generate segments from the Voronoi map
  segments_.clear();
  generateSegments(grid_map, distance_map, voronoi_map, grid->resolution, segments_);

  const float origin_x = static_cast<float>(grid->bounds.min_col) * grid->resolution;
  const float origin_y = static_cast<float>(grid->bounds.min_row) * grid->resolution;
  origin_ = Eigen::Vector2d(origin_x, origin_y);

  // Convert segments to graph
  for (const auto& segment : segments_) {
    const auto pose_source = segment.getPath().front() * grid->resolution + origin_;
    const auto pose_target = segment.getPath().back() * grid->resolution + origin_;
    if (graph_->nodes.count(pose_source) == 0) {
      NodeSymbol source_id(config.prefix, static_cast<NodeId>(graph_->nodes.size()));
      graph_->nodes[pose_source] =
          Node{source_id, segment.getMinPathSpace() * grid->resolution};
    } else {
      graph_->nodes[pose_source].width =
          std::min(graph_->nodes[pose_source].width,
                   segment.getMinPathSpace() * grid->resolution);
    }

    if (graph_->nodes.count(pose_target) == 0) {
      NodeSymbol target_id(config.prefix, static_cast<NodeId>(graph_->nodes.size()));
      graph_->nodes[pose_target] = Node{target_id, 0.0f};
    }
    graph_->edges.emplace_back(Edge{graph_->nodes[pose_source].id,
                                    graph_->nodes[pose_target].id,
                                    static_cast<double>(segment.getLength())});
  }
  // if (graph_->nodes.empty()) {
  //   LOG(WARNING)
  //       << "Extracted graph is empty. Check the input occupancy grid and
  //       parameters.";
  //   cv::imwrite("/developer/ros2_hydra_ws/grid_map.png", grid_map);
  //   cv::imwrite("/developer/ros2_hydra_ws/prepared_map.png", map);
  //   cv::imwrite("/developer/ros2_hydra_ws/distance_map.png", distance_map);
  //   cv::imwrite("/developer/ros2_hydra_ws/voronoi_map.png", voronoi_map);
  //   LOG(INFO) << "Resolution: " << grid->resolution;
  //   LOG(INFO) << "Origin: " << origin_.transpose();
  // }
}

void GraphExtractor::prepareMap(const cv::Mat& grid_map,
                                cv::Mat& map,
                                const float& resolution) const {
  static cv::Mat src_map;
  grid_map.convertTo(src_map, CV_8UC1);
  int erode_size = static_cast<int>(config.map_inflation / resolution);

  for (int i = 0; i < src_map.cols * src_map.rows; i++) {
    if ((signed char)grid_map.data[i] < 0) {
      src_map.data[i] = 100;
    }
  }

  map = src_map;

  cv::bitwise_not(src_map, src_map);
  cv::threshold(src_map, map, 10, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

  if (erode_size <= 0) {
    erode_size = 1;
  }
  if (erode_size > 0) {
    cv::Mat element =
        cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                  cv::Size(2 * erode_size + 1, 2 * erode_size + 1),
                                  cv::Point(erode_size, erode_size));

    cv::erode(map, map, element);
  }
}

void GraphExtractor::computeVoronoiMap(const cv::Mat& distance_map,
                                       cv::Mat& voronoi_map) const {
  cv::Mat src_map = distance_map;
  src_map.convertTo(voronoi_map, CV_8UC1, 0.0);

  voronoi::greyscaleThinning(src_map, voronoi_map);
  cv::threshold(voronoi_map, voronoi_map, 1, 255, cv::THRESH_BINARY);
  voronoi::sceletonize(voronoi_map, voronoi_map);
}

void GraphExtractor::generateSegments(cv::Mat& map,
                                      cv::Mat& distance_map,
                                      cv::Mat& voronoi_map,
                                      const float& resolution,
                                      std::vector<Segment>& _segments) const {
  SegmentExpander exp;
  exp.Initialize(map, distance_map, voronoi_map);

  std::unique_ptr<float[]> potential(new float[map.cols * map.rows]);
  std::vector<std::vector<Eigen::Vector2d>> points = exp.calcEndpoints(potential.get());
  std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> segments;

  int nx = map.cols;
  int ny = map.rows;

  exp.Initialize(map, distance_map, voronoi_map);
  std::fill(potential.get(), potential.get() + nx * ny, -1);

  std::vector<Segment> segs =
      exp.getGraph(points,
                   potential.get(),
                   config.segment_length / resolution,
                   config.crossing_optimization / resolution,
                   config.end_segmentation_optimization / resolution);

  for (uint32_t i = 0; i < segs.size(); i++) {
    std::vector<uint32_t> predecessors = segs[i].getPredecessors();
    std::vector<uint32_t> successors = segs[i].getSuccessors();
  }

  _segments = std::move(segs);
}

}  // namespace voronoi
}  // namespace hydra
