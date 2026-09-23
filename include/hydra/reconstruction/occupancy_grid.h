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

#include <Eigen/Core>
#include <map>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <optional>

#include "hydra/openset/openset_types.h"
#include "hydra/reconstruction/volumetric_map.h"

namespace hydra {

using Point2d = Eigen::Vector2f;
using GridIndex = std::pair<int, int>;
using SemanticLabel = uint32_t;

enum class CellState : int8_t {
  Unknown = -1,
  ObservedFree = 0,
  ObservedOccupied = 100,
  OutOfBounds = 50
};

struct GridIndexHash {
  size_t operator()(const GridIndex& p) const {
    return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1);
  }
};

struct OccupancyGrid {
  using Ptr = std::shared_ptr<OccupancyGrid>;
  struct GridData {
    CellState state;
    std::optional<FeatureVector> features;
    std::map<SemanticLabel, size_t> semantic_class_labels;
    size_t observation_count = 0;

    GridData();
  };
  std::unordered_map<GridIndex, GridData, GridIndexHash> grid;
  float resolution = 0.0f;
  struct GridBounds {
    int min_row = std::numeric_limits<int>::max();
    int max_row = std::numeric_limits<int>::min();
    int min_col = std::numeric_limits<int>::max();
    int max_col = std::numeric_limits<int>::min();
    size_t width = 0;
    size_t height = 0;

    void update(const int& r, const int& c);
  } bounds;

  bool empty() const { return grid.empty(); }

  GridIndex worldToGrid(const Point2d& p) const;

  Point2d gridToWorld(const GridIndex& idx) const;

  cv::Mat toCvMat() const;
};

namespace occupancy {

static inline bool isObserved(const TsdfVoxel& v, double w) {
  return v.weight >= w && v.distance >= 0;
}

static inline bool isHardUnknown(const TsdfVoxel& v, double w) {
  return v.weight >= w && v.distance < 0;
}

static inline bool isOccupied(const TsdfVoxel& v, double dist) {
  return std::abs(v.distance) < (static_cast<float>(dist) - 1e-3f);
}

}  // namespace occupancy
}  // namespace hydra
