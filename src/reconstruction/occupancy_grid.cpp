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
#include "hydra/reconstruction/occupancy_grid.h"

#include <opencv2/core.hpp>

namespace hydra {

OccupancyGrid::GridData::GridData()
    : state(CellState::Unknown),
      features(std::nullopt),
      semantic_class_labels(),
      observation_count(0) {}

void OccupancyGrid::GridBounds::update(const int& r, const int& c) {
  if (r < min_row) min_row = r;
  if (r > max_row) max_row = r;
  if (c < min_col) min_col = c;
  if (c > max_col) max_col = c;
  width = static_cast<size_t>(max_col - min_col + 1);
  height = static_cast<size_t>(max_row - min_row + 1);
}

GridIndex OccupancyGrid::worldToGrid(const Point2d& p) const {
  int col = static_cast<int>(std::floor(p.x() / resolution));
  int row = static_cast<int>(std::floor(p.y() / resolution));
  return {row, col};
}

Point2d OccupancyGrid::gridToWorld(const GridIndex& idx) const {
  const int row = idx.first;
  const int col = idx.second;

  // Return cell center in world coordinates
  float x = (static_cast<float>(col) + 0.5f) * resolution;
  float y = (static_cast<float>(row) + 0.5f) * resolution;

  return Point2d(x, y);
}

cv::Mat OccupancyGrid::toCvMat() const {
  if (grid.empty() || bounds.width == 0 || bounds.height == 0) {
    return cv::Mat();
  }

  cv::Mat mat(static_cast<int>(bounds.height),
              static_cast<int>(bounds.width),
              CV_8SC1,
              cv::Scalar(static_cast<int8_t>(CellState::Unknown)));

  // Fill from hash map
  for (const auto& [idx, cell] : grid) {
    const int r = idx.first;
    const int c = idx.second;

    const int row = r - bounds.min_row;
    const int col = c - bounds.min_col;

    if (row >= 0 && row < mat.rows && col >= 0 && col < mat.cols) {
      mat.at<int8_t>(row, col) = static_cast<int8_t>(cell.state);
    }
  }

  return mat;
}

}  // namespace hydra
