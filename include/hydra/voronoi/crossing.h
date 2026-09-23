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

#include <Eigen/Dense>
#include <memory>
#include <vector>

#include "hydra/voronoi/segment.h"

namespace hydra {
namespace voronoi {
class Crossing {
 public:
  /**
   * @brief constructor
   * @param _segment_points the endpoint of each segment in the crossing, which is on
   * the crossing side
   */
  explicit Crossing(const std::vector<Eigen::Vector2d>& _segment_points);
  /**
   * @brief tries to add the segment to the crossing and adds the given segment to each
   * segments neighbors in the crossing
   * @param _seg the segment to add
   * @return if the action succeeded
   */
  bool tryAddSegment(Segment& _seg);
  /**
   * @brief returns the center of the crossing (average of all pts)
   * @returns the center of the crossing
   */
  Eigen::Vector2d getCenter() const;
  /**
   * @brief saves the reference to the vector containing all segments to have access to
   * them to alter other segments from try Add segments
   */
  void setSegmentReference(const std::shared_ptr<std::vector<Segment>>& segs);

 private:
  std::vector<Eigen::Vector2d> surrounding_points_;
  std::vector<uint32_t> segments_start_;
  std::vector<uint32_t> segments_end_;
  Eigen::Vector2d center_;
  std::shared_ptr<std::vector<Segment>> segment_reference_;
};

}  // namespace voronoi
}  // namespace hydra
