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

#include <opencv2/core/core.hpp>
#include <queue>

namespace hydra {
namespace voronoi {

/**
 * @brief Perform one thinning iteration.(Normally you wouldn't call this function
 * directly from your code)
 * @param im    Binary image with range = [0,1]
 * @param iter  0=even, 1=odd
 */
void sceletonizeIteration(cv::Mat& img, int iter);
/**
 * @brief Function for thinning the given binary image	(Paper Zhang-Suen Thinning)
 * @param src  The source image, binary with range = [0,255]
 * @param dst  The destination image
 */
void sceletonize(const cv::Mat& src, cv::Mat& dst);

/**
 * @brief Function for finding the ridge of a distance graph
 * @param src	the source image containing the distance field (Mat float)
 * @param dst 	the destination image containing 0 for non voronoi graph pixels (else
 * voronoigraph) (Mat uint8_t)
 */
void greyscaleThinning(const cv::Mat& src, cv::Mat& dst);

class Index {
 public:
  Index(int x, int y, float pot) {
    i = x;
    j = y;
    potential = pot;
  }
  Index offset(int x, int y) { return Index(i + x, j + y, potential); }
  int i;
  int j;
  float potential;
};

/**
 * @brief function for finding the maximum Neighbour ignoring the last detected pixels
 * @param i the row of the pixel
 * @param j the column of the pixel
 * @param src	the source image containing the distance field (Mat float)
 * @param dst 	the destination image containing 0 for non voronoi graph pixels (else
 * voronoigraph) (Mat uint8_t)
 */
Index getMaximumNeighbour(int i, int j, const cv::Mat& src, cv::Mat& dst);

}  // namespace voronoi
}  // namespace hydra
