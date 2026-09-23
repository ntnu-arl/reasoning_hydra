// The contents of this file are originally from Panoptic-Mapping,
// under the following license:
//
// BSD 3-Clause License
// Copyright (c) 2021, ETHZ ASL
// All rights reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// See https://github.com/ethz-asl/panoptic_mapping for original code and paper
//
// Modifications (including work done by Lukas Schmid for Khronos) fall under the same
// license as Hydra and are subject to the following copyright and disclaimer:
//
// Copyright 2022 Massachusetts Institute of Technology.
// All Rights Reserved
//
// Research was sponsored by the United States Air Force Research Laboratory and
// the United States Air Force Artificial Intelligence Accelerator and was
// accomplished under Cooperative Agreement Number FA8750-19-2-1000. The views
// and conclusions contained in this document are those of the authors and should
// not be interpreted as representing the official policies, either expressed or
// implied, of the United States Air Force or the U.S. Government. The U.S.
// Government is authorized to reproduce and distribute reprints for Government
// purposes notwithstanding any copyright notation herein.
#include "hydra/reconstruction/projection_interpolators.h"

#include <config_utilities/config.h>
#include <config_utilities/factory.h>
#include <config_utilities/validation.h>

#include <Eigen/Core>
#include <cmath>
#include <limits>
#include <unordered_map>

namespace hydra {
namespace {

static const auto nearest_registration =
    config::RegistrationWithConfig<ProjectionInterpolator,
                                   InterpolatorNearest,
                                   InterpolatorNearest::Config>("nearest");

static const auto bilinear_registration =
    config::RegistrationWithConfig<ProjectionInterpolator,
                                   InterpolatorBilinear,
                                   InterpolatorBilinear::Config>("bilinear");

static const auto adaptive_registration =
    config::RegistrationWithConfig<ProjectionInterpolator,
                                   InterpolatorAdaptive,
                                   InterpolatorAdaptive::Config>("adaptive");

}  // namespace

using spark_dsg::Color;
using Weights = InterpolationWeights;

void declare_config(InterpolatorNearest::Config&) {
  config::name("InterpolatorNearest::Config");
}

Weights InterpolatorNearest::computeWeights(float u,
                                            float v,
                                            const cv::Mat& img) const {
  Weights weights(std::round(u), std::round(v));
  if (weights.u < 0 || weights.u >= img.cols || weights.v < 0 ||
      weights.v >= img.rows) {
    return weights;
  }

  weights.valid = true;
  return weights;
}

float InterpolatorNearest::interpolateRange(const cv::Mat& range_image,
                                            const Weights& weights) const {
  return range_image.at<float>(weights.v, weights.u);
}

Color InterpolatorNearest::interpolateColor(const cv::Mat& color_image,
                                            const Weights& weights) const {
  const cv::Vec3b color = color_image.at<cv::Vec3b>(weights.v, weights.u);
  return Color(color[0], color[1], color[2]);
}

int InterpolatorNearest::interpolateID(const cv::Mat& id_image,
                                       const Weights& weights) const {
  return id_image.at<int32_t>(weights.v, weights.u);
}

std::optional<FeatureVector> InterpolatorNearest::interpolateFeatures(
    const std::optional<cv::Mat>& panoptic_image,
    FeatureMap<int> semantic_features,
    const Weights& weights) const {
  if (!panoptic_image || semantic_features.empty()) {
    return std::nullopt;
  }

  const auto pan_id = panoptic_image.value().at<uint16_t>(weights.v, weights.u);

  if (pan_id == 0) {
    return std::nullopt;
  }
  return semantic_features.at(pan_id);
}

std::optional<uint16_t> InterpolatorNearest::interpolatePanoptic(
    const std::optional<cv::Mat>& panoptic_image, const Weights& weights) const {
  if (!panoptic_image) {
    return std::nullopt;
  }
  const auto pan_id = panoptic_image.value().at<uint16_t>(weights.v, weights.u);
  if (pan_id == 0) {
    return std::nullopt;
  }
  return pan_id;
}

std::optional<FeatureVector> InterpolatorNearest::interpolatePixelwiseFeature(
    const std::optional<cv::Mat>& pixelwise_feature_image,
    const Weights& weights,
    const int orig_height,
    const int orig_width) const {
  if (!pixelwise_feature_image) {
    return std::nullopt;
  }

  const cv::Mat& feature_image = pixelwise_feature_image.value();
  CV_Assert(feature_image.depth() == CV_32F);

  const int feat_h = feature_image.rows;
  const int feat_w = feature_image.cols;
  const int channels = feature_image.channels();

  // Map original coords -> feature map coords
  float scale_x = static_cast<float>(feat_w) / orig_width;
  float scale_y = static_cast<float>(feat_h) / orig_height;

  int u_feat = std::min(feat_w - 1, std::max(0, static_cast<int>(weights.u * scale_x)));
  int v_feat = std::min(feat_h - 1, std::max(0, static_cast<int>(weights.v * scale_y)));

  FeatureVector feature(channels);
  const float* pixel = feature_image.ptr<float>(v_feat, u_feat);

  for (int i = 0; i < channels; ++i) {
    feature[i] = pixel[i];
  }

  return feature;
}

void declare_config(InterpolatorBilinear::Config&) {
  config::name("InterpolatorBilinear::Config");
}

Weights InterpolatorBilinear::computeWeights(float u,
                                             float v,
                                             const cv::Mat& img) const {
  Weights weights(std::floor(u), std::floor(v));
  if (weights.u < 0 || weights.v < 0 || weights.u >= img.cols - 1 ||
      weights.v >= img.rows - 1) {
    return weights;
  }

  weights.valid = true;
  weights.use_bilinear = true;
  float du = u - static_cast<float>(weights.u);
  float dv = v - static_cast<float>(weights.v);
  weights.w0 = (1.f - du) * (1.f - dv);
  weights.w1 = (1.f - du) * dv;
  weights.w2 = du * (1.f - dv);
  weights.w3 = du * dv;
  return weights;
}

float InterpolatorBilinear::interpolateRange(const cv::Mat& range_image,
                                             const Weights& weights) const {
  return range_image.at<float>(weights.v, weights.u) * weights.w0 +
         range_image.at<float>(weights.v + 1, weights.u) * weights.w1 +
         range_image.at<float>(weights.v, weights.u + 1) * weights.w2 +
         range_image.at<float>(weights.v + 1, weights.u + 1) * weights.w3;
}

Color InterpolatorBilinear::interpolateColor(const cv::Mat& color_image,
                                             const Weights& weights) const {
  Eigen::Vector3f color(0, 0, 0);
  auto c1 = color_image.at<cv::Vec3b>(weights.v, weights.u);
  auto c2 = color_image.at<cv::Vec3b>(weights.v + 1, weights.u);
  auto c3 = color_image.at<cv::Vec3b>(weights.v, weights.u + 1);
  auto c4 = color_image.at<cv::Vec3b>(weights.v + 1, weights.u + 1);
  for (size_t i = 0; i < 3; ++i) {
    color[i] = c1[i] * weights.w0 + c2[i] * weights.w1 + c3[i] * weights.w2 +
               c4[i] * weights.w3;
  }

  return Color(color[0], color[1], color[2]);
}

int InterpolatorBilinear::interpolateID(const cv::Mat& id_image,
                                        const Weights& weights) const {
  // Since IDs can not be interpolated we assign weights to all IDs in the image
  // based on the corner weights and return the highest weights ID.
  // NOTE(nathan) this is not the same as just picking the maximum weight from the
  // pixels
  // TODO(nathan) consider manually implementing to avoid std::unordered_map memory
  // usage
  std::unordered_map<int, float> ids;  // These are zero initialized by default.
  ids[id_image.at<int32_t>(weights.v, weights.u)] += weights.w0;
  ids[id_image.at<int32_t>(weights.v + 1, weights.u)] += weights.w1;
  ids[id_image.at<int32_t>(weights.v, weights.u + 1)] += weights.w2;
  ids[id_image.at<int32_t>(weights.v + 1, weights.u + 1)] += weights.w3;
  return std::max_element(
             std::begin(ids),
             std::end(ids),
             [](const auto& p1, const auto& p2) { return p1.second < p2.second; })
      ->first;
}

std::optional<FeatureVector> InterpolatorBilinear::interpolateFeatures(
    const std::optional<cv::Mat>& panoptic_image,
    FeatureMap<int> semantic_features,
    const Weights& weights) const {
  if (!panoptic_image || semantic_features.empty()) {
    return std::nullopt;
  }

  const auto pan_id = panoptic_image.value().at<uint16_t>(weights.v, weights.u);
  if (pan_id == 0) {
    return std::nullopt;
  }

  const size_t feature_size = semantic_features.begin()->second.size();
  auto w0 = weights.w0;
  auto w1 = weights.w1;
  auto w2 = weights.w2;
  auto w3 = weights.w3;

  FeatureVector c0, c1, c2, c3;
  // Lambda to process weights and feature indices
  auto process_feature =
      [&](float& weight, int v_offset, int u_offset, FeatureVector& result) {
        auto& pan_id = panoptic_image.value().at<uint16_t>(weights.v + v_offset,
                                                           weights.u + u_offset);
        if (pan_id == 0 || semantic_features.count(pan_id) == 0) {
          weight = 0.f;
          result = FeatureVector::Zero(feature_size);
        } else {
          result = semantic_features.at(pan_id);
        }
      };

  // Apply the lambda for each weight and offset combination
  process_feature(w0, 0, 0, c0);
  process_feature(w1, 1, 0, c1);
  process_feature(w2, 0, 1, c2);
  process_feature(w3, 1, 1, c3);

  const float normalizer = w0 + w1 + w2 + w3;

  if (normalizer == 0.f) {
    return std::nullopt;
  }

  w0 /= normalizer;
  w1 /= normalizer;
  w2 /= normalizer;
  w3 /= normalizer;

  return w0 * c0 + w1 * c1 + w2 * c2 + w3 * c3;
}

std::optional<uint16_t> InterpolatorBilinear::interpolatePanoptic(
    const std::optional<cv::Mat>& panoptic_image, const Weights& weights) const {
  if (!panoptic_image) {
    return std::nullopt;
  }
  std::unordered_map<uint16_t, float> panoptic_weights;
  panoptic_weights[panoptic_image.value().at<uint16_t>(weights.v, weights.u)] +=
      weights.w0;
  panoptic_weights[panoptic_image.value().at<uint16_t>(weights.v + 1, weights.u)] +=
      weights.w1;
  panoptic_weights[panoptic_image.value().at<uint16_t>(weights.v, weights.u + 1)] +=
      weights.w2;
  panoptic_weights[panoptic_image.value().at<uint16_t>(weights.v + 1, weights.u + 1)] +=
      weights.w3;

  auto max_element = std::max_element(std::begin(panoptic_weights),
                                      std::end(panoptic_weights),
                                      [](const auto& p1, const auto& p2) {
                                        return p1.second < p2.second;
                                      })
                         ->first;

  if (max_element == 0) {
    return std::nullopt;
  }
  return max_element;
}

std::optional<FeatureVector> InterpolatorBilinear::interpolatePixelwiseFeature(
    const std::optional<cv::Mat>& pixelwise_feature_image,
    const Weights& weights,
    const int orig_height,
    const int orig_width) const {
  if (!pixelwise_feature_image) {
    return std::nullopt;
  }

  const cv::Mat& feature_image = pixelwise_feature_image.value();
  CV_Assert(feature_image.depth() == CV_32F);

  const int feat_h = feature_image.rows;
  const int feat_w = feature_image.cols;
  const int channels = feature_image.channels();

  float scale_x = static_cast<float>(feat_w) / orig_width;
  float scale_y = static_cast<float>(feat_h) / orig_height;

  float fx = weights.u * scale_x;
  float fy = weights.v * scale_y;

  int x0 = static_cast<int>(fx);
  int y0 = static_cast<int>(fy);
  int x1 = std::min(x0 + 1, feat_w - 1);
  int y1 = std::min(y0 + 1, feat_h - 1);

  float dx = fx - x0;
  float dy = fy - y0;

  x0 = std::clamp(x0, 0, feat_w - 1);
  y0 = std::clamp(y0, 0, feat_h - 1);

  const float* p00 = feature_image.ptr<float>(y0, x0);
  const float* p10 = feature_image.ptr<float>(y1, x0);
  const float* p01 = feature_image.ptr<float>(y0, x1);
  const float* p11 = feature_image.ptr<float>(y1, x1);

  FeatureVector feature(channels);
  for (int i = 0; i < channels; ++i) {
    feature[i] = (1 - dx) * (1 - dy) * p00[i] + (1 - dx) * dy * p10[i] +
                 dx * (1 - dy) * p01[i] + dx * dy * p11[i];
  }

  return feature;
}

void declare_config(InterpolatorAdaptive::Config& config) {
  using namespace config;
  config::name("InterpolatorAdaptive::Config");
  field(config.max_depth_difference_m, "max_depth_difference_m", "m");
  check(config.max_depth_difference_m, GE, 0.0f, "max_depth_difference_m");
}

InterpolatorAdaptive::InterpolatorAdaptive(const Config& config)
    : InterpolatorBilinear({}), config(config::checkValid(config)) {}

Weights InterpolatorAdaptive::computeWeights(float u,
                                             float v,
                                             const cv::Mat& ranges) const {
  Weights weights(std::floor(u), std::floor(v));
  // Check max range difference.
  bool use_nearest = false;
  float min = std::numeric_limits<float>::max();
  float max = std::numeric_limits<float>::lowest();
  for (size_t i = 0; i < 4; ++i) {
    const auto curr_v = weights.v + v_offset_[i];
    const auto curr_u = weights.u + u_offset_[i];
    if (curr_v < 0 || curr_u < 0 || curr_v >= ranges.rows || curr_u >= ranges.cols) {
      continue;
    }

    const float range = ranges.at<float>(curr_v, curr_u);
    max = std::max(range, max);
    min = std::min(range, min);
    if (max - min > config.max_depth_difference_m) {
      use_nearest = true;
      break;
    }
  }

  if (use_nearest) {
    Weights weights(std::round(u), std::round(v));
    weights.valid = weights.u >= 0 && weights.u < ranges.cols && weights.v >= 0 &&
                    weights.v < ranges.rows;
    return weights;
  }

  return InterpolatorBilinear::computeWeights(u, v, ranges);
}

float InterpolatorAdaptive::interpolateRange(const cv::Mat& range_image,
                                             const Weights& weights) const {
  if (weights.use_bilinear) {
    return InterpolatorBilinear::interpolateRange(range_image, weights);
  }

  return range_image.at<float>(weights.v, weights.u);
}

Color InterpolatorAdaptive::interpolateColor(const cv::Mat& color_image,
                                             const Weights& weights) const {
  if (weights.use_bilinear) {
    return InterpolatorBilinear::interpolateColor(color_image, weights);
  }

  auto color = color_image.at<cv::Vec3b>(weights.v, weights.u);
  return Color(color[0], color[1], color[2]);
}

int InterpolatorAdaptive::interpolateID(const cv::Mat& id_image,
                                        const Weights& weights) const {
  if (weights.use_bilinear) {
    return InterpolatorBilinear::interpolateID(id_image, weights);
  }

  return id_image.at<int32_t>(weights.v, weights.u);
}

std::optional<FeatureVector> InterpolatorAdaptive::interpolateFeatures(
    const std::optional<cv::Mat>& panoptic_image,
    FeatureMap<int> semantic_features,
    const Weights& weights) const {
  if (weights.use_bilinear) {
    return InterpolatorBilinear::interpolateFeatures(
        panoptic_image, semantic_features, weights);
  }
  if (!panoptic_image || semantic_features.empty()) {
    return std::nullopt;
  }
  const auto pan_id = panoptic_image.value().at<uint16_t>(weights.v, weights.u);
  if (pan_id == 0) {
    return std::nullopt;
  }
  return semantic_features.at(pan_id);
}

std::optional<uint16_t> InterpolatorAdaptive::interpolatePanoptic(
    const std::optional<cv::Mat>& panoptic_image, const Weights& weights) const {
  if (!panoptic_image) {
    return std::nullopt;
  }

  if (weights.use_bilinear) {
    return InterpolatorBilinear::interpolatePanoptic(panoptic_image, weights);
  }

  const auto pan_id = panoptic_image.value().at<uint16_t>(weights.v, weights.u);
  if (pan_id == 0) {
    return std::nullopt;
  }
  return pan_id;
}

std::optional<FeatureVector> InterpolatorAdaptive::interpolatePixelwiseFeature(
    const std::optional<cv::Mat>& pixelwise_feature_image,
    const Weights& weights,
    const int orig_height,
    const int orig_width) const {
  if (!pixelwise_feature_image) {
    return std::nullopt;
  }
  if (weights.use_bilinear) {
    return InterpolatorBilinear::interpolatePixelwiseFeature(
        pixelwise_feature_image, weights, orig_height, orig_width);
  }

  const cv::Mat& feature_image = pixelwise_feature_image.value();
  CV_Assert(feature_image.depth() == CV_32F);

  const int feat_h = feature_image.rows;
  const int feat_w = feature_image.cols;
  const int channels = feature_image.channels();

  // Map original coords -> feature map coords
  float scale_x = static_cast<float>(feat_w) / orig_width;
  float scale_y = static_cast<float>(feat_h) / orig_height;

  int u_feat = std::min(feat_w - 1, std::max(0, static_cast<int>(weights.u * scale_x)));
  int v_feat = std::min(feat_h - 1, std::max(0, static_cast<int>(weights.v * scale_y)));

  FeatureVector feature(channels);
  const float* pixel = feature_image.ptr<float>(v_feat, u_feat);

  for (int i = 0; i < channels; ++i) {
    feature[i] = pixel[i];
  }

  return feature;
}

}  // namespace hydra
