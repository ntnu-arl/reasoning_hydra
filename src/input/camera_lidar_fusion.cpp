#include "hydra/input/camera_lidar_fusion.h"

namespace hydra {

void declare_config(CameraLidarFusion::Config& config) {
  using namespace config;
  name("CameraLidarFusion");
  base<Sensor::Config>(config);
  field(config.width, "width", "px");
  field(config.height, "height", "px");
  field(config.cx, "cx", "px");
  field(config.cy, "cy", "px");
  field(config.fx, "fx", "px");
  field(config.fy, "fy", "px");

  check(config.width, GT, 0, "width");
  check(config.height, GT, 0, "height");
  check(config.cx, GT, 0, "cx");
  check(config.cy, GT, 0, "cy");
  check(config.fx, GT, 0, "fx");
  check(config.fy, GT, 0, "fy");
  checkCondition(config.cx <= config.width, "param 'cx' is expected <= 'width'");
  checkCondition(config.cy <= config.height, "param 'cy' is expected <= 'height'");
}

CameraLidarFusion::CameraLidarFusion(const Config& config)
    : Sensor(config), config_(config::checkValid(config)) {
  // Pre-compute the view frustum (top, right, bottom, left, plane normals).
  const auto scale_factor = config_.fx / config_.fy;
  Eigen::Vector3f p1(-config_.cx, -config_.cy * scale_factor, config_.fx);
  Eigen::Vector3f p2(
      config_.width - config_.cx, -config_.cy * scale_factor, config_.fx);
  view_frustum_.row(0) = p1.cross(p2).normalized();

  p1 = Eigen::Vector3f(config_.width - config_.cx,
                       (config_.height - config_.cy) * scale_factor,
                       config_.fx);
  view_frustum_.row(1) = p2.cross(p1).normalized();

  p2 = Eigen::Vector3f(
      -config_.cx, (config_.height - config_.cy) * scale_factor, config_.fx);
  view_frustum_.row(2) = p1.cross(p2).normalized();

  p1 = Eigen::Vector3f(-config_.cx, -config_.cy * scale_factor, config_.fx);
  view_frustum_.row(3) = p2.cross(p1).normalized();
}

float CameraLidarFusion::computeRayDensity(float voxel_size, float depth) const {
  return config_.fx * config_.fy * std::pow(voxel_size / depth, 2.f);
}

bool CameraLidarFusion::finalizeRepresentations(InputData& input,
                                                bool force_world_frame) const {
  if (input.vertex_map.empty()) {
    LOG(ERROR) << "pointcloud required to finalize data!";
    return false;
  }

  // TODO(nathan) check that input is normalized

  if (input.vertex_map.size() != input.label_image.size()) {
    LOG(ERROR) << "label input size does not match pointcloud!";
    return false;
  }

  if (!input.color_image.empty() &&
      input.vertex_map.size() != input.color_image.size()) {
    LOG(ERROR) << "color input size does not match pointcloud!";
    return false;
  }

  if (force_world_frame && !input.points_in_world_frame) {
    LOG(ERROR) << "force_world_frame is not supported for CameraLidarFusion!";
    return false;
  }

  const auto sensor_T_world = input.getSensorPose().cast<float>().inverse();
  input.min_range = std::numeric_limits<float>::max();
  input.max_range = std::numeric_limits<float>::lowest();

  bool has_panoptic = input.features_mask.has_value();
  input.range_image = cv::Mat(config_.height, config_.width, CV_32FC1, 0.0f);
  cv::Mat labels(config_.height, config_.width, CV_32SC1, -1);
  cv::Mat color(config_.height, config_.width, CV_8UC3);
  color = 0;
  cv::Mat panoptic;
  if (has_panoptic) {
    panoptic = cv::Mat(config_.height, config_.width, CV_16UC1, 0);
  }

  auto point_iter = input.vertex_map.begin<cv::Vec3f>();
  auto label_iter = input.label_image.begin<int32_t>();
  auto color_iter = input.color_image.begin<cv::Vec3b>();
  size_t num_invalid = 0;
  size_t panoptic_index = 0;

  while (point_iter != input.vertex_map.end<cv::Vec3f>()) {
    int u, v;
    const auto& p = *point_iter;
    Eigen::Vector3f p_C(p[0], p[1], p[2]);
    if (input.points_in_world_frame) {
      p_C = sensor_T_world * p_C;
    }

    if (!projectPointToImagePlane(p_C, u, v)) {
      ++num_invalid;
      ++point_iter;
      ++label_iter;
      ++panoptic_index;
      continue;
    }

    const auto range_m = p_C.norm();
    input.min_range = std::min(input.min_range, range_m);
    input.max_range = std::max(input.max_range, range_m);

    input.range_image.at<float>(v, u) = p_C.norm();
    labels.at<int32_t>(v, u) = *label_iter;
    color.at<cv::Vec3b>(v, u) = *color_iter;
    if (has_panoptic) {
      panoptic.at<uint16_t>(v, u) = (*input.features_mask).at<uint16_t>(panoptic_index);
    }

    ++point_iter;
    ++label_iter;
    ++panoptic_index;
  }

  size_t total_lidar = input.vertex_map.rows * input.vertex_map.cols;
  double percent_invalid = static_cast<double>(num_invalid) / total_lidar;
  VLOG(5) << "Converted lidar points! invalid: " << num_invalid << " / " << total_lidar
          << " (percent: " << percent_invalid << ")";
  input.label_image = labels;
  input.color_image = color;
  input.features_mask = panoptic;
  return true;
}

bool CameraLidarFusion::projectPointToImagePlane(const Eigen::Vector3f& p_C,
                                                 float& u,
                                                 float& v) const {
  if (p_C.z() <= 0.f) {
    return false;
  }

  // all points are considered valid as long as the are contained in the image plane
  // with bounds [0, w] x [0, h]
  u = p_C.x() * config_.fx / p_C.z() + config_.cx;
  if (u > config_.width || u < 0) {
    return false;
  }

  v = p_C.y() * config_.fy / p_C.z() + config_.cy;
  if (v > config_.height || v < 0) {
    return false;
  }

  return true;
}

bool CameraLidarFusion::projectPointToImagePlane(const Eigen::Vector3f& p_C,
                                                 int& u,
                                                 int& v) const {
  float u_float = -1.0f;
  float v_float = -1.0f;
  if (!projectPointToImagePlane(p_C, u_float, v_float)) {
    return false;
  }

  u = std::floor(u_float);
  v = std::floor(v_float);
  if (u >= config_.width || u < 0 || v >= config_.height || v < 0) {
    return false;
  }

  return true;
}

bool CameraLidarFusion::pointIsInViewFrustum(const Eigen::Vector3f& point_C,
                                             float inflation_distance) const {
  if (point_C.z() < -inflation_distance) {
    return false;
  }

  if (point_C.norm() > config_.max_range + inflation_distance) {
    return false;
  }

  for (int i = 0; i < view_frustum_.rows(); ++i) {
    if (point_C.dot(view_frustum_.row(i)) < -inflation_distance) {
      return false;
    }
  }

  return true;
}

}  // namespace hydra
