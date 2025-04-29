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
  field(config.k1, "k1", "px");
  field(config.k2, "k2", "px");
  field(config.k3, "k3", "px");
  field(config.k4, "k4", "px");
  field(config.undistort, "undistort");

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

  if (!input.label_image.empty() && input.label_image.type() != CV_32SC1) {
    LOG(ERROR) << "label_image must be CV_8UC3!";
    return false;
  }
  if (!input.color_image.empty() && input.color_image.type() != CV_8UC3) {
    LOG(ERROR) << "color_image must be CV_8UC3!";
    return false;
  }

  // TODO(nathan) check that input is normalized

  if (!input.label_image.empty() && (input.label_image.rows != input.vertex_map.rows ||
                                     input.label_image.cols != input.vertex_map.cols)) {
    LOG(ERROR) << "color input dimensions do not match pointcloud!";
    return false;
  }

  if (!input.color_image.empty() && (input.color_image.rows != input.vertex_map.rows ||
                                     input.color_image.cols != input.vertex_map.cols)) {
    LOG(ERROR) << "color input dimensions do not match pointcloud!";
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
  cv::Size size(config_.width, config_.height);
  input.range_image = cv::Mat::zeros(size, CV_32FC1);
  cv::Mat labels = -cv::Mat::ones(size, CV_32SC1);
  cv::Mat colors = cv::Mat::zeros(size, CV_8UC3);

  if (has_panoptic && (*input.features_mask).type() != CV_16UC1) {
    LOG(ERROR) << "features_mask must be CV_16UC1!";
    return false;
  }
  cv::Mat panoptic;
  if (has_panoptic) {
    panoptic = cv::Mat::zeros(size, CV_16UC1);
  }

  size_t num_invalid = 0;
  for (int row = 0; row < input.vertex_map.rows; ++row) {
    for (int col = 0; col < input.vertex_map.cols; ++col) {
      int u, v;
      const auto& p = input.vertex_map.at<cv::Vec3f>(row, col);
      Eigen::Vector3f p_C(p[0], p[1], p[2]);
      if (input.points_in_world_frame) {
        p_C = sensor_T_world * p_C;
      }

      if (!projectPointToImagePlane(p_C, u, v)) {
        ++num_invalid;
        continue;
      }

      const auto range_m = p_C.norm();
      input.min_range = std::min(input.min_range, range_m);
      input.max_range = std::max(input.max_range, range_m);

      input.range_image.at<float>(v, u) = p_C.norm();
      labels.at<int32_t>(v, u) = input.label_image.at<int32_t>(row, col);
      colors.at<cv::Vec3b>(v, u) = input.color_image.at<cv::Vec3b>(row, col);
      if (has_panoptic) {
        panoptic.at<uint16_t>(v, u) = (*input.features_mask).at<uint16_t>(row, col);
      }
    }
  }

  size_t total_lidar = input.vertex_map.rows * input.vertex_map.cols;
  double percent_invalid = static_cast<double>(num_invalid) / total_lidar;
  VLOG(5) << "Converted lidar points! invalid: " << num_invalid << " / " << total_lidar
          << " (percent: " << percent_invalid << ")";
  input.label_image = labels;
  input.color_image = colors;
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
  if (u >= config_.width || u < 0) {
    return false;
  }

  v = p_C.y() * config_.fy / p_C.z() + config_.cy;
  if (v >= config_.height || v < 0) {
    return false;
  }

  // Apply distortion
  if (!config_.undistort) {
    return true;
  }
  float r2 = (p_C.x() * p_C.x() + p_C.y() * p_C.y()) / (p_C.z() * p_C.z());
  float distortion_factor =
      (1 + config_.k1 * r2 + config_.k2 * r2 * r2 + config_.k3 * r2 * r2 * r2 +
       config_.k4 * r2 * r2 * r2 * r2);
  u *= distortion_factor;
  v *= distortion_factor;
  if (u >= config_.width || u < 0) {
    return false;
  }
  if (v >= config_.height || v < 0) {
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
