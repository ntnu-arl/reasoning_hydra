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
    : Sensor(config),
      config_(config::checkValid(config)),
      width_(config_.horizontal_fov / config_.horizontal_resolution),
      height_(config_.vertical_fov / config_.vertical_resolution),
      vertical_fov_rad_(config_.vertical_fov * M_PI / 180.0f),
      vertical_fov_top_rad_(config_.vertical_fov_top * M_PI / 180.0f),
      horizontal_fov_rad_(config_.horizontal_fov * M_PI / 180.0f) {
  cam_extrinsics_ = std::make_unique<PoseStatus>(config_.cam2body_rotation,
                                                 config_.cam2body_translation);
  // Pre-compute the view frustum (top, right, bottom, left, plane normals).
  // compute upper phi limit and associated z at unit focal distance
  const auto phi_up =
      config_.is_asymmetric ? vertical_fov_top_rad_ : vertical_fov_rad_ / 2.0f;
  const auto z_up = std::tan(phi_up);
  // left upper x right upper corner
  top_frustum_normal_ = Eigen::Vector3f(1.0f, 1.0f, z_up)
                            .cross(Eigen::Vector3f(1.0, -1.0f, z_up))
                            .normalized();

  // compute lower phi limit and associated z at unit focal distance
  const auto phi_down = config_.is_asymmetric
                            ? vertical_fov_top_rad_ - vertical_fov_rad_
                            : -vertical_fov_rad_ / 2.0f;
  const auto z_down = std::tan(phi_down);
  // right lower x left lower corner
  bottom_frustum_normal_ = Eigen::Vector3f(1.0f, -1.0f, z_down)
                               .cross(Eigen::Vector3f(1.0, 1.0f, z_down))
                               .normalized();

  const auto half_fov = horizontal_fov_rad_ / 2.0f;
  // compute left theta extent and flip if greater than 90 degrees
  const auto theta_left = half_fov >= M_PI / 2.0f ? M_PI - half_fov : half_fov;
  // flip associated unit focal length if required and compute actual coordinates
  const auto x = half_fov >= M_PI / 2.0f ? -1.0f : 1.0f;
  const auto y_left = std::tan(theta_left);
  // left lower x left upper
  left_frustum_normal_ = Eigen::Vector3f(x, y_left, -1.0f)
                             .cross(Eigen::Vector3f(x, y_left, 1.0f))
                             .normalized();
  // right upper x right lower
  right_frustum_normal_ = Eigen::Vector3f(x, -y_left, 1.0f)
                              .cross(Eigen::Vector3f(x, -y_left, -1.0f))
                              .normalized();
}

float CameraLidarFusion::computeRayDensity(float voxel_size, float depth) const {
  // we want rays per meter... we can do this by computing a virtual focal length
  // compute focal lengths based on percent of spherical image inside 90 degree FOV
  // focal_length = (dim / 2) / tan(fov / 2) and tan(fov / 2) = 1
  const auto virtual_fx = (width_ * 90.0 / config_.horizontal_fov) / 2.0;
  const auto virtual_fy = (height_ * 90.0 / config_.vertical_fov) / 2.0;
  const auto voxel_density = voxel_size / depth;
  return virtual_fx * virtual_fy * voxel_density * voxel_density;
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
    const auto world_T_sensor = input.getSensorPose().cast<float>();
    auto point_iter = input.vertex_map.begin<cv::Vec3f>();
    while (point_iter != input.vertex_map.end<cv::Vec3f>()) {
      auto& p = *point_iter;
      Eigen::Vector3f p_S(p[0], p[1], p[2]);
      const auto p_W = world_T_sensor * p_S;
      p[0] = p_W.x();
      p[1] = p_W.y();
      p[2] = p_W.z();
    }

    input.points_in_world_frame = true;
  }
  const auto world_T_lidar = input.getSensorPose().cast<float>();
  const auto lidar_T_world = world_T_lidar.inverse();
  const auto cam_T_world =
      (input.world_T_body * (*cam_extrinsics_)).cast<float>().inverse();
  input.min_range = std::numeric_limits<float>::max();
  input.max_range = std::numeric_limits<float>::lowest();

  bool has_panoptic = input.features_mask.has_value();
  cv::Size size(width_, height_);
  input.range_image = cv::Mat::zeros(size, CV_32FC1);
  cv::Mat labels = -cv::Mat::ones(size, CV_32SC1);
  cv::Mat colors = cv::Mat::zeros(size, CV_8UC3);
  input.pointcloud.reset(new pcl::PointCloud<pcl::PointXYZRGBL>());

  if (has_panoptic && (*input.features_mask).type() != CV_16UC1) {
    LOG(ERROR) << "features_mask must be CV_16UC1!";
    return false;
  }
  cv::Mat panoptic;
  if (has_panoptic) {
    panoptic = cv::Mat::zeros(size, CV_16UC1);
  }

  size_t num_invalid = 0;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr debug_pointcloud(
      new pcl::PointCloud<pcl::PointXYZRGB>());
  for (int row = 0; row < input.vertex_map.rows; ++row) {
    for (int col = 0; col < input.vertex_map.cols; ++col) {
      int u, v;
      const auto& p = input.vertex_map.at<cv::Vec3f>(row, col);
      Eigen::Vector3f p_C(p[0], p[1], p[2]);
      Eigen::Vector3f p_L(p[0], p[1], p[2]);
      if (input.points_in_world_frame) {
        Eigen::Vector4f p_C_h = cam_T_world * Eigen::Vector4f(p[0], p[1], p[2], 1.0f);
        p_C(0) = p_C_h.x();
        p_C(1) = p_C_h.y();
        p_C(2) = p_C_h.z();
        p_L = lidar_T_world * p_L;
      } else {
        if (!input.sensor1_T_sensor2) {
          LOG(ERROR) << "sensor1_T_sensor2 required to convert points!";
          return false;
        }
        p_C = (*input.sensor1_T_sensor2).cast<float>() * p_C;
      }
      int u_img, v_img;
      if (!projectPointToCameraPlane(p_C, u_img, v_img)) {
        ++num_invalid;
        continue;
      }
      // pcl::PointXYZRGB debug_point;
      // debug_point.x = p[0];
      // debug_point.y = p[1];
      // debug_point.z = p[2];
      // debug_point.r = input.color_image.at<cv::Vec3b>(row, col)[2];
      // debug_point.g = input.color_image.at<cv::Vec3b>(row, col)[1];
      // debug_point.b = input.color_image.at<cv::Vec3b>(row, col)[0];
      // debug_pointcloud->points.push_back(debug_point);
      // Pointcloud
      pcl::PointXYZRGBL rgbl_point;
      if (input.points_in_world_frame) {
        rgbl_point.x = p[0];
        rgbl_point.y = p[1];
        rgbl_point.z = p[2];
      } else {
        Eigen::Vector4f p_W = world_T_lidar * Eigen::Vector4f(p[0], p[1], p[2], 1.0f);
        rgbl_point.x = p_W.x();
        rgbl_point.y = p_W.y();
        rgbl_point.z = p_W.z();
      }
      rgbl_point.r = input.color_image.at<cv::Vec3b>(row, col)[2];
      rgbl_point.g = input.color_image.at<cv::Vec3b>(row, col)[1];
      rgbl_point.b = input.color_image.at<cv::Vec3b>(row, col)[0];
      rgbl_point.label =
          static_cast<uint32_t>(input.label_image.at<int32_t>(row, col) + 1);
      input.pointcloud->points.push_back(rgbl_point);

      if (!input.valid[row][col]) {
        ++num_invalid;
        continue;
      }
      if (!projectPointToImagePlane(p_L, u, v)) {
        ++num_invalid;
        continue;
      }
      const auto range_m = p_L.norm();
      input.min_range = std::min(input.min_range, range_m);
      input.max_range = std::max(input.max_range, range_m);
      input.range_image.at<float>(v, u) = range_m;
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
  // pcl::io::savePLYFile("/home/albert/Desktop/pts/" +
  // std::to_string(input.timestamp_ns) +
  //                    "_debug.ply", *debug_pointcloud);

  // cv::Mat scaled_img;
  // float scale = 255.0f / (config.max_range - config_.min_range);
  // input.range_image.convertTo(scaled_img, CV_8U, scale, -config.min_range * scale);
  // cv::Mat shifted, output_img;
  // shifted = input.label_image + 1;
  // shifted.convertTo(output_img, CV_16U);
  // cv::imwrite("/home/albert/Desktop/pts/" + std::to_string(input.timestamp_ns) +
  // "_range_image.png", scaled_img); cv::imwrite("/home/albert/Desktop/pts/" +
  // std::to_string(input.timestamp_ns) + "_colors.png", input.color_image);
  // cv::imwrite("/home/albert/Desktop/pts/" + std::to_string(input.timestamp_ns) +
  // "_labels.png", output_img);

  return true;
}

bool CameraLidarFusion::projectPointToImagePlane(const Eigen::Vector3f& p_C,
                                                 float& u,
                                                 float& v) const {
  if (p_C.norm() <= config_.min_range) {
    return false;
  }

  // map lidar point to [0, w] x [0, h]
  // assumes forward-left-up and fov center aligned with x-axis for a spherical model
  const auto bearing = p_C.normalized();
  const auto phi = std::asin(bearing.z());
  const auto theta = std::atan2(bearing.y(), bearing.x());

  if (config_.is_asymmetric) {
    // phi is [-pi/2, pi/2], ratio is [1, 0], maps to [height, 0]
    const auto vertical_ratio = (vertical_fov_top_rad_ - phi) / vertical_fov_rad_;
    v = height_ * vertical_ratio;
  } else {
    // phi is [-pi/2, pi/2], ratio is [1, 0], maps to [height, 0]
    const auto vertical_ratio = (vertical_fov_rad_ / 2.0f - phi) / vertical_fov_rad_;
    v = height_ * vertical_ratio;
  }

  if (v < 0.0f || v > height_) {
    return false;
  }

  const auto h_ratio = (horizontal_fov_rad_ / 2.0f - theta) / horizontal_fov_rad_;
  u = width_ * h_ratio;
  if (u < 0.0f || u > width_) {
    return false;
  }

  return true;
}

bool CameraLidarFusion::projectPointToImagePlane(const Eigen::Vector3f& p_C,
                                                 int& u,
                                                 int& v) const {
  float temp_u = -1.0f;
  float temp_v = -1.0f;
  if (!projectPointToImagePlane(p_C, temp_u, temp_v)) {
    return false;
  }

  // assumption is pixel indices point to top left corner
  u = std::floor(temp_u);
  v = std::floor(temp_v);
  if (u >= width_ || u < 0 || v >= height_ || v < 0) {
    return false;
  }

  return true;
}

bool CameraLidarFusion::projectPointToCameraPlane(const Eigen::Vector3f& p_C,
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

bool CameraLidarFusion::projectPointToCameraPlane(const Eigen::Vector3f& p_C,
                                                  int& u,
                                                  int& v) const {
  float u_float = -1.0f;
  float v_float = -1.0f;
  if (!projectPointToCameraPlane(p_C, u_float, v_float)) {
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
  if (point_C.norm() > config_.max_range + inflation_distance) {
    return false;
  }

  double radius_2d = std::sqrt(point_C.x() * point_C.x() + point_C.y() * point_C.y());
  Eigen::Vector3f point_C_2d(radius_2d, 0, point_C.z());
  if (point_C_2d.dot(top_frustum_normal_) < -inflation_distance) {
    return false;
  }

  if (point_C_2d.dot(bottom_frustum_normal_) < -inflation_distance) {
    return false;
  }

  const auto left_prod = point_C.dot(left_frustum_normal_);
  const auto right_prod = point_C.dot(right_frustum_normal_);
  if (horizontal_fov_rad_ <= M_PI) {
    // normal camera or half-plane case
    return left_prod >= -inflation_distance && right_prod >= -inflation_distance;
  }

  // check to make sure that we're not in the exluded region
  return !(left_prod <= -inflation_distance && right_prod <= -inflation_distance);
}
}  // namespace hydra
