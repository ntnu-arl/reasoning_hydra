#pragma once

#include <config_utilities/config_utilities.h>
#include <config_utilities/factory.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <unordered_map>
#include <vector>

#include "hydra/input/input_data.h"
#include "hydra/input/input_module.h"
#include "hydra/input/lidar.h"
#include "hydra/input/sensor.h"
#include "hydra/input/sensor_utilities.h"

namespace hydra {

/**
 * @brief Utility class bundling camera related operations and data.
 */
class CameraLidarFusion : public Sensor {
 public:
  // Note: negative parameters are REQUIRED
  struct Config : public Sensor::Config {
    /// Camera resolution (columns)
    int width = -1;
    /// Camera resolution (rows)
    int height = -1;
    /// Camera center point (x-axis)
    float cx = -1.0;
    /// Camera center point (y-axis)
    float cy = -1.0;
    /// Camera focal length (x-axis)
    float fx = -1.0;
    /// Camera focal length (y-axis)
    float fy = -1.0;
    // Distortion parameters
    float k1 = 0.0;
    float k2 = 0.0;
    float k3 = 0.0;
    float k4 = 0.0;
    /// Undistort flag
    bool undistort = false;
    /// Quaternion rotation
    Eigen::Quaterniond cam2body_rotation = Eigen::Quaterniond::Identity();
    /// Translation vector
    Eigen::Vector3d cam2body_translation = Eigen::Vector3d::Zero();
  };

  explicit CameraLidarFusion(const Config& config);

  virtual ~CameraLidarFusion() = default;

  const Config& getConfig() const { return config_; }

  float computeRayDensity(float voxel_size, float depth) const override;

  bool finalizeRepresentations(InputData& input,
                               bool force_world_frame = false) const override;

  bool projectPointToImagePlane(const Eigen::Vector3f& p_C,
                                float& u,
                                float& v) const override;

  bool projectPointToImagePlane(const Eigen::Vector3f& p_C,
                                int& u,
                                int& v) const override;

  bool projectPointToCameraPlane(const Eigen::Vector3f& p_C, int& u, int& v) const;

  bool projectPointToCameraPlane(const Eigen::Vector3f& p_C, float& u, float& v) const;

  bool pointIsInViewFrustum(const Eigen::Vector3f& point_C,
                            float inflation_distance = 0.0f) const override;

 private:
  const Config config_;
  const int width_;
  const int height_;
  const float vertical_fov_rad_;
  const float vertical_fov_top_rad_;
  const float horizontal_fov_rad_;

  // Pre-computed stored values.
  Eigen::Vector3f top_frustum_normal_;
  Eigen::Vector3f bottom_frustum_normal_;
  Eigen::Vector3f left_frustum_normal_;
  Eigen::Vector3f right_frustum_normal_;

  std::unique_ptr<PoseStatus> cam_extrinsics_;

  inline static const auto registration_ =
      config::RegistrationWithConfig<Sensor,
                                     CameraLidarFusion,
                                     CameraLidarFusion::Config>("camera_lidar_fusion");
};

void declare_config(CameraLidarFusion::Config& config);

}  // namespace hydra
