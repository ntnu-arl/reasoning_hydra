#pragma once

#include <config_utilities/config_utilities.h>
#include <config_utilities/factory.h>

#include <unordered_map>
#include <vector>

#include "hydra/input/input_data.h"
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

  bool pointIsInViewFrustum(const Eigen::Vector3f& point_C,
                            float inflation_distance = 0.0f) const override;

 private:
  const Config config_;

  // Pre-computed stored values.
  Eigen::Matrix<float, 4, 3> view_frustum_;  // Top, right, bottom, left plane normals.

  inline static const auto registration_ =
      config::RegistrationWithConfig<Sensor,
                                     CameraLidarFusion,
                                     CameraLidarFusion::Config>("camera_lidar_fusion");
};

void declare_config(CameraLidarFusion::Config& config);

}  // namespace hydra
