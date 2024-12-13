#pragma once

#include <config_utilities/config.h>

#include <Eigen/Core>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace hydra {

struct KMeansConfig {
  size_t num_clusters = 10;
  size_t max_iterations = 500;
  double tolerance = 1e-4;
};

// Templated KMeans clustering class
template <typename Scalar>
class KMeans {
 public:
  using Ptr = std::unique_ptr<KMeans>;

  KMeansConfig config;

  explicit KMeans(const KMeansConfig& config);

  void cluster(const std::vector<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>& points,
               std::vector<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>& centroids);

 protected:
  // Helper functions
  void initialize_centroids(
      const std::vector<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>& points,
      std::vector<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>& centroids);
  Scalar compute_distance(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& point,
                          const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& centroid);
  size_t find_closest_centroid(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& point,
      const std::vector<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>& centroids);
};

void declare_config(KMeansConfig& config);

}  // namespace hydra
