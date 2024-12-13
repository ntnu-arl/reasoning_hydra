#include "hydra/utils/kmeans_clustering.h"

namespace hydra {

template <typename Scalar>
KMeans<Scalar>::KMeans(const KMeansConfig& config) : config(config) {}

template <typename Scalar>
void KMeans<Scalar>::initialize_centroids(
    const std::vector<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>& points,
    std::vector<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>& centroids) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, points.size() - 1);

  centroids.clear();
  for (size_t i = 0; i < config.num_clusters; ++i) {
    centroids.push_back(points[dis(gen)]);
  }
}

template <typename Scalar>
Scalar KMeans<Scalar>::compute_distance(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& point,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& centroid) {
  return -point.dot(centroid) / (point.norm() * centroid.norm());
}

template <typename Scalar>
size_t KMeans<Scalar>::find_closest_centroid(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& point,
    const std::vector<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>& centroids) {
  size_t closest = 0;
  Scalar min_distance = std::numeric_limits<Scalar>::max();

  for (size_t i = 0; i < centroids.size(); ++i) {
    Scalar distance = compute_distance(point, centroids[i]);
    if (distance < min_distance) {
      min_distance = distance;
      closest = i;
    }
  }

  return closest;
}

template <typename Scalar>
void KMeans<Scalar>::cluster(
    const std::vector<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>& points,
    std::vector<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>& centroids) {
  initialize_centroids(points, centroids);
  std::vector<size_t> assignments(points.size());

  for (size_t iter = 0; iter < config.max_iterations; ++iter) {
    // Assign each point to the closest centroid
    for (size_t i = 0; i < points.size(); ++i) {
      assignments[i] = find_closest_centroid(points[i], centroids);
    }

    // Update centroids
    std::vector<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> temp_centroids(
        config.num_clusters,
        Eigen::Matrix<Scalar, 1, Eigen::Dynamic>::Zero(points[0].rows()));
    std::vector<size_t> counts(config.num_clusters, 0);

    for (size_t i = 0; i < points.size(); ++i) {
      temp_centroids[assignments[i]] += points[i];
      counts[assignments[i]]++;
    }

    for (size_t i = 0; i < centroids.size(); ++i) {
      if (counts[i] > 0) {
        temp_centroids[i] /= static_cast<Scalar>(counts[i]);
      } else {
        temp_centroids[i] =
            centroids[i];  // Retain old centroid if no points are assigned
      }
    }

    // Check for convergence
    Scalar max_change = 0;
    for (size_t i = 0; i < centroids.size(); ++i) {
      max_change = std::max(max_change,
                            (temp_centroids[i] - centroids[i]).cwiseAbs().maxCoeff());
    }

    if (static_cast<double>(max_change) < config.tolerance) {
      break;
    }

    centroids = temp_centroids;
  }
}

// Explicit instantiation
template class KMeans<float>;
template class KMeans<double>;
template class KMeans<int>;

void declare_config(KMeansConfig& config) {
  using namespace config;
  name("KMeansConfig");
  field(config.num_clusters, "num_clusters");
  field(config.max_iterations, "max_iterations");
  field(config.tolerance, "tolerance");
}
}  // namespace hydra
