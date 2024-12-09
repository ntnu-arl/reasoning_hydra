#include "hydra/frontend/place_features_fusion.h"

namespace hydra {

float cosineSimilarity(const Eigen::VectorXf& v1, const Eigen::VectorXf& v2) {
  assert(v1.size() == v2.size());
  return v1.dot(v2) / (v1.norm() * v2.norm());
}

void PlaceFeaturesAverage::updateFeatures(
    spark_dsg::PlaceNodeAttributes& place_attribute,
    const Eigen::VectorXf& feature_vector) const {
  if (place_attribute.num_observations.empty()) {
    place_attribute.feature_vectors.push_back(feature_vector);
    place_attribute.num_observations.push_back(1);
  } else {
    auto& place_feature = place_attribute.feature_vectors[0];
    place_feature =
        (place_feature * place_attribute.num_observations[0] + feature_vector) /
        (place_attribute.num_observations[0] + 1);
    place_attribute.num_observations[0] += 1;
  }
}

PlaceFeaturesClusters::PlaceFeaturesClusters(const Config& config) : config(config) {}

void PlaceFeaturesClusters::updateFeatures(
    spark_dsg::PlaceNodeAttributes& place_attribute,
    const Eigen::VectorXf& feature_vector) const {
  size_t index;

  if (findClosestCluster(place_attribute, feature_vector, index)) {
    auto& place_feature = place_attribute.feature_vectors[index];
    auto& num_observations = place_attribute.num_observations[index];
    place_feature =
        (place_feature * num_observations + feature_vector) / (num_observations + 1);
    num_observations += 1;
  }

  if (place_attribute.feature_vectors.size() < config.max_clusters) {
    place_attribute.feature_vectors.push_back(feature_vector);
    place_attribute.num_observations.push_back(1);
  }
}

bool PlaceFeaturesClusters::findClosestCluster(
    const spark_dsg::PlaceNodeAttributes& place_attribute,
    const Eigen::VectorXf& feature_vector,
    size_t& index) const {
  if (place_attribute.feature_vectors.empty()) {
    return false;
  }
  float min_distance = std::numeric_limits<float>::max();
  bool found = false;
  for (size_t i = 0; i < place_attribute.feature_vectors.size(); ++i) {
    const auto distance =
        cosineSimilarity(place_attribute.feature_vectors[i], feature_vector);
    if (distance < min_distance) {
      min_distance = distance;
      index = i;
      found = true;
    }
  }
  return found;
}

void declare_config(PlaceFeaturesClusters::Config& conf) {
  using namespace config;
  name("PlaceFeaturesClusters::Config");
  field(conf.min_distance, "min_distance");
  field(conf.max_clusters, "max_clusters");
}
}  // namespace hydra
