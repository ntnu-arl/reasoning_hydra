#pragma once

#include <config_utilities/config_utilities.h>
#include <config_utilities/virtual_config.h>
#include <spark_dsg/node_attributes.h>

namespace hydra {

class PlaceFeaturesFusion {
 public:
  virtual void updateFeatures(spark_dsg::PlaceNodeAttributes& place_attribute,
                              const Eigen::VectorXf& feature_vector) const = 0;
};

class PlaceFeaturesAverage : public PlaceFeaturesFusion {
 public:
  PlaceFeaturesAverage() = default;
  void updateFeatures(spark_dsg::PlaceNodeAttributes& place_attribute,
                      const Eigen::VectorXf& feature_vector) const override;

 private:
  inline static const auto registration_ =
      config::Registration<PlaceFeaturesFusion, PlaceFeaturesAverage>(
          "place_features_average");
};

class PlaceFeaturesClusters : public PlaceFeaturesFusion {
 public:
  struct Config {
    float min_distance = 0.5;
    size_t max_clusters = 10;
  } const config;

  explicit PlaceFeaturesClusters(const Config& config);
  void updateFeatures(spark_dsg::PlaceNodeAttributes& place_attribute,
                      const Eigen::VectorXf& feature_vector) const override;

 private:
  bool findClosestCluster(const spark_dsg::PlaceNodeAttributes& place_attribute,
                          const Eigen::VectorXf& feature_vector,
                          size_t& index) const;
  inline static const auto registration_ =
      config::RegistrationWithConfig<PlaceFeaturesFusion,
                                     PlaceFeaturesClusters,
                                     Config>("place_features_clusters");
};

void declare_config(PlaceFeaturesClusters::Config& conf);

}  // namespace hydra
