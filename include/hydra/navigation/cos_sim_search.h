#pragma once

#include <config_utilities/config.h>
#include <config_utilities/factory.h>
#include <glog/logging.h>

#include <Eigen/Dense>
#include <memory>
#include <numeric>
#include <vector>

#include "hydra/navigation/search.h"

namespace hydra {

class CosSimSearch : public Search {
 public:
  struct Config {
    struct Entity {
      float prob_threshold = 0.5;
      bool normalize_similarities = false;
      bool use_softmax = false;
      bool use_normalize = false;
      bool use_mean = false;
    };
    Entity room = Entity();
    Entity object = Entity();
  } const config;

  explicit CosSimSearch(const Config& config);
  ~CosSimSearch();

  bool searchRoom(const Eigen::VectorXf& text_room_embedding,
                  const std::vector<std::vector<Eigen::VectorXf>>& room_embeddings,
                  size_t& result) const override;
  bool searchObject(const Eigen::VectorXf& text_object_embedding,
                    const std::vector<Eigen::VectorXf>& object_embeddings,
                    std::vector<size_t>& result) const override;

 protected:
  float cosSim(const Eigen::VectorXf& a,
               const Eigen::VectorXf& b,
               const float eps = 1e-6) const;
  void softmax(const std::vector<float>& sims,
               std::vector<float>& probs,
               bool normalize = true) const;

 private:
  inline static const auto registration =
      config::RegistrationWithConfig<Search, CosSimSearch, Config>("CosSimSearch");

  void normalize(const std::vector<float>& sims, std::vector<float>& probs) const;
};

void declare_config(CosSimSearch::Config& config);

}  // namespace hydra
