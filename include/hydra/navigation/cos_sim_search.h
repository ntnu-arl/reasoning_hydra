#pragma once

#include <config_utilities/config.h>
#include <config_utilities/factory.h>
#include <glog/logging.h>

#include <Eigen/Dense>
#include <memory>
#include <numeric>
#include <vector>

namespace hydra {

class CosSimSearch {
 public:
  using Ptr = std::unique_ptr<CosSimSearch>;
  struct Config {
    struct Entity {
      float prob_threshold = 0.5;
      bool normalize_similarities = true;
      bool use_softmax = true;
      bool use_mean = false;
    };
    Entity room;
    Entity object;
  } const config;

  explicit CosSimSearch(const Config& config);
  ~CosSimSearch();

  bool searchRoom(const Eigen::VectorXf& text_room_embedding,
                  const std::vector<std::vector<Eigen::VectorXf>>& room_embeddings,
                  size_t& result) const;
  bool searchObject(const Eigen::VectorXf& text_object_embedding,
                    const std::vector<Eigen::VectorXf>& object_embeddings,
                    std::vector<size_t>& result) const;

 protected:
  float cosSim(const Eigen::VectorXf& a,
               const Eigen::VectorXf& b,
               const float eps = 1e-6) const;
  void softmax(const std::vector<float>& sims,
               std::vector<float>& probs,
               bool normalize = true) const;

  void normalize(std::vector<float>& sims) const;
};

}  // namespace hydra
