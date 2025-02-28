#include "hydra/navigation/cos_sim_search.h"

namespace hydra {

CosSimSearch::CosSimSearch(const Config& config) : config(config) {}

CosSimSearch::~CosSimSearch() {}

bool CosSimSearch::searchRoom(
    const Eigen::VectorXf& text_room_embedding,
    const std::vector<std::vector<Eigen::VectorXf>>& room_embeddings,
    size_t& result) const {
  if (room_embeddings.empty()) {
    return false;
  }

  std::vector<float> sims(room_embeddings.size(), 0.0);
  for (size_t i = 0; i < room_embeddings.size(); ++i) {
    std::vector<float> room_sims(room_embeddings[i].size(), 0.0);
    for (size_t j = 0; j < room_embeddings[i].size(); ++j) {
      room_sims[j] += cosSim(text_room_embedding, room_embeddings[i][j]);
    }
    if (config.room.use_mean) {
      sims[i] =
          std::accumulate(room_sims.begin(), room_sims.end(), 0.0) / room_sims.size();
    } else {
      sims[i] = *std::max_element(room_sims.begin(), room_sims.end());
    }
  }
  std::vector<float> probs;
  if (config.room.use_softmax) {
    softmax(sims, probs, config.room.normalize_similarities);
  } else {
    normalize(sims);
  }
  result = std::distance(probs.begin(), std::max_element(probs.begin(), probs.end()));
  return result > config.room.prob_threshold;
}

bool CosSimSearch::searchObject(const Eigen::VectorXf& text_object_embedding,
                                const std::vector<Eigen::VectorXf>& object_embeddings,
                                std::vector<size_t>& results) const {
  if (object_embeddings.empty()) {
    return false;
  }

  std::vector<float> sims(object_embeddings.size(), 0.0);
  for (size_t i = 0; i < object_embeddings.size(); ++i) {
    sims[i] = cosSim(text_object_embedding, object_embeddings[i]);
  }
  std::vector<float> probs;
  if (config.object.use_softmax) {
    softmax(sims, probs, config.object.normalize_similarities);
  } else {
    normalize(sims);
  }
  for (size_t i = 0; i < probs.size(); ++i) {
    if (probs[i] > config.object.prob_threshold) {
      results.push_back(i);
    }
  }
  return !results.empty();
}

float CosSimSearch::cosSim(const Eigen::VectorXf& a,
                           const Eigen::VectorXf& b,
                           const float eps) const {
  const auto res = a.dot(b) / (a.norm() * b.norm() + eps);
  return std::clamp(res, static_cast<float>(-1.0), static_cast<float>(1.0));
}

void CosSimSearch::softmax(const std::vector<float>& sims,
                           std::vector<float>& probs,
                           bool normalize) const {
  probs.resize(sims.size());
  float normalizer = 0.0;
  float min_value = -1.0;
  float max_value = 1.0;
  if (normalize) {
    min_value = *std::min_element(sims.begin(), sims.end());
    max_value = *std::max_element(sims.begin(), sims.end());
  }
  for (size_t i = 0; i < sims.size(); ++i) {
    if (normalize) {
      probs[i] = std::exp((sims[i] - min_value) / (max_value - min_value));
    } else {
      probs[i] = std::exp(sims[i]);
    }
    normalizer += probs[i];
  }
  for (size_t i = 0; i < probs.size(); ++i) {
    probs[i] /= normalizer;
  }
}

void CosSimSearch::normalize(std::vector<float>& sims) const {
  float min_value = *std::min_element(sims.begin(), sims.end());
  float max_value = *std::max_element(sims.begin(), sims.end());
  for (size_t i = 0; i < sims.size(); ++i) {
    sims[i] = (sims[i] - min_value) / (max_value - min_value);
  }
}

}  // namespace hydra
