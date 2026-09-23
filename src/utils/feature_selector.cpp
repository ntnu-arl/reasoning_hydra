/* -----------------------------------------------------------------------------
 * Copyright 2022 Massachusetts Institute of Technology.
 * All Rights Reserved
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Research was sponsored by the United States Air Force Research Laboratory and
 * the United States Air Force Artificial Intelligence Accelerator and was
 * accomplished under Cooperative Agreement Number FA8750-19-2-1000. The views
 * and conclusions contained in this document are those of the authors and should
 * not be interpreted as representing the official policies, either expressed or
 * implied, of the United States Air Force or the U.S. Government. The U.S.
 * Government is authorized to reproduce and distribute reprints for Government
 * purposes notwithstanding any copyright notation herein.
 * -------------------------------------------------------------------------- */
#include "hydra/utils/feature_selector.h"

#include <config_utilities/config.h>
#include <config_utilities/factory.h>
#include <glog/logging.h>

namespace hydra {

static const auto greedy_reg =
    config::RegistrationWithConfig<FeatureSelector,
                                   GreedyCosineDiversitySelector,
                                   GreedyCosineDiversitySelector::Config>(
        "GreedyCosineDiversitySelector");

static const auto dpp_reg =
    config::RegistrationWithConfig<FeatureSelector, DPPSelector, DPPSelector::Config>(
        "DPPSelector");

namespace {
constexpr float REG = 1e-6f;  // Cholesky regularization
constexpr float EPS = 1e-8f;  // log safety clamp
}  // namespace

template <typename Scalar>
inline Scalar cosineSimilarity(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& a,
                               const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& b,
                               const Scalar eps = static_cast<Scalar>(1e-6)) {
  const auto res = a.dot(b) / (a.norm() * b.norm() + eps);
  return std::clamp(res, static_cast<Scalar>(-1.0), static_cast<Scalar>(1.0));
}

void declare_config(FeatureSelector::Config& config) {
  using namespace config;
  name("FeatureSelector::Config");
  field(config.max_num_features, "max_num_features");
}

const std::vector<size_t> GreedyCosineDiversitySelector::select(
    const FeatureList& features) const {
  const size_t M = features.size();
  if (config.max_num_features >= M) {
    std::vector<size_t> all(M);
    std::iota(all.begin(), all.end(), 0);
    return all;
  }

  // Precompute cosine similarity matrix
  Eigen::MatrixXf K(M, M);
  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < M; ++j) {
      K(i, j) = cosineSimilarity(features[i], features[j]);
    }
  }
  std::vector<size_t> selected;
  selected.reserve(config.max_num_features);

  // Start from an arbitrary random feature
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, M - 1);
  int firstIdx = dis(gen);
  selected.push_back(firstIdx);
  FeatureVector cumulative = K.col(firstIdx);

  while (selected.size() < config.max_num_features) {
    float bestScore = std::numeric_limits<float>::infinity();
    int bestIdx = -1;

    for (size_t i = 0; i < M; ++i) {
      if (std::find(selected.begin(), selected.end(), i) != selected.end()) {
        continue;
      }

      float score = cumulative(i);
      if (score < bestScore) {
        bestScore = score;
        bestIdx = static_cast<int>(i);
      }
    }
    if (bestIdx == -1) {
      break;
    }
    selected.push_back(static_cast<size_t>(bestIdx));
    cumulative += K.col(bestIdx);
  }
  return selected;
}

void declare_config(GreedyCosineDiversitySelector::Config& config) {
  using namespace config;
  base<FeatureSelector::Config>(config);
  name("GreedyCosineDiversitySelector::Config");
}

const std::vector<size_t> DPPSelector::select(const FeatureList& features) const {
  const size_t M = features.size();
  if (config.max_num_features >= M) {
    std::vector<size_t> all(M);
    std::iota(all.begin(), all.end(), 0);
    return all;
  }

  // Kernel matrix
  Eigen::MatrixXf K(M, M);
  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < M; ++j) {
      K(i, j) = features[i].dot(features[j]);
    }
  }

  std::vector<size_t> selected;
  selected.reserve(config.max_num_features);

  // Start with max self-similarity
  int first = 0;
  K.diagonal().maxCoeff(&first);
  selected.push_back(first);

  Eigen::MatrixXf L(1, 1);
  L(0, 0) = K(first, first);

  while (selected.size() < config.max_num_features) {
    float bestGain = -std::numeric_limits<float>::infinity();
    int bestIdx = -1;

    for (size_t i = 0; i < M; ++i) {
      if (std::find(selected.begin(), selected.end(), i) != selected.end()) continue;

      FeatureVector k(selected.size());
      for (size_t j = 0; j < selected.size(); ++j) {
        k(j) = K(i, selected[j]);
      }

      Eigen::MatrixXf L_reg = L;
      L_reg.diagonal().array() += REG;

      Eigen::LLT<Eigen::MatrixXf> llt(L_reg);
      if (llt.info() != Eigen::Success) {
        LOG(WARNING) << "DPPSelector: LLT decomposition failed during selection.";
        continue;
      }

      float quad = k.dot(llt.solve(k));
      float value = K(i, i) - quad;
      value = std::max(value, EPS);
      float gain = std::log(value);

      if (gain > bestGain) {
        bestGain = gain;
        bestIdx = static_cast<int>(i);
      }
    }
    if (bestIdx == -1) {
      break;
    }

    // Update L
    size_t s = selected.size();
    Eigen::MatrixXf L_new(s + 1, s + 1);
    L_new.topLeftCorner(s, s) = L;

    for (size_t j = 0; j < s; ++j) {
      L_new(s, j) = K(bestIdx, selected[j]);
      L_new(j, s) = L_new(s, j);
    }

    L_new(s, s) = K(bestIdx, bestIdx);
    L = L_new;

    selected.push_back(bestIdx);
  }

  return selected;
}

void declare_config(DPPSelector::Config& config) {
  using namespace config;
  base<FeatureSelector::Config>(config);
  name("DPPSelector::Config");
}

}  // namespace hydra
