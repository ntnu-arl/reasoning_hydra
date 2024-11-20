#pragma once
#ifndef INCLUDE_HYDRA_REASONING_REASONING_OUTPUT_H_
#define INCLUDE_HYDRA_REASONING_REASONING_OUTPUT_H_
#endif  // INCLUDE_HYDRA_REASONING_REASONING_OUTPUT_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace hydra {

using Nodes = std::vector<std::string>;
using Edges = std::unordered_map<int, std::unordered_map<int, std::string>>;
using NodeProbs = std::vector<std::vector<double>>;
using EdgeProbs = std::vector<std::vector<double>>;
using FeatureVectors = std::vector<std::vector<double>>;

struct ReasoningOutput {
  using ConstPtr = std::shared_ptr<const ReasoningOutput>;
  using Ptr = std::shared_ptr<ReasoningOutput>;

  Nodes nodes;
  Edges edges;
  NodeProbs node_probs;
  EdgeProbs edge_probs;
  FeatureVectors feature_vectors;
};
}  // namespace hydra
