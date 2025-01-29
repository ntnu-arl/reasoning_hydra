#pragma once
#ifndef INCLUDE_HYDRA_REASONING_REASONING_OUTPUT_H_
#define INCLUDE_HYDRA_REASONING_REASONING_OUTPUT_H_
#endif  // INCLUDE_HYDRA_REASONING_REASONING_OUTPUT_H_

#include <spark_dsg/scene_graph_types.h>

#include <Eigen/Core>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace spark_dsg;

namespace hydra {

using Nodes = std::vector<std::string>;
using Edges = std::unordered_map<int, std::unordered_map<int, std::string>>;
using NodeProbs = std::vector<std::vector<double>>;
using EdgeProbs = std::vector<std::vector<double>>;
using Features = std::vector<Eigen::MatrixXf>;
using EdgeIndices = std::vector<std::pair<size_t, size_t>>;
using EdgeIds = std::vector<std::pair<NodeId, NodeId>>;
using Labels = std::vector<std::string>;

struct ReasoningOutput {
  using ConstPtr = std::shared_ptr<const ReasoningOutput>;
  using Ptr = std::shared_ptr<ReasoningOutput>;

  Nodes nodes;
  Edges edges;
  NodeProbs node_probs;
  EdgeProbs edge_probs;
  Features features;
  EdgeIndices edge_indices;
};

struct BLIPLabels {
  using ConstPtr = std::shared_ptr<const BLIPLabels>;
  using Ptr = std::shared_ptr<BLIPLabels>;

  EdgeIds edge_ids;
  Labels labels;
};
}  // namespace hydra
