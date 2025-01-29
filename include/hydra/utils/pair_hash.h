#pragma once
#include <Eigen/Dense>
#include <algorithm>
#include <unordered_map>
#include <utility>

namespace hydra {
// Custom hash function for pairs
struct PairHash {
  size_t operator()(const std::pair<uint16_t, uint16_t>& p) const {
    // Use a hash combination technique for the pair
    uint16_t a = std::min(p.first, p.second);
    uint16_t b = std::max(p.first, p.second);
    return std::hash<uint16_t>()(a) ^ (std::hash<uint16_t>()(b) << 1);
  }
};

// Equality comparison for unordered_map
struct PairEqual {
  bool operator()(const std::pair<uint16_t, uint16_t>& lhs,
                  const std::pair<uint16_t, uint16_t>& rhs) const {
    return (std::min(lhs.first, lhs.second) == std::min(rhs.first, rhs.second)) &&
           (std::max(lhs.first, lhs.second) == std::max(rhs.first, rhs.second));
  }
};

using PairHashMap = std::
    unordered_map<std::pair<uint16_t, uint16_t>, Eigen::MatrixXf, PairHash, PairEqual>;

}  // namespace hydra
