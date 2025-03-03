#pragma once

#include <Eigen/Dense>
#include <memory>
#include <vector>

namespace hydra {
class Search {
 public:
  using Ptr = std::unique_ptr<Search>;
  Search() {}
  virtual ~Search() = default;

  virtual bool searchRoom(
      const Eigen::VectorXf& text_room_embedding,
      const std::vector<std::vector<Eigen::VectorXf>>& room_embeddings,
      size_t& result) const = 0;
  virtual bool searchObject(const Eigen::VectorXf& text_object_embedding,
                            const std::vector<Eigen::VectorXf>& object_embeddings,
                            std::vector<size_t>& result) const = 0;
};

}  // namespace hydra
