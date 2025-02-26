#include "hydra/navigation/dijkstra.h"

namespace hydra {

void dijkstra(const std::map<NodeId, SceneGraphNode::Ptr>& nodes,
              const std::set<EdgeKey>& edges,
              const NodeId& start,
              const NodeId& goal,
              std::vector<NodeId>& path,
              std::vector<Eigen::Vector3d>& path_points) {
  std::unordered_map<NodeId, double> dist;
  std::unordered_map<NodeId, NodeId> prev;
  MinHeap pq;

  for (const auto& [id, _] : nodes) {
    dist[id] = std::numeric_limits<double>::infinity();
  }
  dist[start] = 0.0;
  pq.emplace(0.0, start);

  while (!pq.empty()) {
    auto [currentDist, current] = pq.top();
    pq.pop();

    if (current == goal) break;

    for (const auto& edge : edges) {
      NodeId neighbor;
      if (edge.k1 == current) {
        neighbor = edge.k2;
      } else if (edge.k2 == current) {
        neighbor = edge.k1;
      } else {
        continue;
      }
      double weight = (nodes.at(current)->attributes<NodeAttributes>().position -
                       nodes.at(neighbor)->attributes<NodeAttributes>().position)
                          .norm();
      double newDist = currentDist + weight;
      if (newDist < dist[neighbor]) {
        dist[neighbor] = newDist;
        prev[neighbor] = current;
        pq.emplace(newDist, neighbor);
      }
    }
  }

  path.clear();
  if (dist[goal] == std::numeric_limits<double>::infinity()) {
    return;
  }

  for (NodeId at = goal; at != start; at = prev[at]) {
    path.push_back(at);
    path_points.push_back(nodes.at(at)->attributes<NodeAttributes>().position);
  }
  path.push_back(start);
  path_points.push_back(nodes.at(start)->attributes<NodeAttributes>().position);
  std::reverse(path.begin(), path.end());
  std::reverse(path_points.begin(), path_points.end());
}

}  // namespace hydra
