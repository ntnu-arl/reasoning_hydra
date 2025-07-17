#include "hydra/navigation/dijkstra.h"

namespace hydra {

void dijkstra(const std::map<NodeId, SceneGraphNode::Ptr>& nodes,
              const std::set<EdgeKey>& edges,
              const NodeId& start,
              const NodeId& goal,
              std::vector<NodeId>& path,
              std::vector<Eigen::Vector3d>& path_points) {
  path.clear();
  path_points.clear();

  // Ensure start and goal exist
  if (nodes.find(start) == nodes.end() || nodes.find(goal) == nodes.end()) {
    return;
  }

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

      // Only consider neighbors that exist in nodes
      if (nodes.find(neighbor) == nodes.end()) continue;

      const auto& currentNode = nodes.at(current);
      const auto& neighborNode = nodes.at(neighbor);
      double weight = (currentNode->attributes<NodeAttributes>().position -
                       neighborNode->attributes<NodeAttributes>().position)
                          .norm();

      double newDist = currentDist + weight;
      if (newDist < dist[neighbor]) {
        dist[neighbor] = newDist;
        prev[neighbor] = current;
        pq.emplace(newDist, neighbor);
      }
    }
  }

  // If goal is unreachable
  if (dist[goal] == std::numeric_limits<double>::infinity()) {
    return;
  }

  // Reconstruct path
  NodeId at = goal;
  while (at != start) {
    auto it = prev.find(at);
    if (it == prev.end()) {
      path.clear();
      path_points.clear();
      return;
    }
    path.push_back(at);
    path_points.push_back(nodes.at(at)->attributes<NodeAttributes>().position);
    at = it->second;
  }
  path.push_back(start);
  path_points.push_back(nodes.at(start)->attributes<NodeAttributes>().position);
  std::reverse(path.begin(), path.end());
  std::reverse(path_points.begin(), path_points.end());
}

}  // namespace hydra
