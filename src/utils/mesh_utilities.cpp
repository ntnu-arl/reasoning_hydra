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
#include "hydra/utils/mesh_utilities.h"

#include <spark_dsg/bounding_box_extraction.h>

namespace hydra {

bool updateNodeCentroid(const spark_dsg::Mesh& mesh,
                        const std::vector<size_t>& indices,
                        NodeAttributes& attrs) {
  size_t num_valid = 0;
  Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
  for (const auto idx : indices) {
    const auto pos = mesh.pos(idx).cast<double>();
    if (!pos.array().isFinite().all()) {
      continue;
    }

    centroid += pos;
    ++num_valid;
  }

  if (!num_valid) {
    return false;
  }

  attrs.position = centroid / num_valid;
  return true;
}

bool updateObjectGeometry(const spark_dsg::Mesh& mesh,
                          ObjectNodeAttributes& attrs,
                          const std::vector<size_t>* indices,
                          std::optional<BoundingBox::Type> type) {
  std::vector<size_t> mesh_connections;
  if (!indices) {
    mesh_connections.assign(attrs.mesh_connections.begin(),
                            attrs.mesh_connections.end());
  }

  const BoundingBox::MeshAdaptor adaptor(mesh, indices ? indices : &mesh_connections);
  attrs.bounding_box = BoundingBox(adaptor, type.value_or(attrs.bounding_box.type));
  if (indices) {
    return updateNodeCentroid(mesh, *indices, attrs);
  } else {
    return updateNodeCentroid(mesh, mesh_connections, attrs);
  }
}

void mergeObjectSemanticFeature(const ObjectNodeAttributes& other_attrs,
                                ObjectNodeAttributes& attrs) {
  if (other_attrs.num_observations > 0 && attrs.num_observations > 0) {
    attrs.semantic_feature =
        (attrs.semantic_feature * attrs.num_observations +
         other_attrs.semantic_feature * other_attrs.num_observations) /
        (attrs.num_observations + other_attrs.num_observations);
    attrs.num_observations += other_attrs.num_observations;
  } else if (other_attrs.num_observations > 0) {
    attrs.semantic_feature = other_attrs.semantic_feature;
    attrs.num_observations = other_attrs.num_observations;
  }
}

void updateObjectSemanticFeature(const Eigen::VectorXf& semantic_feature,
                                 ObjectNodeAttributes& attrs) {
  if (attrs.num_observations > 0) {
    attrs.semantic_feature =
        (attrs.semantic_feature * attrs.num_observations + semantic_feature) /
        (attrs.num_observations + 1);
    ++attrs.num_observations;
  } else {
    attrs.semantic_feature = semantic_feature;
    attrs.num_observations = 1;
  }
}

MeshLayer::Ptr getActiveMesh(const MeshLayer& mesh_layer,
                             const BlockIndices& archived_blocks) {
  auto active_mesh = std::make_shared<MeshLayer>(mesh_layer.blockSize());
  const BlockIndexSet archived_set(archived_blocks.begin(), archived_blocks.end());
  for (const auto& block : mesh_layer.updatedBlockIndices()) {
    if (archived_set.count(block)) {
      continue;
    }
    auto& block_data = mesh_layer.getBlock(block);
    active_mesh->allocateBlock(block) = block_data;
  }
  return active_mesh;
}

void mergeEdges(DynamicSceneGraph& graph,
                const NodeId& old_node_id,
                const NodeId& new_node_id,
                std::unordered_map<NodeId, std::set<NodeId>>& active_edges) {
  auto it = active_edges[old_node_id].begin();
  while (it != active_edges[old_node_id].end()) {
    auto target_id = *it;                      // Copy the target ID
    it = active_edges[old_node_id].erase(it);  // Erase and get next valid iterator

    // Case when the old node is merged with the target node
    if (target_id == new_node_id) {
      graph.removeEdge(old_node_id, target_id);
      active_edges[target_id].erase(old_node_id);
      continue;
    }

    if (graph.hasEdge(old_node_id, target_id)) {
      auto edge = graph.getEdge(old_node_id, target_id).info->clone();
      edge->setNewId(old_node_id, new_node_id);
      if (graph.hasEdge(new_node_id, target_id)) {
        auto new_edge = graph.getEdge(new_node_id, target_id).info->clone();
        new_edge->merge(*edge);
        graph.setEdgeAttributes(new_node_id, target_id, std::move(new_edge));
      } else {
        graph.insertEdge(new_node_id, target_id, std::move(edge));
      }

      graph.removeEdge(old_node_id, target_id);
    }

    active_edges[target_id].erase(old_node_id);
    if (graph.hasEdge(target_id, new_node_id)) {
      active_edges[target_id].insert(new_node_id);
      active_edges[new_node_id].insert(target_id);
    }
  }
}
}  // namespace hydra
