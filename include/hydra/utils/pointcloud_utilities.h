#pragma once

#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "hydra/reconstruction/volumetric_map.h"

namespace hydra {

template <typename PointT>
void computeNormals(const typename pcl::PointCloud<PointT>::Ptr& cloud,
                    pcl::PointCloud<pcl::Normal>::Ptr& normals,
                    double radius = 0.03) {
  typename pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
  tree->setInputCloud(cloud);

  pcl::NormalEstimation<PointT, pcl::Normal> ne;
  ne.setInputCloud(cloud);
  ne.setSearchMethod(tree);
  ne.setRadiusSearch(radius);
  ne.compute(*normals);
}

template <typename PointT>
void savePointCloud(const typename pcl::PointCloud<PointT>::Ptr& cloud,
                    const std::string& path,
                    const std::string& format = ".pcd") {
  std::string object_cloud_path = path + format;
  if (format == ".pcd") {
    pcl::io::savePCDFileBinary(object_cloud_path, *cloud);
  } else if (format == ".ply") {
    pcl::io::savePLYFile(object_cloud_path, *cloud, true);
  }
}

inline void saveVectorToBinary(const std::vector<uint32_t>& vec,
                               const std::string& filename) {
  std::ofstream output(filename + ".bin", std::ios::binary);
  if (!output) {
    std::cerr << "Could not open the file for writing!\n";
    return;
  }
  uint64_t size = vec.size();
  output.write(reinterpret_cast<const char*>(&size), sizeof(size));
  output.write(reinterpret_cast<const char*>(vec.data()),
               vec.size() * sizeof(uint32_t));
  output.close();
}

inline bool toPcl(const BaseSemanticPointCloud::Ptr in_pcl,
                  pcl::PointCloud<pcl::PointXYZRGBL>::Ptr out_pcl) {
  if (!in_pcl || !out_pcl) {
    return false;
  }
  out_pcl->clear();
  for (const auto& block : *in_pcl) {
    if (!block.updated) {
      continue;
    }
    for (size_t x = 0; x < block.voxels_per_side; ++x) {
      for (size_t y = 0; y < block.voxels_per_side; ++y) {
        for (size_t z = 0; z < block.voxels_per_side; ++z) {
          const VoxelIndex voxel_index(x, y, z);
          const auto& voxel = block.getVoxel(voxel_index);
          if (voxel.empty) {
            continue;
          }
          Point point = block.getVoxelPosition(voxel_index);
          pcl::PointXYZRGBL point_pcl;
          point_pcl.x = point.x();
          point_pcl.y = point.y();
          point_pcl.z = point.z();
          point_pcl.label = voxel.semantic_label;
          point_pcl.r = voxel.color.r;
          point_pcl.g = voxel.color.g;
          point_pcl.b = voxel.color.b;
          out_pcl->push_back(point_pcl);
        }
      }
    }
  }
  return out_pcl->size() > 0;
}
}  // namespace hydra
