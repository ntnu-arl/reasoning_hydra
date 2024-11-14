#pragma once

#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_types.h>

#include <fstream>
#include <iostream>
#include <vector>

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

}  // namespace hydra
