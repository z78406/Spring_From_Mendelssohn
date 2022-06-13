// Copyright 2022 Zzy
// Author: Zzy-buffer
// Data IO class for SFM
/* Version 0.1 @ 2022
*/


#ifndef _data_io
#define _data_io

#include <iostream>
#include <vector>
#include <fstream>
#include <string>

#include "boost/filesystem.hpp"
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <eigen3/Eigen/Eigen>

namespace sfm
{
template <class T = std::string,class M = Eigen::Matrix3f,class P = pcl::PointCloud<pcl::PointXYZRGB>::Ptr&> // T,M,P for string/eigen mat/pc data type
class IO {
public:
  // read image list from dirs
  void ImportImageFiles(const T& dir, const T& extension, std::vector<T>& paths, std::vector<T>& filenames);
  // read a single image
  void ImportImage();
  //read calibration mat
  void ImportCalib(const T& file_name, M& k_mat);
  // write out a point cloud as ply
  void WritePlyFile(const T& file_name, P& point_cloud);

};




template <class T,class M,class P>
void IO<T, M, P>::ImportImageFiles(const T& dir, const T& extension, std::vector<T>& paths, std::vector<T>& filenames) {
  boost::filesystem::path p(dir);
  for (auto i = boost::filesystem::directory_iterator(p); i != boost::filesystem::directory_iterator(); i++) {
    if (boost::filesystem::extension(i->path().filename().string()) == extension) {
      filenames.push_back(i->path().filename().string());
      paths.push_back(i->path().string());
    }
  }
  std::sort(paths.begin(), paths.end());
  std::sort(filenames.begin(), filenames.end());
}

template <class T,class M,class P>
void IO<T, M, P>::ImportCalib(const T& file_name, M& k_mat) {
  std::ifstream in_file(file_name, std::ios::in);
  if (!in_file) {
    throw std::runtime_error("Input Error: camera calibration file not found.");
  }
  int i = 0;
  while (!in_file.eof() && i < 3) {
    in_file >> k_mat(i, 0) >> k_mat(i, 1) >> k_mat(i, 2);
    i++;
  }
  in_file.close();
  std::cout<<"Finish reading camera calibration file."<<std::endl;
}

}




#endif