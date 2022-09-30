// Copyright 2022 Zzy
// Author: Zzy-buffer
// Frame class for SFM
/* Version 0.1 @ 2022
*/

#ifndef _frame
#define _frame

#include <iostream>
#include <vector>
#include <fstream>
#include <string>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
//Eigen
#include <eigen3/Eigen/Eigen>
// PCL
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>

namespace sfm {

// single frame class
struct Frame {
  public:
    Frame() {};
    Frame(const cv::Mat3b& img_rgb, const int& img_id, const std::string& img_path)
    {this->img_rgb = img_rgb, frame_id = img_id, this->img_path = img_path;}

    int frame_id;
    std::string img_path;
    cv::Mat3b img_rgb;
    Eigen::Matrix4f k_ext;  // extrinsic mat
    Eigen::Matrix3f k_int;  // intrinsic mat
    Eigen::Matrix4f P;      // projection mat

    std::vector<cv::KeyPoint> keypoints; // extracted keypoints of a frame
    cv::Mat descriptors; // extracted descriptors of keypoints of a frame

    std::vector<int> unique_pixel_id; // save idx of keypoint j that matches keypoint i
    std::vector<bool> unique_pixel_has_match; // save if keypoint i finds a matching point
    void init_p_id();
};


// frame pair class
struct Pair {
  public:
    unsigned int frame_id_1;
    unsigned int frame_id_2;

    std::vector<cv::DMatch> initial_matches;
    std::vector<cv::DMatch> best_matches;

    Eigen::Matrix4f Transform;  // transform mat from 1 to 2

    double appro_depth;

    bool operator==(const Pair* pair) const { // for constructing customized map
    return std::to_string(this->frame_id_1) + '_' + std::to_string(this->frame_id_2) ==
           std::to_string(pair->frame_id_1) + '_' + std::to_string(pair->frame_id_2);
    }

    Pair() {};
    Pair(unsigned int i, unsigned int j, std::vector<cv::DMatch>& input_matches,
      Eigen::Matrix4f & Tf, double mean_depth) {
      frame_id_1 = i;
      frame_id_2 = j;
      initial_matches = input_matches;
      Transform = Tf;
      appro_depth = mean_depth;
    }
};

// global graph class
template <class F = Frame, class M = Eigen::Matrix3f,class P = pcl::PointCloud<pcl::PointXYZRGB>::Ptr&> // F, M,P for frame/eigen mat/pc data type
struct Graph {
  public:
    Graph() {};
    Graph(const std::vector<Frame>& f_buf, const int& f_len, const Eigen::Matrix3f& k) :
    frame_buffer(f_buf), frame_length(f_len), K(k)
    {};
    std::vector<Frame> frame_buffer;
    int frame_length;
    M K;
  };

// struct Graph {

//   public:
//     Graph() {};
//     Graph(const std::vector<Frame>& f_buf, const int& f_len, const Eigen::Matrix3f& k) :
//     frame_buffer(f_buf), frame_length(f_len), K(k)
//     {};
//     std::vector<Frame> frame_buffer;
//     int frame_length;
//     Eigen::Matrix3f K;

// };

struct PointCloud {
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_rgb;
  std::vector<int> unique_point_id;
  std::vector<int> has_inlier;

  PointCloud() {
    pc_rgb = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
  }
};

// utility for various feature processing funcs
struct Utility {
  void AssignUnmatchedPointId(Frame& cur_frame, int& global_unique_point_id);
  void SetIntrinsicMat(std::vector<Frame>& frame_buffer, const Eigen::Matrix3f& K);
  void LinkMatchedPointID(Frame& frameID_1, Frame& frameID_2, const std::vector<cv::DMatch>& inlier_matches);
  void RecordIDInMat(std::vector<std::vector<bool>>& feature_track_matrix,
  const int& frameID, const std::vector<Frame>& frame_buffer);
  std::string type2str(int type); // check cv::Mat data type
};















}
#endif
