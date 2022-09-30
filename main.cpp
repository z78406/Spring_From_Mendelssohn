// Copyright 2022 Zzy
// Author: Zzy-buffer
/* Version 0.1 @ 2022
*/
#include <iostream>
#include <algorithm>
#include <vector>
#include <fstream>
#include <string>
#include <map>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>

#include <opencv2/opencv.hpp>
#include <gflags/gflags.h>
#include <boost/filesystem.hpp>

#include "data_io.hpp"
#include "feature_process.hpp"
#include "my_package.hpp"
#include "viewer.hpp"
#include "frame.hpp"
#include "estimate_motion.hpp"

using namespace sfm;


// global def
typedef Eigen::Matrix3f m3f;
typedef pcl::PointCloud<pcl::PointXYZRGB>::Ptr& pcptr;
typedef cv::DMatch dmach;

// gflags def
DEFINE_string(input_img, "xxx.png", "Input images");
DEFINE_string(dir_img, "..", "Folder of Input images");
DEFINE_string(input_calib, "..", "Folder of Input calibration file");
DEFINE_string(keypoint_type, "SIFT", "keypoint type for feature extraction and matching");
DEFINE_bool(launch_viewer, false, "if display results");


// global classes
IO<std::string, m3f, pcptr> io_obj;
Graph<Frame, m3f, pcptr> graph;
Utility util;
Feature feat;
MapViewer mv;
MotionEstimator me;

// global vars
// img path list
static std::vector<std::string> paths_img;
// img name list
static std::vector<std::string> filenames_img;
// camera calibration mat ([fx, 0 ,cx; 0, fy, cy; 0 0 1])
static m3f K;
// frame buffer
static std::vector<Frame> frame_buffer;
// keypoint type
static std::string kp_type;
// global Pair set
static std::unordered_map<std::string, Pair> img_match_graph; // <pair_key: "i_j", Pair Struct>
// static std::vector<std::vector<Pair>> img_match_graph;
static bool show_result = false;
// <keypoint type, int> dict
static std::map<std::string, int> kp_type_dict =
{{"SIFT", 0}, {"SURF", 1}, {"ORB", 2}};


bool ReadImage(bool flag_not_set_img_folder) {
  // read file
  if (!flag_not_set_img_folder) {
    try {
      io_obj.ImportImageFiles(FLAGS_dir_img, ".jpg", paths_img, filenames_img);
      for (unsigned ind = 0; ind < filenames_img.size(); ++ind) {
        std::string path_img = std::string(FLAGS_dir_img) + "/" + filenames_img[ind];
        // std::cout<<path_img<<std::endl;
        cv::Mat3b input_img = cv::imread(path_img);
        if (input_img.empty()) {
          throw std::runtime_error("Input Error: input img not found.");
        }
        Frame cur_frame(input_img, ind, path_img);
        frame_buffer.push_back(cur_frame);
      }
      std::cout<<"Finish reading image list."<<std::endl;
      return 1;
    }
    catch (std::exception const& ex) {
    std::cerr << "Exception: " << ex.what() <<std::endl;
    return -1;
      }
  }
  else {
    throw std::runtime_error("Input Error: input img folder not found.");
    return 0;
  }
}

bool ReadCalib(bool flag_not_set_calib_folder) {
  // read file
  if (!flag_not_set_calib_folder) {
    try {
      io_obj.ImportCalib(FLAGS_input_calib, K);
      // std::cout<<K<<std::endl;
      if (!K.size()) {
        throw std::runtime_error("Input Error: input calibration file is empty.");
      }
      return 1;
    }
    catch (std::exception const& ex) {
    std::cerr << "Exception: " << ex.what() <<std::endl;
    return -1;
      }
  }
  else {
    throw std::runtime_error("Input Error: input calibration file folder not found.");
    return 0;
  }
}


bool FeatureMatching() {
  // Detect feature points in all the images
  std::cout << "Begin feature extraction" << std::endl;
  std::printf("Using %s as keypoint detection type.\n", kp_type.c_str());
  int keypoints_total_count = 0;
  for (unsigned int i = 0; i < frame_buffer.size(); i++) {
    if (i == 0)
    {
        std::cout << "Imported Calibration Matrix:\n"
                  << K << std::endl;
    }
    std::cout << "Feature extraction of Frame [ " << i << " ]" << std::endl;
    switch (kp_type_dict[kp_type]) {
      default:
        feat.DetectFeatureSIFT(frame_buffer[i], show_result);
        break;
      case 1:
        feat.DetectFeatureSURF(frame_buffer[i], 350, show_result);
        break;
      case 2:
        feat.DetectFeatureORB(frame_buffer[i], 5000, show_result);
        break;
    }
    keypoints_total_count += frame_buffer[i].keypoints.size();
    frame_buffer[i].init_p_id();
  }
  if (show_result) {  // show example keypoint extraction result at frame 0
    mv.DisplayFrame(frame_buffer[0]);
  }
  std::cout << "Feature extraction done" << std::endl;

  // pair-wise frame matching
  std::cout << "Begin pairwise feature matching" << std::endl;
  int num_min_pair = 20;  // min number of keypoint pairs
  int max_total_feature_num = keypoints_total_count;  // number of keypoints in all frames
  int global_unique_point_id = 0;  // count global ID of unique keypoint points
  //Feature track matrix: row: frames; colum: unique feature points
  std::vector<std::vector<bool>> feature_track_matrix(frame_buffer.size(), std::vector<bool>(max_total_feature_num, 0));



  // pairwise matching of frame: <i, j>. To do: try different matching sequences
  int max_frame_interval = 1; // max interval == i - j of matched frame pair <i, j>
  for (int i = 0; i < frame_buffer.size(); i++) {  // main frame
    // std::vector<Pair> cur_pairs;
    for (int j = std::max(i - max_frame_interval, 0); j < i; j++) {  // reference frame
      std::cout<<"Selcted frame pair is:"<<i<<" and "<<j<<std::endl;
      std::vector<dmach> initial_matches;  // initial matches from keypoint matching
      std::vector<dmach> inlier_matches;   // inlier matches after ransac refinement
      Eigen::Matrix4f extrinsic = Eigen::Matrix4f::Identity();
      double relative_depth = 1;

      // switch keypoint matching method
      // get initial matching results
      switch (kp_type_dict[kp_type]) {
        default:
          feat.Match2viewSIFT(frame_buffer[i], frame_buffer[j], initial_matches, (double)0.6, show_result);
          // feat.Match2viewSIFTBidirectional(frame_buffer[i], frame_buffer[j], initial_matches, (double)0.6, show_result);
          break;
        case 1:
          feat.Match2viewSURF(frame_buffer[i], frame_buffer[j], initial_matches, 0.5, show_result);
          break;
        case 2:
          feat.Match2viewORB(frame_buffer[i], frame_buffer[j], initial_matches, 0.5, show_result);
          break;
      }

      std::cout<<"size of initial matched keypoint set is:"<<(int)initial_matches.size()<<std::endl;

      /*
      potential bugs in me.EstimateE5points_RANSAC!!! (check within func for more details)
      */
      if (!initial_matches.empty() && initial_matches.size() > num_min_pair) { // refine by RANSAC
        me.EstimateE5points_RANSAC(frame_buffer[i], frame_buffer[j], initial_matches, inlier_matches, extrinsic);
        me.GetDepth(frame_buffer[i], frame_buffer[j], extrinsic, inlier_matches, relative_depth);
        if (show_result) {
          mv.DisplayFrameMatch(frame_buffer[i], frame_buffer[j], inlier_matches);
        }
      }

      // link matched point idx in frame i and j.
      util.LinkMatchedPointID(frame_buffer[i], frame_buffer[j], inlier_matches);
      // std::cout<<"matcher size is: " << inlier_matches.size()<<std::endl;
      // pair and all pairs related to frame i
      Pair cur_pair(i, j, inlier_matches, extrinsic, relative_depth);
      std::string pair_key = std::to_string(i) + '_' + std::to_string(j);
      img_match_graph.insert({pair_key, cur_pair});
      // cur_pairs.push_back(cur_pair);
      std::cout << "Frame [" << i << "] and Frame [" << j << "] matching done." << std::endl;
    }

    // img_match_graph.push_back(cur_pairs);
    // assign unique ID to unmatched keypoints in frame I
    util.AssignUnmatchedPointId(frame_buffer[i], global_unique_point_id);
    // record keypoints ID into a global recording matrix
    util.RecordIDInMat(feature_track_matrix, i, frame_buffer);
  }
  std::cout << "Pairwise feature matching done." << std::endl;
  std::cout << "Feature tracking done, there are " << global_unique_point_id << " unique points in total." << std::endl;
  return 0;
}

bool Build_Graph() {
  std::cout << "Start Triangulation to generate point cloud." << std::endl;
  PointCloud sparse_pointcloud;
  // copy buffer
  graph.frame_buffer = frame_buffer;
  graph.frame_length = frame_buffer.size();
  graph.K = K;
  // Notice the sequence order is important !!!
  // Since we define main frame i and reference frame j such that j < i,
  // Here we need to set main frame as i and reference frame as j such that (j < i)
  int init_frameID_1 = 3;
  int init_frameID_2 = 2;
  double depth_init = 10.0;
  // skip best initial frame selection
  // Initialization
  std::string pair_key = std::to_string(init_frameID_1) + "_" + std::to_string(init_frameID_2);
  if (img_match_graph.find(pair_key) == img_match_graph.end()) {
    std::cerr << "Exception: did not find matched frame pair!" << std::endl;
    return -1;
  }

  frame_buffer[init_frameID_1].k_ext = Eigen::Matrix4f::Identity();
  std::cout << "Frame [" << init_frameID_1 << "] 's pose: " << std::endl
            << frame_buffer[init_frameID_1].k_ext << std::endl;
  frame_buffer[init_frameID_2].k_ext = img_match_graph[pair_key].Transform * frame_buffer[init_frameID_1].k_ext;
  std::cout << "Frame [" << init_frameID_2 << "] 's pose: " << std::endl
            << frame_buffer[init_frameID_2].k_ext << std::endl;
  if (show_result)
    mv.DisplayFrame(frame_buffer[init_frameID_1]);
  if (show_result)
    mv.DisplayFrame(frame_buffer[init_frameID_2]);

  //Initial Frame Pair Triangulation
  me.doTriangulation(frame_buffer[init_frameID_1], frame_buffer[init_frameID_2],
    img_match_graph[pair_key].initial_matches, sparse_pointcloud, true);


  return 0;
}


int main(int argc, char* argv[]) {
  // load flags into gflags
  google::ParseCommandLineFlags(&argc, &argv, true);
  show_result = FLAGS_launch_viewer;
  // std::cout<<show_result<<std::endl;
  kp_type = FLAGS_keypoint_type;

  // read image list thread
  bool flag_not_set_img_folder = \
  gflags::GetCommandLineFlagInfoOrDie("dir_img").is_default;
  std::thread r_img_thread(ReadImage, flag_not_set_img_folder);

  // read calibration file thread
  bool flag_not_set_calib_folder = \
  gflags::GetCommandLineFlagInfoOrDie("input_calib").is_default;
  std::thread r_calib_thread(ReadCalib, flag_not_set_calib_folder);

  // collect thtread
  r_img_thread.join();
  r_calib_thread.join();

  // set intrinsic mat to frames
  util.SetIntrinsicMat(frame_buffer, K);

  // feature extraction and matching
  // read image list thread
  FeatureMatching();

  // build global graph from SFM
  Build_Graph();
  return 0;
}




