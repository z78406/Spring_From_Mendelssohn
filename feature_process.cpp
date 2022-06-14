#include <iostream>
#include <chrono>
#include <vector>
#include <fstream>
#include <string>

// #include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "feature_process.hpp"
using namespace sfm;

void Feature::DetectFeatureSIFT(Frame& cur_frame, bool show) {
  // extract keypoint and descriptors from the input frame.
  // Input:
  //  cur_frame: an image frame struct to work on
  //  show: if display result

  cv::Ptr<cv::FeatureDetector> detector = cv::SIFT::create();
  std::chrono::steady_clock::time_point tic = std::chrono::steady_clock::now();
  detector->detectAndCompute(cur_frame.img_rgb, cv::noArray(), cur_frame.keypoints, cur_frame.descriptors);

  std::chrono::steady_clock::time_point toc = std::chrono::steady_clock::now();
  std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(toc - tic);
  std::cout << "extract SIFT cost = " << time_used.count() << " seconds. " << std::endl;
  std::cout << "Found " << cur_frame.descriptors.size() << " features" << std::endl;

  if (show) {
    cv::Mat feature_image;
    cv::namedWindow("SIFT features",0);
    cv::drawKeypoints(cur_frame.img_rgb, cur_frame.keypoints, feature_image, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
    cv::imshow("SIFT features", feature_image);
    cv::waitKey(0); // Wait for a keystroke in the window
  }
}


void Feature::DetectFeatureORB(Frame& cur_frame, int max_num, bool show) {
  cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create(max_num);

  std::chrono::steady_clock::time_point tic = std::chrono::steady_clock::now();
  detector->detectAndCompute(cur_frame.img_rgb, cv::noArray(), cur_frame.keypoints, cur_frame.descriptors);

  std::chrono::steady_clock::time_point toc = std::chrono::steady_clock::now();
  std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(toc - tic);
  std::cout << "extract ORB cost = " << time_used.count() << " seconds. " << std::endl;
  std::cout << "Found " << cur_frame.descriptors.size() << " features" << std::endl;
  if (show) {
    cv::Mat feature_image;
    cv::namedWindow("ORB features",0);
    cv::drawKeypoints(cur_frame.img_rgb, cur_frame.keypoints, feature_image, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
    cv::imshow("ORB features", feature_image);
    cv::waitKey(0); // Wait for a keystroke in the window
  }
}

void Feature::DetectFeatureSURF(Frame& cur_frame, int minHessian, bool show) {
  cv::Ptr<cv::FeatureDetector> detector = cv::xfeatures2d::SURF::create(minHessian);

  std::chrono::steady_clock::time_point tic = std::chrono::steady_clock::now();
  detector->detectAndCompute(cur_frame.img_rgb, cv::noArray(), cur_frame.keypoints, cur_frame.descriptors);

  std::chrono::steady_clock::time_point toc = std::chrono::steady_clock::now();
  std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(toc - tic);
  std::cout << "extract SURF cost = " << time_used.count() << " seconds. " << std::endl;
  std::cout << "Found " << cur_frame.descriptors.size() << " features." << std::endl;
  std::cout<<show<<std::endl;
  if (show) {
    cv::Mat feature_image;
    cv::namedWindow("SURF features",0);
    cv::drawKeypoints(cur_frame.img_rgb, cur_frame.keypoints, feature_image, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
    cv::imshow("SURF features", feature_image);
    cv::waitKey(0); // Wait for a keystroke in the window
  }
}

void Feature::Match2viewSIFT(Frame& frameID_1, Frame& frameID_2,
    std::vector<cv::DMatch>& matches_buffer, double dist_ratio, bool show) {
  // match keypoints given two nearby frames.
  // Input:
  //  frameID_1:  image frame struct 1
  //  frameID_2:  image frame struct 2
  //  matches_buffer: buffer to store matching result
  //  dist_ratio distance ratio, Default is 0.6 (follow sfmedu)
  //  show: if display result

  /*
  related opencv class: cv::DMatch, cv::FlannBasedMatcher.
  */

  // def required vars
  int min_neighboringmatching = 20;

  // use flann to find best keypoint matching pairs
  cv::FlannBasedMatcher matcher;
  // knn matcher (kd-tree index)
  std::vector<std::vector<cv::DMatch>> knn_matches;
  // matcher (kd-tree index)
  std::vector<cv::DMatch> best_matches;
  // obtain initial k nearest matching result
  std::chrono::steady_clock::time_point tic = std::chrono::steady_clock::now();
  matcher.knnMatch(frameID_1.descriptors, frameID_2.descriptors, knn_matches, 2); // k == 2
  std::cout << "Initial SIFT keypoint matching done." << std::endl;


  // Initial matching result
  for (int i = 0; i < knn_matches.size(); i++) {
    if (show)
        best_matches.push_back(knn_matches[i][0]);
  }
  // Filter matches based on lowe's algorithm (matching distance)
  // iteratively increase ratio to add more match pairs
  while (matches_buffer.size() < min_neighboringmatching) {
    matches_buffer.clear();
    for (int i = 0; i < knn_matches.size(); i++) {
      if (knn_matches[i][0].distance < dist_ratio * knn_matches[i][1].distance) {
          matches_buffer.push_back(knn_matches[i][0]);
      }
    }
    if (dist_ratio <= 1) {
      if (dist_ratio < 1)
        dist_ratio += 0.1;
      else
        break;
    }
  }

  if (matches_buffer.size() < min_neighboringmatching)
    std::printf("Warning: did not find enough keypoint matching pairs \
    between frame %d and frame %d\n", frameID_1.frame_id, frameID_2.frame_id);

  std::chrono::steady_clock::time_point toc = std::chrono::steady_clock::now();
  std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(toc - tic);
  std::cout << "match SIFT cost = " << time_used.count() << " seconds. " << std::endl;
  std::cout << "# Correspondence: Initial [ " << knn_matches.size() << " ]  Inlier [ " << matches_buffer.size() << " ]" << std::endl;

  if (show) {
    cv::Mat initial_match_image;
    cv::Mat match_image;
    cv::namedWindow("Initial matches",0);
    cv::namedWindow("Inlier matches",0);
    cv::drawMatches(frameID_1.img_rgb, frameID_1.keypoints, frameID_2.img_rgb, frameID_2.keypoints, best_matches, initial_match_image);
    cv::drawMatches(frameID_1.img_rgb, frameID_1.keypoints, frameID_2.img_rgb, frameID_2.keypoints, matches_buffer, match_image);
    cv::imshow("Initial matches", initial_match_image);
    cv::imshow("Inlier matches", match_image);
    cv::waitKey(0);
  }
}

void Feature::Match2viewSIFTBidirectional(Frame& frameID_1, Frame& frameID_2,
      std::vector<cv::DMatch> &matches_buffer, double dist_ratio, bool show) {
  // match keypoints given two nearby frames.
  // Input:
  //  frameID_1:  image frame struct 1
  //  frameID_2:  image frame struct 2
  //  matches_buffer: buffer to store matching result
  //  dist_ratio distance ratio, Default is 0.6 (follow sfmedu)
  //  show: if display result

  /*
  related opencv class: cv::DMatch, cv::FlannBasedMatcher.
  */

  // def required vars
  int min_neighboringmatching = 20;

  // use flann to find best keypoint matching pairs
  cv::FlannBasedMatcher matcher1;
  cv::FlannBasedMatcher matcher2;
  // knn matcher (kd-tree index)
  std::vector<std::vector<cv::DMatch>> knn_matches1;
  std::vector<std::vector<cv::DMatch>> knn_matches2;
  // matcher (kd-tree index)
  std::vector<cv::DMatch> best_matches;
  // obtain initial k nearest matching result
  std::chrono::steady_clock::time_point tic = std::chrono::steady_clock::now();
  matcher1.knnMatch(frameID_1.descriptors, frameID_2.descriptors, knn_matches1, 2); // k == 2
  matcher2.knnMatch(frameID_2.descriptors, frameID_1.descriptors, knn_matches2, 2); // k == 2
  std::cout << "Initial SIFT keypoint matching done." << std::endl;
  // Initial matching result
  for (int i = 0; i < knn_matches1.size(); i++) {
    if (show)
        best_matches.push_back(knn_matches1[i][0]);
  }
  // Filter matches based on bidirectional matching check.
  // iteratively increase ratio to add more match pairs
  while (matches_buffer.size() < min_neighboringmatching) {
    matches_buffer.clear();
    // for (int i = 0; i < knn_matches1.size(); i++) {
    //   // find best keypoint j in frame 2 for keypoint i in frame 1
    //   int best_j_for_i = knn_matches1[i][0].trainIdx;
    //   float best_dist = knn_matches1[i][0].distance;
    //   int sec_best_j_for_i = knn_matches1[i][1].trainIdx;
    //   float sec_best_dist = knn_matches1[i][1].distance;

    //   if (best_dist < dist_ratio * sec_best_dist) {
    //     // find best i in frame 1 for "best keypoint j" in frame 2
    //     int best_i_for_j = knn_matches2[best_j_for_i][0].trainIdx;
    //     float best_dist = knn_matches2[best_j_for_i][0].distance;
    //     int best_i_for_j = knn_matches2[sec_best_j_for_i][0].trainIdx;
    //     float best_dist = knn_matches2[sec_best_j_for_i][0].distance;
    //     if (best_dist < dist_ratio * sec_best_dist)
    //       matches_buffer.push_back(knn_matches[i][0]);
    //   }
    // }

    // from https://github.com/kipr/opencv/blob/master/samples/cpp/descriptor_extractor_matcher.cpp
    // to do: add distance constraint such as in Match2viewSIFT();
    // Current matching result seems to cover too many false matches.
    for (int i = 0; i < knn_matches1.size(); i++) { // every matched keypoint i in frame 1
      bool findCrossCheck = false;
      for (int c_j = 0; c_j < knn_matches1[i].size(); c_j++) {  // c_j'th matched keypoint j in frame 2
        cv::DMatch forward = knn_matches1[i][c_j];  // struct DMatch of keypoint j
          int matched_i_for_fk = forward.trainIdx;  // matched index list of keypoint i, for keypoint j
          for (int c_i = 0; c_i < knn_matches2[matched_i_for_fk].size(); c_i++) {
            cv::DMatch backward = knn_matches2[matched_i_for_fk][c_i]; // matched keypoint i, for keypoint j
            if (backward.trainIdx == forward.queryIdx) { // cross-match
              matches_buffer.push_back(forward);
              findCrossCheck = true;
              break;
            }
          }
          if (findCrossCheck)
            break;
      }
    }

    if (dist_ratio <= 1) {
      if (dist_ratio < 1)
        dist_ratio += 0.1;
      else
        break;
    }
  }

  if (matches_buffer.size() < min_neighboringmatching) {
    std::printf("Warning: did not find enough keypoint matching pairs \
    between frame %d and frame %d\n", frameID_1.frame_id, frameID_2.frame_id);
  }

  std::chrono::steady_clock::time_point toc = std::chrono::steady_clock::now();
  std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(toc - tic);
  std::cout << "match SIFT cost = " << time_used.count() << " seconds. " << std::endl;
  std::cout << "# Correspondence: Initial [ " << knn_matches1.size() << " ]  Inlier [ " << matches_buffer.size() << " ]" << std::endl;

  if (show) {
    cv::Mat initial_match_image;
    cv::Mat match_image;
    cv::namedWindow("Initial matches",0);
    cv::namedWindow("Inlier matches",0);
    cv::drawMatches(frameID_1.img_rgb, frameID_1.keypoints, frameID_2.img_rgb, frameID_2.keypoints, best_matches, initial_match_image);
    cv::drawMatches(frameID_1.img_rgb, frameID_1.keypoints, frameID_2.img_rgb, frameID_2.keypoints, matches_buffer, match_image);
    cv::imshow("Initial matches", initial_match_image);
    cv::imshow("Inlier matches", match_image);
    cv::waitKey(0);
  }
}

void Feature::Match2viewSURF(Frame& frameID_1, Frame& frameID_2, std::vector<cv::DMatch>& matches_buffer,
                     double dist_ratio, bool show) {
  // match keypoints given two nearby frames.
  // Input:
  //  frameID_1:  image frame struct 1
  //  frameID_2:  image frame struct 2
  //  matches_buffer: buffer to store matching result
  //  dist_ratio distance ratio, Default is 0.6 (follow sfmedu)
  //  show: if display result

  /*
  related opencv class: cv::DMatch, cv::FlannBasedMatcher.
  */

  // def required vars
  int min_neighboringmatching = 20;

  // use flann to find best keypoint matching pairs
  cv::FlannBasedMatcher matcher;
  // knn matcher (kd-tree index)
  std::vector<std::vector<cv::DMatch>> knn_matches;
  // matcher (kd-tree index)
  std::vector<cv::DMatch> best_matches;
  // obtain initial k nearest matching result
  std::chrono::steady_clock::time_point tic = std::chrono::steady_clock::now();
  matcher.knnMatch(frameID_1.descriptors, frameID_2.descriptors, knn_matches, 2); // k == 2
  std::cout << "Initial SURF keypoint matching done." << std::endl;


  // Initial matching result
  for (int i = 0; i < knn_matches.size(); i++) {
    if (show)
        best_matches.push_back(knn_matches[i][0]);
  }
  // Filter matches based on lowe's algorithm (matching distance)
  // iteratively increase ratio to add more match pairs
  while (matches_buffer.size() < min_neighboringmatching) {
    matches_buffer.clear();
    for (int i = 0; i < knn_matches.size(); i++) {
      if (knn_matches[i][0].distance < dist_ratio * knn_matches[i][1].distance) {
          matches_buffer.push_back(knn_matches[i][0]);
      }
    }
    if (dist_ratio <= 1) {
      if (dist_ratio < 1)
        dist_ratio += 0.1;
      else
        break;
    }
  }
  // std::cout<<"size is:"<<matches_buffer.size()<<std::endl;
  if (matches_buffer.size() < min_neighboringmatching)
    std::printf("Warning: did not find enough keypoint matching pairs \
    between frame %d and frame %d\n", frameID_1.frame_id, frameID_2.frame_id);

  std::chrono::steady_clock::time_point toc = std::chrono::steady_clock::now();
  std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(toc - tic);
  std::cout << "match SURF cost = " << time_used.count() << " seconds. " << std::endl;
  std::cout << "# Correspondence: Initial [ " << knn_matches.size() << " ]  Inlier [ " << matches_buffer.size() << " ]" << std::endl;

  if (show) {
    cv::Mat initial_match_image;
    cv::Mat match_image;
    cv::namedWindow("Initial matches",0);
    cv::namedWindow("Inlier matches",0);
    cv::drawMatches(frameID_1.img_rgb, frameID_1.keypoints, frameID_2.img_rgb, frameID_2.keypoints, best_matches, initial_match_image);
    cv::drawMatches(frameID_1.img_rgb, frameID_1.keypoints, frameID_2.img_rgb, frameID_2.keypoints, matches_buffer, match_image);
    cv::imshow("Initial matches", initial_match_image);
    cv::imshow("Inlier matches", match_image);
    cv::waitKey(0);
  }
}

void Feature::Match2viewORB(Frame& frameID_1, Frame& frameID_2, std::vector<cv::DMatch>& match_buffer,
                     double dist_ratio, bool show) {

}

















