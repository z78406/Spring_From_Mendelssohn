// c++
#include <iostream>
#include <fstream>

// opencv lib
// #include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// pcl lib
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/visualization/common/common.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "viewer.hpp"
#include "estimate_motion.hpp"

using namespace sfm;

void MapViewer::DisplayFrame(const Frame& cur_frame, const std::string& window_name, int lasting_time) {
	cv::Mat frame;
	cv::namedWindow(window_name, 0);
	cv::drawKeypoints(cur_frame.img_rgb, cur_frame.keypoints, frame, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
	cv::imshow(window_name, frame);
	cv::waitKey(lasting_time);
}

void MapViewer::DisplayFrameMatch(const Frame& frameID_1, const Frame& frameID_2, const std::vector<cv::DMatch> &inlier_matches,
    const std::string& window_name, int lasting_time) {
	cv::Mat frame;
	cv::namedWindow(window_name, 0);
	cv::drawMatches(frameID_1.img_rgb, frameID_1.keypoints, frameID_2.img_rgb, frameID_2.keypoints, inlier_matches, frame);
	cv::imshow(window_name, frame);
	cv::waitKey(lasting_time);
}