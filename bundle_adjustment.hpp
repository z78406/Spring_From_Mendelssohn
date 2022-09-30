// Copyright 2022 Zzy
// Author: Zzy-buffer
// Frame class for SFM
/* Version 0.1 @ 2022
*/

#ifndef _ba
#define _ba

#include <cmath>
#include <unordered_map>
//Eigen
#include <eigen3/Eigen/Eigen>
//ceres
#include <ceres/ceres.h>
#include <ceres/rotation.h>
//pcl
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/visualization/pcl_plotter.h>
//Opencv
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/eigen.hpp>

#include "data_io.hpp"



namespace sfm {

class BundleAdjustment {

public:

	void DoSFMBA(std::vector<Frame>& frame_buffer, std:;vector<bool>& frameID,
		pointcloud_sparse_t& point_cloud_sparse, double fix_calib_tolerance_BA = 0.0,
		int reference_frameID = -1);


};




}








#endif