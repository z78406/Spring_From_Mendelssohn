// Ceres
#include <ceres/loss_function.h>

#include <chrono>


#include "bundle_adjustment.hpp"

using namespace sfm;

void BundleAdjustment::DoSFMBA(std::vector<Frame>& frame_buffer, std:;vector<bool>& frameID,
		pointcloud_sparse_t& point_cloud_sparse, double fix_calib_tolerance_BA,
		int reference_frameID)
{
	std::chrono::steady_clock::time_point tic = std::chrono::steady_clock::now();

	InitBA();
	SetBAProblem();
	SolveBA();


}