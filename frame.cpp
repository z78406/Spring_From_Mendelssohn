
#include "frame.hpp"

using namespace sfm;


void Frame::init_p_id() {
      for (int i = 0; i < keypoints.size(); i++)
        unique_pixel_id.push_back(-1); // default matching idx
        unique_pixel_has_match.push_back(false);
    }

void Utility::AssignUnmatchedPointId(Frame& cur_frame, int& global_unique_point_id) {
  // Assign ID to keypoint in each frame
  /* Notice that if two keypoints matched, one ID (id2) would be copied to the other (id1) by LinkMatchedPointID().
     E.g., p2(id2) = p1(id1) for matched <p1, p2> between frame i and j.
     Otherwise, it will be assigned a unique ID (id) that might be used in the next frame matching process.
  */
  int cur_frame_unique_point_id = 0;
  for (unsigned int index = 0; index < cur_frame.unique_pixel_id.size(); index++) {
  	// std::cout<<index<<std::endl;
    if (cur_frame.unique_pixel_id[index] < 0) {  // unmatched keypoint
      cur_frame.unique_pixel_id[index] = global_unique_point_id + cur_frame_unique_point_id;
      cur_frame_unique_point_id++;  // update local id for unmatched point in frame i
    }
  }
  // std::cout<<"########"<<std::endl;
  global_unique_point_id += cur_frame_unique_point_id;  // update global id for next unmatched point
}


void Utility::SetIntrinsicMat(std::vector<Frame>& frame_buffer, const Eigen::Matrix3f& K) {
  // set input intrinsic mat to each frame
  for (int i = 0; i < (int)frame_buffer.size(); i++) {
    frame_buffer[i].k_int = K;
  }
}


void Utility::LinkMatchedPointID(Frame& frameID_1, Frame& frameID_2,
	const std::vector<cv::DMatch>& inlier_matches) {
  // link matched point idx in frame frameID_1 and frameID_2
  // if frame frameID_1<p2> matches frame frameID_2<p1>, then frame frameID_1.unique_pixel_ids[p2] = p1;
  // perform like union-find point id registration for all matched points in every frame.
  for (int idx = 0; idx < inlier_matches.size(); idx++) {
    int quiry_point_idx = inlier_matches[idx].queryIdx; // query point idx (relative order in inlier_matches)
    int match_point_idx = inlier_matches[idx].trainIdx; // matched point idx (relative order in inlier_matches)
    int quiry_point_id = frameID_1.unique_pixel_id[quiry_point_idx]; // global ID of the query point
    int match_point_id = frameID_2.unique_pixel_id[match_point_idx]; // global ID of the matched point
    if (quiry_point_id < 0 || quiry_point_id != match_point_id) { // if quiry point is new and did not link to match point's id
      bool is_duplicated = 0;
      if (std::find(frameID_1.unique_pixel_id.begin(),
        frameID_1.unique_pixel_id.end(), match_point_id) != frameID_1.unique_pixel_id.end())
      { // match point links to another quiry point in frame 1. Allow only 1-1 point match between frame 1 and 2.
        is_duplicated = 1;
      }
      if (!is_duplicated) {  // no duplicate
        quiry_point_id = match_point_id;
        frameID_1.unique_pixel_has_match[quiry_point_idx] = 1;
      }
    }
  }
}

void Utility::RecordIDInMat(std::vector<std::vector<bool>>& feature_track_matrix,
  const int& frameID, const std::vector<Frame>& frame_buffer) {
  for (unsigned int i = 0; i < frame_buffer[frameID].unique_pixel_id.size(); i++) {
    feature_track_matrix[frameID][frame_buffer[frameID].unique_pixel_id[i]] = 1;
  }
}

std::string Utility::type2str(int type) {
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}