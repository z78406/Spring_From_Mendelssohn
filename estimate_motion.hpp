#ifndef _estimate_motion
#define _estimate_motion

#include <opencv2/calib3d.hpp>

#include "frame.hpp"


namespace sfm
{

class MotionEstimator {
public:
  // 8 points method to estimate fundamental mat
  void EstimateF8points(std::vector<cv::KeyPoint> &keypoints1,
                        std::vector<cv::KeyPoint> &keypoints2,
                        std::vector<cv::DMatch> &matches,
                        Eigen::Matrix3f &K,
                        Eigen::Matrix4f &T);
  // essential mat from fundamental mat (known intrinsic mat)
  void EstimateEfromF();

  // camera relative pose [R|T] from essenstial mat
  void EstimateRTfromE(cv::Mat& essential_matrix,
  std::vector<cv::Point2f>& pst1, std::vector<cv::Point2f>& pst2,
  cv::Mat& cam_intrinsic, cv::Mat& Rotation, cv::Mat& Translation,
  cv::Mat& inlier_matches_indicator);

  void Triangulation(cv::Mat& T_1, cv::Mat& T_2,
  std::vector<cv::Point2f>& pts_1, std::vector<cv::Point2f>& pts_2, cv::Mat& pts_3d);

  // camera pose (wrt world) [R|T] from 3D-2D projection pairs
  void EstimateRTfromP(std::vector<cv::Point3f>& pointset3d, std::vector<cv::Point2f>& pointset2d,
    cv::Mat& camera_mat, cv::Mat& distort_para, cv::Mat& r_vec, cv::Mat& t_vec, cv::Mat& inlier_matches_indicator,
    bool use_initial_guess = false, int iterationsCount = 100, double ransac_thre = 8.0, double ransac_prob = 0.99,
     int method = cv::SOLVEPNP_EPNP);

  // 5 points method to estimate essential mat with RANSAC
  void EstimateE5points_RANSAC(Frame& frameID_1, Frame& frameID_2,
      std::vector<cv::DMatch>& initial_matches, std::vector<cv::DMatch>& inlier_matches,
      Eigen::Matrix4f &Transform, double ransac_thre = 1.0, double ransac_prob = 0.99, bool show = false);

  // void EstimateE5points_RANSAC(Frame& cur_frame_1, Frame& cur_frame_2,
  //                              std::vector<cv::DMatch> &matches, std::vector<cv::DMatch> &inlier_matches,
  //                              Eigen::Matrix4f &T,
  //                              double ransac_thre = 1.0, double ransac_prob = 0.99, bool show = false);


  // P3P problem (https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html)
  void EstimateP3P_RANSAC(Frame& cur_frame, PointCloud &cur_pc_3d,
                               double ransac_prob = 0.99, double ransac_thre = 2.5, int iterationsCount = 50000, bool show = false);

  void GetDepth(Frame& frameID_1, Frame& frameID_2,
                          Eigen::Matrix4f& Transform, std::vector<cv::DMatch>& matches,
                          double &appro_depth, int random_rate = 20);
  // tansform points from pixel space to camera coordinate space
  cv::Point2f pixel2cam(const cv::Point2f &p, const cv::Mat &K);

};

}


#endif