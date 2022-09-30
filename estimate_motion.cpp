// Eigen
#include <eigen3/Eigen/Eigen>

//OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
// #include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

// PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/common/common.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/statistical_outlier_removal.h>

// Others
#include <boost/thread.hpp>
#include <boost/chrono.hpp>
#include <chrono>
#include <iostream>

#include "estimate_motion.hpp"
#include "frame.hpp"

using namespace sfm;

Utility util_in_em;

void MotionEstimator::EstimateEfromF() {

}


void MotionEstimator::EstimateRTfromE(cv::Mat& essential_matrix,
  std::vector<cv::Point2f>& pst1, std::vector<cv::Point2f>& pst2,
  cv::Mat& cam_intrinsic, cv::Mat& Rotation, cv::Mat& Translation,
  cv::Mat& inlier_matches_indicator) {
  /*
  Decompose Rotation/Translation Mat from Essential Matrix.
  Input:
    - essential_matrix: calculated E mat
    - pst1: matched keypoint set1 from frame 1
    - pst2: matched keypoint set2 from frame 2
    - cam_intrinsic: intrinsic(calib) mat 3 x 3 of the camera
    - inlier_matches_indicator: indicator of inlier matched keypoints (bool loc == 1)
      from ransac essential mat calculation output
  Output:
    - Rotation: estimated rotation mat
    - Translation: estimated translation mat
  */
  cv::recoverPose(essential_matrix, pst1, pst2, cam_intrinsic, Rotation, Translation, inlier_matches_indicator);

}

void MotionEstimator::EstimateRTfromP(std::vector<cv::Point3f>& pointset3d, std::vector<cv::Point2f>& pointset2d,
    cv::Mat& camera_mat, cv::Mat& distort_para, cv::Mat& r_vec, cv::Mat& t_vec, cv::Mat& inlier_matches_indicator,
    bool use_initial_guess, int iterationsCount, double ransac_thre, double ransac_prob,
     int method) {
  /*
  Estimate camera Extrinsic mat from 3D/2D observation pairs.
  Note the opencv func solvePnPRansac() only solves for extrinsic mat.
  General PnP solutions are able to solve also intrinsic mat.
  Input:
    - pointset2d: frame where 2D points locate
    - pointset3d: scene where 3D points locate
    - camera_mat: camera calib(Intrinsic) file
    - distort_para: camera distortion params
    - use_initial_guess: if initial guess of r_vec/t_vec is provided
    - iterationsCount: num of ransac iterations
    - ransac_thre: params for cv::solvePnPRansac
    - ransac_prob: params for cv::solvePnPRansac
    - inlier_matches_indicator: indicator of final inlier pairs used for estimating R/T
    - method: PnP solver method
  Output:
    - r_vec rotation matrix that changes of basis from frame 1 to frame 2
    - t_vec translation matrix that changes of basis from frame 1 to frame 2
  */
  cv::solvePnPRansac(pointset3d, pointset2d, camera_mat, distort_para, r_vec, t_vec,
                       false, iterationsCount, ransac_thre, ransac_prob, inlier_matches_indicator, cv::SOLVEPNP_EPNP);

}


void MotionEstimator::Triangulation(cv::Mat& T_1, cv::Mat& T_2,
  std::vector<cv::Point2f>& pts_1, std::vector<cv::Point2f>& pts_2, cv::Mat& pts_3d) {
  /*
  Perform triangulation given paired point sets.
  Input:
    - T_1: Transformation Mat of frame 1 defined as [R|T] where R is identity and T is [0, 0, 0]
    - T_2: Transformation Mat of frame 2 defined as Transform * T_1
    - pts_1: matched keypoint set in frame 1
    - pts_2: matched keypoint set in frame 2
  Output:
    - pts_3d: triangulated points in 3D (4, N)
  */
  cv::triangulatePoints(T_1, T_2, pts_1, pts_2, pts_3d);
}

void MotionEstimator::doTriangulation(Frame& frameID_1, Frame& frameID_2, const std::vector<cv::DMatch>& matches,
  PointCloud& sparse_pointcloud, bool show) {
  /*
  Input:
    - frameID_X: Xth frame
    - matches: frame matching result buffer
  Output:
    - sparse_pointcloud: pointcloud class
  */
  std::chrono::steady_clock::time_point tic = std::chrono::steady_clock::now();
  // cam pose 1, came pose 2, cam intrinsic
  cv::Mat T1_mat, T2_mat, k_int;
  // Extrinsic Mat
  Eigen::Matrix<float, 3, 4> T1_pose = frameID_1.k_ext.block(0, 0, 3, 4);
  Eigen::Matrix<float, 3, 4> T2_pose = frameID_2.k_ext.block(0, 0, 3, 4);
  cv::eigen2cv(T1_pose, T1_mat);
  cv::eigen2cv(T2_pose, T2_mat);
  cv::eigen2cv(frameID_1.k_int, k_int);

  std::vector<cv::Point2f> pointset1, pointset2;
  int new_triangular_pts_num = 0;
  // std::cout<<"################"<<matches.size()<<std::endl;
  for (int i = 0; i < matches.size(); i++) {
    int quiry_point_id =frameID_1.unique_pixel_id[matches[i].queryIdx];
    std::vector<int> pointcloud_pointset = sparse_pointcloud.unique_point_id;

    bool pts_already_in_pc = std::find(pointcloud_pointset.begin(),
      pointcloud_pointset.end(), quiry_point_id) != pointcloud_pointset.end();

    // bool pts_already_in_pc = 0;
    // for (int k = 0; k < sparse_pointcloud.unique_point_id.size(); k++)
    // {
    //     if (sparse_pointcloud.unique_point_id[k] == quiry_point_id)
    //     {
    //         pts_already_in_pc = 1;
    //         break;
    //     }
    // }

    if (!pts_already_in_pc) { // if a matched keypoint in frame 1 is a newly incoming point in the pointcloud.
      //update point cloud
      pointcloud_pointset.push_back(quiry_point_id);
      sparse_pointcloud.has_inlier.push_back(1);
      // update 2d pointset (from image space to cam space as homogeneous result)
      pointset1.push_back(pixel2cam(frameID_1.keypoints[matches[i].queryIdx].pt, k_int));
      pointset2.push_back(pixel2cam(frameID_2.keypoints[matches[i].trainIdx].pt, k_int));
      // update triangulation counter
      new_triangular_pts_num++;
    }
  }

  // triangulation newly incoming matched point pairs into points in the point cloud
  cv::Mat pts_3d_homo; // (4, N) triangulated point sets
  if (pointset1.size() > 0) {
    Triangulation(T1_mat, T2_mat, pointset1, pointset2, pts_3d_homo);
  }

  // get new point cloud point from calculation results (by homogeneous to cartesian mapping)
  for (int i = 0; i < pts_3d_homo.cols; i++) {
    cv::Mat pts_3d = pts_3d_homo.col(i);
    pts_3d /= pts_3d.at<float>(3, 0);
    pcl::PointXYZRGB new_pt;
    // XYZ
    new_pt.x = pts_3d.at<float>(0, 0);
    new_pt.y = pts_3d.at<float>(1, 0);
    new_pt.z = pts_3d.at<float>(2, 0);
    // RGB
    cv::Point2f pts_2d = frameID_1.keypoints[matches[i].queryIdx].pt;
    uchar blue = frameID_1.img_rgb.at<cv::Vec3b>(pts_2d.y, pts_2d.x)[0];
    uchar green = frameID_1.img_rgb.at<cv::Vec3b>(pts_2d.y, pts_2d.x)[1];
    uchar red = frameID_1.img_rgb.at<cv::Vec3b>(pts_2d.y, pts_2d.x)[2];

    new_pt.r = 1.0 * red;
    new_pt.g = 1.0 * green;
    new_pt.b = 1.0 * blue;

    sparse_pointcloud.pc_rgb->points.push_back(new_pt);
  }
  std::cout << "Triangulate [ " << new_triangular_pts_num << " ] new points, [ " << sparse_pointcloud.pc_rgb->points.size()
  << " ] points in total." << std::endl;
  std::chrono::steady_clock::time_point toc = std::chrono::steady_clock::now();
  std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(toc - tic);
  std::cout << "Triangularization done in " << time_used.count() << " seconds. " << std::endl;

  if (show) {
    // show matched img pair result
    cv::Mat matched_img_pair;
    cv::namedWindow("Triangulation Matched Points Visualization", 0);
    cv::drawMatches(frameID_1.img_rgb, frameID_1.keypoints, frameID_2.img_rgb, frameID_2.keypoints, matches, matched_img_pair);
    cv::imshow("Triangularization matches", matched_img_pair);
    cv::waitKey(0);
    cv::destroyAllWindows();
    // start pointcloud visualizer
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("SFM viewer."));
    viewer->setBackgroundColor(1.0, 1.0, 1.0);

    // Draw camera object in viewer
    char t[256];
    std::string s;
    int n = 0;
    float frame_color_r, frame_color_g, frame_color_b;
    float sphere_size = 0.2;
    float line_size_cam = 0.4;

    pcl::PointXYZ pt_cam1(frameID_1.k_ext(0, 3), frameID_1.k_ext(1, 3), frameID_1.k_ext(2, 3)); // camera1's position
    pcl::PointXYZ pt_cam2(frameID_2.k_ext(0, 3), frameID_2.k_ext(1, 3), frameID_2.k_ext(2, 3)); // camera2's position
    // draw cam body and base line
    sprintf(t, "%d", n);
    s = t;
    viewer->addSphere(pt_cam1, sphere_size, 1.0 ,0.0, 0.0, s);
    n++;

    sprintf(t, "%d", n);
    s = t;
    viewer->addSphere(pt_cam2, sphere_size, 0.0 ,0.0, 1.0, s);
    n++;

    sprintf(t, "%d", n);
    s = t;
    viewer->addLine(pt_cam1, pt_cam2, 0.0 ,1.0, 0.0, s);
    n++;

    viewer->addPointCloud(sparse_pointcloud.pc_rgb, "Sparse PointCloud.");
    std::cout << "Click X(close) to continue..." << std::endl;
     while (!viewer->wasStopped()) {
      viewer->spinOnce(1000); // wait and press q to exit
        // viewer->spin();
        // viewer->close();
        // boost::this_thread::sleep(boost::posix_time::microseconds(100000));
        // boost::this_thread::sleep_for(boost::chrono::milliseconds(100000));
    }

  }
  std::cout << "Generate new sparse point cloud done." << std::endl;
}



void MotionEstimator::EstimateE5points_RANSAC(Frame& frameID_1, Frame& frameID_2,
      std::vector<cv::DMatch>& initial_matches, std::vector<cv::DMatch>& inlier_matches,
      Eigen::Matrix4f& Transform, double ransac_thre, double ransac_prob, bool show) {
  // Estimating [R|T] from matched keypoint pairs using x'Fx = 0 with RANSAC.

	std::chrono::steady_clock::time_point tic = std::chrono::steady_clock::now();
	std::vector<cv::Point2f> pointset1;
	std::vector<cv::Point2f> pointset2;

	for (int i = 0; i < (int)initial_matches.size(); i++) {  // initial keypoint matching result
    cv::Point2f pt_frame1 = frameID_1.keypoints[initial_matches[i].queryIdx].pt;  // coord of kp in frame 1
    cv::Point2f pt_frame2 = frameID_2.keypoints[initial_matches[i].trainIdx].pt;  // coord of kp in frame 2
    pointset1.push_back(pt_frame1);
    pointset2.push_back(pt_frame2);
	}

  cv::Mat cam_intrinsic;
  cv::eigen2cv(frameID_1.k_int, cam_intrinsic);
  cv::Mat essential_matrix;
  cv::Mat inlier_matches_indicator; // indicating inlier sets of keypoint pairs given essential_matrix
  // 5 point methods with ransac
  essential_matrix = cv::findEssentialMat(pointset1, pointset2, cam_intrinsic,
                cv::RANSAC, ransac_prob, ransac_thre, inlier_matches_indicator);
  std::cout << "essential_matrix is " << std::endl
            << essential_matrix << std::endl;


  std::string ty = util_in_em.type2str(inlier_matches_indicator.type());
  printf("Inlier Mask Matrix: %s %dx%d \n", ty.c_str(), inlier_matches_indicator.rows, inlier_matches_indicator.cols);

  // pick inlier keypoint match pairs


  // cv::Mat inlier_mask;
  // inlier_matches_indicator.convertTo(inlier_mask, CV_32F); // size: (n, 1)
  // for (int i = 0; i < initial_matches.size(); i++) {
  //   if (inlier_mask.at<float>(i) == 1) // inliear point pairs
  //     inlier_matches.push_back(initial_matches[i]);
  // }

  /*
  Potential bugs!!!
  inlier_matches_indicator.at<int>(i) == 1 will not cause heap corruption
  but element in the mat is somehow wiredly large.
  */
  for (int i = 0; i < initial_matches.size(); i++) {
    if (inlier_matches_indicator.at<int>(i) == 1) {// inlier point pairs
      inlier_matches.push_back(initial_matches[i]);
      // std::cout<<i<<std::endl;
    }
  }


  // // pick inlier keypoint match pairs
  // cv::Mat mask;
  // inlier_matches_indicator.convertTo(mask, CV_32F);
  // for (auto it = mask.begin<float>(); it != mask.end<float>(); it++) {
  //   if (*it == 1) // inliear point pairs
  //     inlier_matches.push_back(initial_matches[int(it - mask.begin<float>())]);
  // }

  // obtain R,T from essential mat. Note that t is up-to-scale.
  cv::Mat R, T;
  EstimateRTfromE(essential_matrix, pointset1, pointset2, cam_intrinsic, R, T, inlier_matches_indicator);
  // cv::recoverPose(essential_matrix, pointset1, pointset2, cam_intrinsic, R, T, inlier_matches_indicator);
  // calculate finishing time
  std::chrono::steady_clock::time_point toc = std::chrono::steady_clock::now();
  std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(toc - tic);
  std::cout << "Estimate the Relative Camera Motion [2D-2D] costs = " << time_used.count() << " seconds. " << std::endl;
  std::cout << "Find [" << inlier_matches.size() << "] inlier matches from [" << initial_matches.size() << "] total matches." << std::endl;

  // copy cv-mat to eigen-mat
  Eigen::Matrix3f R_eigen;
  Eigen::Vector3f T_eigen;
  cv::cv2eigen(R, R_eigen);
  cv::cv2eigen(T, T_eigen);
  Transform.block(0, 0, 3, 3) = R_eigen;
  Transform.block(0, 3, 3, 1) = T_eigen;
  Transform.row(3) << 0, 0, 0, 1;

  std::cout << "Estimated Relative Camera Transform is " << std::endl
            << Transform << std::endl;

  if (show) {
    cv::Mat ransac_match_image;
    cv::namedWindow("RANSAC inlier matches", 0);
    cv::drawMatches(frameID_1.img_rgb, frameID_1.keypoints, frameID_2.img_rgb, frameID_2.keypoints, inlier_matches, ransac_match_image);
    cv::imshow("RANSAC inlier matches", ransac_match_image);
    cv::waitKey(0);
  }
}


/* From EasySFM file, for ablation test.
*/
// void MotionEstimator::EstimateE5points_RANSAC(Frame &frameID_1, Frame &frameID_2,
//                                               std::vector<cv::DMatch> &initial_matches, std::vector<cv::DMatch> &inlier_matches,
//                                               Eigen::Matrix4f &Transform, double ransac_thre, double ransac_prob, bool show) {
//   // Estimating [R|T] from matched keypoint pairs using x'Fx = 0 with RANSAC.
//   std::chrono::steady_clock::time_point tic = std::chrono::steady_clock::now();

//   std::vector<cv::Point2f> pointset1;
//   std::vector<cv::Point2f> pointset2;

//   for (int i = 0; i < (int)initial_matches.size(); i++) { // initial keypoint matching result
//       pointset1.push_back(frameID_1.keypoints[initial_matches[i].queryIdx].pt); // coord of kp in frame 1
//       pointset2.push_back(frameID_2.keypoints[initial_matches[i].trainIdx].pt); // coord of kp in frame 2
//   }

//   cv::Mat k_int;
//   cv::eigen2cv(frameID_1.k_int, k_int);
//   cv::Mat essential_matrix;
//   cv::Mat inlier_matches_indicator;
//   // 5 point methods with ransac
//   essential_matrix = cv::findEssentialMat(pointset1, pointset2, k_int,
//                                           cv::RANSAC, ransac_prob, ransac_thre, inlier_matches_indicator);

//   std::cout << "essential_matrix is " << std::endl
//             << essential_matrix << std::endl;
//   // pick inlier keypoint match pairs
//   for (int i = 0; i < (int)initial_matches.size(); i++) {
//     std::cout<<inlier_matches_indicator.at<bool>(0, i)<<std::endl;
//     if (inlier_matches_indicator.at<bool>(0, i) == 1) {
//       std::cout<<"##############################"<<std::endl;
//       inlier_matches.push_back(initial_matches[i]);
//     }
//   }
//   // obtain R,T from essential mat. Note that t is up-to-scale.
//   cv::Mat R;
//   cv::Mat t;

//   cv::recoverPose(essential_matrix, pointset1, pointset2, k_int, R, t, inlier_matches_indicator);

//   std::chrono::steady_clock::time_point toc = std::chrono::steady_clock::now();
//   std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(toc - tic);
//   std::cout << "Estimate Motion [2D-2D] cost = " << time_used.count() << " seconds. " << std::endl;
//   std::cout << "Find [" << inlier_matches.size() << "] inlier matches from [" << initial_matches.size() << "] total matches." << std::endl;
//   // copy cv-mat to eigen-mat
//   Eigen::Matrix3f R_eigen;
//   Eigen::Vector3f t_eigen;
//   cv::cv2eigen(R, R_eigen);
//   cv::cv2eigen(t, t_eigen);
//   Transform.block(0, 0, 3, 3) = R_eigen;
//   Transform.block(0, 3, 3, 1) = t_eigen;
//   Transform(3, 0) = 0;
//   Transform(3, 1) = 0;
//   Transform(3, 2) = 0;
//   Transform(3, 3) = 1;

//   std::cout << "Transform is " << std::endl
//             << Transform << std::endl;

//   if (show) {
//     cv::Mat ransac_match_image;
//     cv::namedWindow("RANSAC inlier matches", 0);
//     cv::drawMatches(frameID_1.img_rgb, frameID_1.keypoints, frameID_2.img_rgb, frameID_2.keypoints, inlier_matches, ransac_match_image);
//     cv::imshow("RANSAC inlier matches", ransac_match_image);
//     cv::waitKey(0);
//   }
// }


void MotionEstimator::EstimateP3P_RANSAC(Frame& cur_frame, PointCloud& cur_pc_3d,
    double ransac_prob, double ransac_thre , int iterationsCount, bool show) {
  // Estimating [R|T] from 3D/2D pairs. Notice DOF(R+t) == 6, which requires 3 points (2 equations for each)
  // to estimate Projection matrix P: x = PX. Then P is decomposed into [R|T].
  // tutorial: http://www.cs.cmu.edu/~16385/s15/lectures/Lecture17.pdf
  // todo: try a new solver (https://github.com/midjji/pnp) instead of cv::solvePnPRansac

  std::chrono::steady_clock::time_point tic = std::chrono::steady_clock::now();

  cv::Mat k_int;
  cv::eigen2cv(cur_frame.k_int, k_int);

}


void MotionEstimator::GetDepth(Frame& frameID_1, Frame& frameID_2,
                        Eigen::Matrix4f& Transform, std::vector<cv::DMatch>& matches,
                        double& appro_depth, int random_rate) {
  // Estimate mean average depth of obtained 3d point cloud subset.
  cv::Mat T1_mat, T2_mat, k_int;
  Eigen::Matrix4f Tbasis = Eigen::Matrix4f::Identity();
  Eigen::Matrix<float, 3, 4> T1 = Tbasis.block(0, 0, 3, 4);
  Eigen::Matrix<float, 3, 4> T2 = (Transform * Tbasis).block(0, 0, 3, 4);
  std::vector<cv::Point2f> pointset1;
  std::vector<cv::Point2f> pointset2;

  cv::eigen2cv(T1, T1_mat);
  cv::eigen2cv(T2, T2_mat);
  cv::eigen2cv(frameID_1.k_int, k_int);

  // select partial matched keypoint pairs to boost depth estimation.
  for (unsigned int i = 0; i < matches.size(); i++) {
    if (i % random_rate == 0) {
      cv::Point2f pt1 = pixel2cam(frameID_1.keypoints[matches[i].queryIdx].pt, k_int);
      cv::Point2f pt2 = pixel2cam(frameID_2.keypoints[matches[i].trainIdx].pt, k_int);
      pointset1.push_back(pt1);
      pointset2.push_back(pt2);
    }
  }

  cv::Mat pts_3d;  // 3d point from triangulariont
  // get 3d pointsets pts_3d
  if (!pointset1.empty())
    Triangulation(T1_mat, T2_mat, pointset1, pointset2, pts_3d);
  // obtain mean relative depth
  double mean_depth = 0;
  for (unsigned int i = 0; i < pts_3d.cols; i++) {
    cv::Mat pt_3d = pts_3d.col(i);
    pt_3d /= pt_3d.at<float>(3, 0);  // normalize
    Eigen::Vector3f pt_vec;
    pt_vec(0) = pt_3d.at<float>(0, 0);
    pt_vec(1) = pt_3d.at<float>(1, 0);
    pt_vec(2) = pt_3d.at<float>(2, 0);

    mean_depth += pt_vec.norm();
  }

  appro_depth = mean_depth / pts_3d.cols;

  std::cout << "Mean relative depth from GetDpeth() is " << appro_depth << " * baseline length. " << std::endl;
}










