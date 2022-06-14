// Copyright 2022 Zzy
// Author: Zzy-buffer
// Feature class for SFM
/* Version 0.1 @ 2022
*/


#ifndef _feat_pcs
#define _feat_pcs

#include "frame.hpp"


namespace sfm
{

class Feature {
  public:
    void DetectFeatureSIFT(Frame& cur_frame, bool show = false);
    void DetectFeatureSURF(Frame& cur_frame, int minHessian = 350, bool show = false);
    void DetectFeatureORB(Frame& cur_frame, int max_num = 5000, bool show = false);

    void Match2viewSIFT(Frame& frameID_1, Frame& frameID_2,
      std::vector<cv::DMatch> &matches_buffer, double dist_ratio = 0.6, bool show = false);
    void Match2viewSIFTBidirectional(Frame& frameID_1, Frame& frameID_2,
      std::vector<cv::DMatch> &matches_buffer, double dist_ratio = 0.6, bool show = false);
    void Match2viewSURF(Frame& frameID_1, Frame& frameID_2, std::vector<cv::DMatch>& matches_buffer,
                         double dist_ratio = 0.5, bool show = false);
    void Match2viewORB(Frame& frameID_1, Frame& frameID_2, std::vector<cv::DMatch>& matches_buffer,
                         double dist_ratio = 0.5, bool show = false);

};




}
#endif