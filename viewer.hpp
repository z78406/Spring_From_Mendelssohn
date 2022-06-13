#ifndef _viewer
#define _viewer


// local header
#include "frame.hpp"

namespace sfm
{

class MapViewer {

public:
  void DisplayFrame(const Frame& cur_frame, const std::string& window_name = "Inlier Matching viewer", int lasting_time = 20);
  void DisplayFrameMatch(const Frame& frameID_1, const Frame& frameID_2, const std::vector<cv::DMatch> &inlier_matches,
    const std::string& window_name = "Inlier Matching viewer", int lasting_time = 20);
  // void DisplaySFM();
};



}

#endif
