// Copyright 2022 Zzy
// Author: Zzy-buffer
/* C++ Library for Computer Vision
updated: 06/2022
*/

#ifndef _my_package
#define _my_package


#include <iostream>
#include <vector>
#include <fstream>
#include <string>

#include <opencv2/opencv.hpp>

template <typename T>
void show_img(const T& img) {
  cv::imshow("Display window", img);
  cv::waitKey(0); // Wait for a keystroke in the window
  return;
}









#endif