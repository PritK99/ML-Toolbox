#ifndef IMAGE_HPP
#define IMAGE_HPP

#include <opencv2/opencv.hpp>
#include <iostream>

cv::Mat read_image(const std::string &img_path);
void show_image(cv::Mat &img);

#endif