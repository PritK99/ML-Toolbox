#include <iostream>
#include "image.hpp"

cv::Mat read_image(const std::string &img_path){
    cv::Mat img = cv::imread(img_path);

    if (img.empty()){
        std::cout << "Image not found." << std::endl;
        return cv::Mat(100, 100, CV_8UC1);    // This is an dummy (100, 100, 1) image
    }

    return img;
}

void show_image(cv::Mat& img){
    cv::imshow("Image", img);
    
    while (true) {
        int key = cv::waitKey(30);
        if (key == 27){
            break;    // This is ESC key
        }

        if (cv::getWindowProperty("Image", cv::WND_PROP_VISIBLE) < 1) {
            break;
        }
    }

    cv::destroyAllWindows();
}

// // This is for testing functions
// int main(){
//     std::string img_path = "../data/cloudy_mountains.jpg";

//     cv::Mat img = read_image(img_path);
//     show_image(img);

//     return 0;
// }