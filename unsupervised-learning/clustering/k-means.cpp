#include "../../utils/image.hpp"

int main(){
    std::string img_path = "../../data/cloudy_mountains.jpg";
    // std::string img_path = "../../data/rocking_horse.jpg";

    cv::Mat img = read_image(img_path);
    show_image(img);

    return 0;
}