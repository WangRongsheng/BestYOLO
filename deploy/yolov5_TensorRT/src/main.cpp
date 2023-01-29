#include "../include/yolov5.h"

int main(int argc, char** argv)
{
    const std::string img_path = "../bus.jpg";
    cv::Mat image = cv::imread(img_path);
    if(image.empty()){
        std::cout<<"Input image path wrong!!"<<std::endl;
        return -1;
    }

    const std::string model_path = "../yolov5s_640x640.engine";
    Yolov5_detector* yolov5_instance = new Yolov5_detector(model_path);
    yolov5_instance->do_detection(image);

    std::string save_path = "./yolov5.jpg";
    cv::imwrite(save_path, image);

    if(yolov5_instance)
    {
        delete yolov5_instance;
        yolov5_instance = nullptr;
    }
    return 0;
}