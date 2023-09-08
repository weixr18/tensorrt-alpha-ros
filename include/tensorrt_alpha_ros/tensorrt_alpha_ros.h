/*********************************************************************************
 *      Project: TensorRT-Alpha-ROS                                              *
 *       Author: Xinran Wei (weixr0605@sina.com)                                 *
 *  Modified on: Apr 27, 2023                                                    *
 *                                                                               *
 *  Copyright (c) 2023, Xinran Wei.                                              *
 *  This code file along with the project is published under MIT LICENCE.        *
 *********************************************************************************/

#ifndef TENSORRT_ALPHA_ROS_TENSORRT_ALPHA_ROS_H
#define TENSORRT_ALPHA_ROS_TENSORRT_ALPHA_ROS_H

#include <ros/ros.h>
#include <sensor_msgs/Image.h>

#include "yolo/yolor.h"
#include "yolo/yolov4.h"
#include "yolo/yolov5.h"
#include "yolo/yolov7.h"
#include "yolo/yolov8.h"


class TensorRTAlphaROS {
public:
    TensorRTAlphaROS() {}
    ~TensorRTAlphaROS() {
    }
    void getNetworkObj();
    void modelParamInit();
    void getParams(ros::NodeHandle& nh);
    void init(ros::NodeHandle& nh);
    void setPublisher(ros::NodeHandle& nh);

    void processData(); // video or image
    void inferenceImage(cv::Mat img);

    bool isCameraMode() {
        return source == TRTAROS::utils::InputStream::CAMERA;
    }
    const std::string& getImageTopic(){
        return camera_topic;
    }
    const std::string& getModelClass(){
        return model_class;
    }
    const bool showImage(){
        return is_show;
    }


private:
    std::string model_class;
    TRTAROS::utils::InitParameter param;

    TRTAROS::utils::InputStream source;
    cv::VideoCapture capture;
    int total_batches = 0;
    std::string trt_model_path;
    std::string camera_topic;
    std::string image_path;
	std::string video_path;

	int batch_size = 8;
    int delay_time = 1;
    bool is_show = false;
	bool is_save = false;

    int ros_batchi = 0;

    std::shared_ptr<TRTAROS::Network> pNet;
    ros::Publisher* pDetImgPub = nullptr;
    sensor_msgs::Image img_det_ssm;
};

void imgCallBack(const sensor_msgs::ImageConstPtr &img_msg);

#endif // TENSORRT_ALPHA_ROS_TENSORRT_ALPHA_ROS_H