/*********************************************************************************
 *      Project: TensorRT-Alpha-ROS                                              *
 *       Author: Xinran Wei (weixr0605@sina.com)                                 *
 *  Modified on: Apr 27, 2023                                                    *
 *                                                                               *
 *  Copyright (c) 2023, Xinran Wei.                                              *
 *  This code file along with the project is published under MIT LICENCE.        *
 *********************************************************************************/

#include "tensorrt_alpha_ros.h"
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>

std::unique_ptr<TensorRTAlphaROS> ptr_sys;

void imgCallBack(const sensor_msgs::ImageConstPtr &img_msg){

	// get image
	cv_bridge::CvImageConstPtr ptr;
    if (img_msg->encoding == "8UC1")
    {
        sensor_msgs::Image img;
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    }
    else{
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);
    }
    cv::Mat image = ptr->image.clone();
    cv::Mat image_bgr;
    cv::cvtColor(image, image_bgr, cv::COLOR_GRAY2BGR);

    // inference & show
    ptr_sys->inferenceImage(image_bgr);
}


int main(int argc, char** argv)
{
    ros::init(argc, argv, "tensorrt_alpha_node");
    ros::NodeHandle nh("tensorrt_alpha_node"); // set up the node

    // system init
    ptr_sys.reset(new TensorRTAlphaROS());
    ptr_sys->init(nh);

    if (ptr_sys->isCameraMode()) {    
        // ROS subscribe 
        std::string image_topic = ptr_sys->getImageTopic();
        ros::Subscriber sub_img = nh.subscribe(image_topic, 100, &imgCallBack);
        ros::spin();
    }
    else {
        ptr_sys->processData();
    }

    return 0;
}