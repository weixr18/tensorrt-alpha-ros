/*********************************************************************************
 *      Project: TensorRT-Alpha-ROS                                              *
 *       Author: Xinran Wei (weixr0605@sina.com)                                 *
 *  Modified on: Apr 27, 2023                                                    *
 *                                                                               *
 *  Copyright (c) 2023, Xinran Wei.                                              *
 *  This code file along with the project is published under MIT LICENCE.        *
 *********************************************************************************/

#include <string>
#include <string.h>
#include <exception>
#include <opencv2/opencv.hpp>
#include <camera_info_manager/camera_info_manager.h>

#include "tensorrt_alpha_ros.h"

bool is_str_equal(std::string a, std::string b){
    if(a.size() != b.size()){
        return false;
    }
    int res = strcmp(a.c_str(), b.c_str());
    return (res == 0);
}

void TensorRTAlphaROS::getNetworkObj(){
    if(is_str_equal(model_class, "YOLOR")){
        std::shared_ptr<TRTAROS::YOLOR> ptr(new TRTAROS::YOLOR(param));
        this->pNet = ptr;
    }
    else if(is_str_equal(model_class, "YOLOV4")){
        std::shared_ptr<TRTAROS::YOLOV4> ptr(new TRTAROS::YOLOV4(param));
        this->pNet = ptr;
    }
    else if(is_str_equal(model_class, "YOLOV5")){
        std::shared_ptr<TRTAROS::YOLOV5> ptr(new TRTAROS::YOLOV5(param));
        this->pNet = ptr;
    }
    else if(is_str_equal(model_class, "YOLOV7")){
        std::shared_ptr<TRTAROS::YOLOV7> ptr(new TRTAROS::YOLOV7(param));
        this->pNet = ptr;
    }
    else if(is_str_equal(model_class, "YOLOV8")){
        std::shared_ptr<TRTAROS::YOLOV8> ptr(new TRTAROS::YOLOV8(param));
        this->pNet = ptr;
    }
    else{
        this->pNet = nullptr;
    }
}
    

void TensorRTAlphaROS::modelParamInit(){
    if(is_str_equal(model_class, "YOLOR")){
        TRTAROS::YOLOR::setParameters(param);
    }
    else if(is_str_equal(model_class, "YOLOV4")){
        TRTAROS::YOLOV4::setParameters(param);
    }
    else if(is_str_equal(model_class, "YOLOV5")){
        TRTAROS::YOLOV5::setParameters(param);
    }
    else if(is_str_equal(model_class, "YOLOV7")){
        TRTAROS::YOLOV7::setParameters(param);
    }
    else if(is_str_equal(model_class, "YOLOV8")){
        TRTAROS::YOLOV8::setParameters(param);
    }
}

void TensorRTAlphaROS::getParams(ros::NodeHandle& nh){

    // model_class 
    if(!nh.getParam("model_class", model_class)) {
        sample::gLogError << "Parameter *model_class* is requierd." << std::endl;
        sample::gLogError << "Supported: [YOLOR|YOLOV4|YOLOV5|YOLOV7|YOLOV8]" << std::endl;
        throw std::exception();
    }
    sample::gLogInfo << "model_class: " << model_class << std::endl;
    modelParamInit(); // default init param values

    // .trt file path
    if(!nh.getParam("engine_file", trt_model_path)) {
        sample::gLogError << "Parameter *trt_model_path* is requierd." << std::endl;
        throw std::exception();
    }
    param.trt_model_path = trt_model_path;
    sample::gLogInfo << "trt_model_path: " << trt_model_path << std::endl;
    

    // input size of the neural network
    int nn_input_h = 640;
    nh.getParam("nn_input_h", nn_input_h);
    sample::gLogInfo << "nn_input_h: " << nn_input_h << std::endl;
    param.dst_h = nn_input_h;
    int nn_input_w = 640;
    nh.getParam("nn_input_w", nn_input_w);
    sample::gLogInfo << "nn_input_w: " << nn_input_w << std::endl;
    param.dst_w = nn_input_w;


    // batch size, default = 1
    nh.getParam("batch_size", batch_size);
    sample::gLogInfo << "batch_size: " << batch_size << std::endl;
    param.batch_size = batch_size;

    // show images
    nh.getParam("show", this->is_show);
    sample::gLogInfo << "is_show: " << this->is_show << std::endl;


    // input mode & source
    int input_mode = 0; // 0 for camera, 1 for video, 2 for image
    nh.getParam("input_mode", input_mode);
    if (input_mode == 0){
        sample::gLogInfo << "input_mode: Camera" << std::endl;
        source = TRTAROS::utils::InputStream::CAMERA;

        // topic
        if(!nh.getParam("cam_topic", camera_topic)) {
            sample::gLogError << "In Camera Mode, Parameter *cam_topic* is requierd." << std::endl;
            throw std::exception();
        }
        sample::gLogInfo << "camera_topic: " << camera_topic << std::endl;

        // size
        int cam_input_w;
        int cam_input_h;
        if(!nh.getParam("cam_input_w", cam_input_w)) {
            sample::gLogError << "In Camera Mode, Parameter *cam_input_w* is requierd." << std::endl;
            throw std::exception();
        }
        if(!nh.getParam("cam_input_h", cam_input_h)) {
            sample::gLogError << "In Camera Mode, Parameter *cam_input_h* is requierd." << std::endl;
            throw std::exception();
        }
        sample::gLogInfo << "cam input size: (" << cam_input_w << ", " << cam_input_h << ")\n" ;
        param.src_w = cam_input_w;
        param.src_h = cam_input_h;
    }
    else if (input_mode == 1){
        sample::gLogInfo << "input_mode: Video" << std::endl;
        source = TRTAROS::utils::InputStream::VIDEO;
        if(!nh.getParam("video", video_path)) {
            sample::gLogError << "In Video Mode, Parameter *video* is requierd." << std::endl;
            throw std::exception();
        }
        sample::gLogInfo << "video_path: " << video_path << std::endl;
    }
    else if (input_mode == 2){
        sample::gLogInfo << "input_mode: Image" << std::endl;
        source = TRTAROS::utils::InputStream::IMAGE;
        if(!nh.getParam("image", image_path)) {
            sample::gLogError << "In Image Mode, Parameter *image* is requierd." << std::endl;
            throw std::exception();
        }
        sample::gLogInfo << "image_path: " << image_path << std::endl;
    }
    else{
        sample::gLogError << "Fatal error: *inpot_mode* should be 0(camera), 1(video) or 2(image)." << std::endl;
        throw std::exception();
    }

}


void TensorRTAlphaROS::setPublisher(ros::NodeHandle& nh){
    // ROS publish
    if (this->showImage()){
        return; // show pic on screen, no ROS topic.
    }   
    this->pDetImgPub = new ros::Publisher(nh.advertise<sensor_msgs::Image>("detect_image", 1));
}


void TensorRTAlphaROS::init(ros::NodeHandle& nh){
        
    // 1. read params
    this->getParams(nh);

    // 2. set input stream
    if (!this->isCameraMode()){    
        bool res = TRTAROS::utils::setInputStream(
            source, image_path, video_path, 0,
            capture, total_batches, delay_time, param);
        if (!res) {
            sample::gLogError << "Error: cannot read the input data!" << std::endl;
            throw std::exception();
        }
    }

    // 3. load trt engine
    std::vector<unsigned char> trt_file = TRTAROS::utils::loadModel(trt_model_path);
    if (trt_file.empty())
    {
        sample::gLogError << "Fatal Error: trt_file is empty!" << std::endl;
        throw std::exception();
    }

    // 4. init model
    this->getNetworkObj();
    if(this->pNet == nullptr){
        sample::gLogError << "Error: false class type." << std::endl;
        throw std::exception();
    }
    if (!this->pNet->init(trt_file))
    {
        sample::gLogError << "Network init error." << std::endl;
        throw std::exception();
    }
    // 5. check model
    this->pNet->check();

    // 6. set publisher
    this->setPublisher(nh);
}


void TensorRTAlphaROS::processData(){
    // process video or image data
    cv::Mat frame;
    std::vector<cv::Mat> imgs_batch;
    imgs_batch.reserve(param.batch_size);
    sample::gLogInfo << "imgs batch capacity: " << imgs_batch.capacity() << std::endl;
    int batchi = 0;

    while (capture.isOpened())
    {
        if (batchi >= total_batches && source != TRTAROS::utils::InputStream::CAMERA) {
            break;
        }
        if (imgs_batch.size() < param.batch_size) // get input
        {
            if (source != TRTAROS::utils::InputStream::IMAGE) {
                capture.read(frame);
            }
            else {
                frame = cv::imread(image_path);
            }

            if (frame.empty()) {
                sample::gLogWarning << "No more video or camera frame." << std::endl;
                this->pNet->task(param, imgs_batch, delay_time, batchi, is_show, is_save);
                imgs_batch.clear(); // clear
                batchi++;
                break;
            }
            else {
                imgs_batch.emplace_back(frame.clone());
            }
        }
        else {
            // infer
            if(batchi == total_batches - 1){
                delay_time = 1000;
            }
            this->pNet->task(param, imgs_batch, delay_time, batchi, is_show, is_save);
            imgs_batch.clear(); // clear
            batchi++;
        }
    }
    sample::gLogInfo <<  "Done." << std::endl;
}


void TensorRTAlphaROS::inferenceImage(cv::Mat img){
    ros_batchi++;
    std::vector<cv::Mat> imgs_batch;
    imgs_batch.push_back(img);
    this->pNet->task(param, imgs_batch, delay_time, ros_batchi, is_show, false);
    if(!is_show) {
        // do not show, but publish images

        // check
        if(pDetImgPub == nullptr){
            return;
        }

        // make message
        cv::Mat& img_det = imgs_batch[0]; // channels: BGR
        int img_size = img_det.cols * img_det.rows * img_det.channels();
        img_det_ssm.data.resize(img_size);
        memcpy(&img_det_ssm.data[0], img_det.data, img_size);
        img_det_ssm.encoding = sensor_msgs::image_encodings::BGR8;
        img_det_ssm.height = img_det.rows;
        img_det_ssm.width = img_det.cols;
        img_det_ssm.step = img_det.cols * 3;
        img_det_ssm.header.frame_id = ros_batchi;
        img_det_ssm.header.stamp = ros::Time::now();

        // publish message
        pDetImgPub->publish(img_det_ssm);
    }
}
