/*********************************************************************************
 *      Project: TensorRT-Alpha-ROS                                              *
 *       Author: @FeiYull (https://github.com/FeiYull)                           *
 *     Modified: Xinran Wei (weixr0605@sina.com)                                 *
 *  Modified on: Apr 27, 2023                                                    *
 *                                                                               *
 *  Copyright (c) 2023, Xinran Wei.                                              *
 *  This code file along with the project is published under MIT LICENCE.        *
 *********************************************************************************/

#include "yolo/yolov7.h"

namespace TRTAROS {


YOLOV7::YOLOV7(const utils::InitParameter& param) :YOLO(param)
{
}

YOLOV7::~YOLOV7()
{
}

void YOLOV7::setParameters(utils::InitParameter& initParameters)
{
    initParameters.class_names = utils::dataSets::coco80;
    initParameters.num_class = 80; // for coco

    initParameters.batch_size = 8;
    initParameters.dst_h = 640;
    initParameters.dst_w = 640;

    /*initParameters.dst_h = 1280;
    initParameters.dst_w = 1280;*/

    initParameters.input_output_names = { "images",  "output" };

    initParameters.conf_thresh = 0.25f;
    initParameters.iou_thresh = 0.45f;

    //initParameters.conf_thresh = 0.1f;
    //initParameters.iou_thresh = 0.2f;
    initParameters.save_path = "";
}

void YOLOV7::task(const utils::InitParameter& param, std::vector<cv::Mat>& imgsBatch, 
    const int& delayTime, const int& batchi, const bool& isShow, const bool& isSave)
{
    this->copy(imgsBatch);
    utils::DeviceTimer d_t1; this->preprocess(imgsBatch);  
	float t1 = d_t1.getUsedTime();
    utils::DeviceTimer d_t2; this->infer();                  
	float t2 = d_t2.getUsedTime();
    utils::DeviceTimer d_t3; this->postprocess(imgsBatch); 
	float t3 = d_t3.getUsedTime();
	
    sample::gLogInfo << "preprocess time = " << t1 / param.batch_size << "; "
        "infer time = " << t2 / param.batch_size << "; "
        "postprocess time = " << t3 / param.batch_size << std::endl;
    if (isShow) {
        utils::show(this->getObjectss(), param.class_names, delayTime, imgsBatch);
    }
    else {
        utils::drawBboxes(this->getObjectss(), m_param.class_names, imgsBatch);
    }
    if (isSave) {
        utils::save(this->getObjectss(), param.class_names, param.save_path, imgsBatch, param.batch_size, batchi);
    }
    this->reset();
}
}
