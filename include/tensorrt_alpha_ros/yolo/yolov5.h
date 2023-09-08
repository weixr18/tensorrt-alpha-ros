/*********************************************************************************
 *      Project: TensorRT-Alpha-ROS                                              *
 *       Author: @FeiYull (https://github.com/FeiYull)                           *
 *     Modified: Xinran Wei (weixr0605@sina.com)                                 *
 *  Modified on: Apr 27, 2023                                                    *
 *                                                                               *
 *  Copyright (c) 2023, Xinran Wei.                                              *
 *  This code file along with the project is published under MIT LICENCE.        *
 *********************************************************************************/

#ifndef TENSORRT_ALPHA_ROS_YOLOV5_H
#define TENSORRT_ALPHA_ROS_YOLOV5_H


#include "yolo.h"
#include "utils/utils.h"

namespace TRTAROS{

class YOLOV5 : public YOLO
{
public:
	YOLOV5(const utils::InitParameter& param);
	~YOLOV5();
	static void setParameters(utils::InitParameter& initParameters);
    virtual void task(const utils::InitParameter& param, std::vector<cv::Mat>& imgsBatch, 
        const int& delayTime, const int& batchi, const bool& isShow, const bool& isSave);
};

}

#endif // TENSORRT_ALPHA_ROS_YOLOV5_H