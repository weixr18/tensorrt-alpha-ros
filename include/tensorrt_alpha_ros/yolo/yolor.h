/*********************************************************************************
 *      Project: TensorRT-Alpha-ROS                                              *
 *       Author: @FeiYull (https://github.com/FeiYull)                           *
 *     Modified: Xinran Wei (weixr0605@sina.com)                                 *
 *  Modified on: Apr 27, 2023                                                    *
 *                                                                               *
 *  Copyright (c) 2023, Xinran Wei.                                              *
 *  This code file along with the project is published under MIT LICENCE.        *
 *********************************************************************************/

#ifndef TENSORRT_ALPHA_ROS_YOLOR_H
#define TENSORRT_ALPHA_ROS_YOLOR_H


#include "yolo.h"
#include "utils/kernel_function.h"

namespace TRTAROS{

class YOLOR : public YOLO
{
public:
	YOLOR(const utils::InitParameter& param);
	~YOLOR();
	static void setParameters(utils::InitParameter& initParameters);
    virtual void task(const utils::InitParameter& param, std::vector<cv::Mat>& imgsBatch, 
        const int& delayTime, const int& batchi, const bool& isShow, const bool& isSave);

private:
	float* m_input_resize_without_padding_device;
	int m_resized_w;
	int m_resized_h;
};

void copyWithPaddingDevice(const int& batchSize, float* src, int srcWidth, int srcHeight,
	float* dst, int dstWidth, int dstHeight, float paddingValue);

}

#endif // TENSORRT_ALPHA_ROS_YOLOR_H
