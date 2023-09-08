/*********************************************************************************
 *      Project: TensorRT-Alpha-ROS                                              *
 *       Author: @FeiYull (https://github.com/FeiYull)                           *
 *     Modified: Xinran Wei (weixr0605@sina.com)                                 *
 *  Modified on: Apr 27, 2023                                                    *
 *                                                                               *
 *  Copyright (c) 2023, Xinran Wei.                                              *
 *  This code file along with the project is published under MIT LICENCE.        *
 *********************************************************************************/

#ifndef TENSORRT_ALPHA_ROS_YOLOV8_H
#define TENSORRT_ALPHA_ROS_YOLOV8_H

#include "yolo.h"
#include "utils/utils.h"

namespace TRTAROS {

class YOLOV8 : public YOLO
{
public:
	YOLOV8(const utils::InitParameter& param);
	~YOLOV8();
	virtual bool init(const std::vector<unsigned char>& trtFile);
	virtual void preprocess(const std::vector<cv::Mat>& imgsBatch);
	virtual void postprocess(const std::vector<cv::Mat>& imgsBatch);
	static void setParameters(utils::InitParameter& initParameters);
    virtual void task(const utils::InitParameter& param, std::vector<cv::Mat>& imgsBatch, 
        const int& delayTime, const int& batchi, const bool& isShow, const bool& isSave);

	void decodeDevice(utils::InitParameter param, float* src, int srcWidth, 
		int srcHeight, int srcArea, float* dst, int dstWidth, int dstHeight);
	void transposeDevice(utils::InitParameter param, 
		float* src, int srcWidth, int srcHeight, int srcArea, 
		float* dst, int dstWidth, int dstHeight);

private:
	float* m_output_src_transpose_device;
};

}

#endif // TENSORRT_ALPHA_ROS_YOLOV8_H