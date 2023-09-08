/*********************************************************************************
 *      Project: TensorRT-Alpha-ROS                                              *
 *       Author: @FeiYull (https://github.com/FeiYull)                           *
 *     Modified: Xinran Wei (weixr0605@sina.com)                                 *
 *  Modified on: Apr 27, 2023                                                    *
 *                                                                               *
 *  Copyright (c) 2023, Xinran Wei.                                              *
 *  This code file along with the project is published under MIT LICENCE.        *
 *********************************************************************************/

#ifndef TENSORRT_ALPHA_ROS_NETWORK_H
#define TENSORRT_ALPHA_ROS_NETWORK_H

#include "utils/common_include.h"
#include "utils/utils.h"

namespace TRTAROS{

class Network
{
public:
    Network() {}
    Network(const utils::InitParameter& param) {}
    ~Network() {}

public:
    virtual bool init(const std::vector<unsigned char>& trtFile) = 0;
    virtual void check() = 0;
    virtual void copy(const std::vector<cv::Mat>& imgsBatch) = 0;
    virtual void preprocess(const std::vector<cv::Mat>& imgsBatch) = 0;
    virtual bool infer() = 0;
    virtual void postprocess(const std::vector<cv::Mat>& imgsBatch) = 0;
    virtual void reset() = 0;
    virtual void drawInfo(std::vector<cv::Mat>& imgsBatch) = 0;

    static void setParameters(utils::InitParameter& initParameters) {}
    virtual void task(const utils::InitParameter& param, std::vector<cv::Mat>& imgsBatch, 
        const int& delayTime, const int& batchi, const bool& isShow, const bool& isSave) = 0;
protected:
    utils::InitParameter m_param;
};


} // namespace TRTAROS



#endif // TENSORRT_ALPHA_ROS_NETWORK_H