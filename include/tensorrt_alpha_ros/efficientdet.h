/*********************************************************************************
 *      Project: TensorRT-Alpha-ROS                                              *
 *       Author: @FeiYull (https://github.com/FeiYull)                           *
 *     Modified: Xinran Wei (weixr0605@sina.com)                                 *
 *  Modified on: Apr 27, 2023                                                    *
 *                                                                               *
 *  Copyright (c) 2023, Xinran Wei.                                              *
 *  This code file along with the project is published under MIT LICENCE.        *
 *********************************************************************************/

#ifndef TENSORRT_ALPHA_ROS_EFFICIENTDET_H
#define TENSORRT_ALPHA_ROS_EFFICIENTDET_H


#include "utils/common_include.h"
#include "utils/utils.h"
#include "utils/kernel_function.h"

namespace TRTAROS{

class EfficientDet
{
public:
    EfficientDet(const utils::InitParameter& param);
    ~EfficientDet();

public:
    bool init(const std::vector<unsigned char>& trtFile);
    void check();
    void copy(const std::vector<cv::Mat>& imgsBatch);
    void preprocess(const std::vector<cv::Mat>& imgsBatch);
    bool infer();
    void postprocess(const std::vector<cv::Mat>& imgsBatch);
    void reset();

public:
    std::vector<std::vector<utils::Box>> getObjectss() const;

protected:
    std::shared_ptr<nvinfer1::ICudaEngine> m_engine;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context;

protected:
    utils::InitParameter m_param;
    std::vector<std::vector<utils::Box>> m_objectss;

    // (m_param.dst_h, m_param.dst_w) to (m_param.src_h, m_param.src_w) 
    utils::AffineMat m_dst2src; // 2*3

    // input
    float* m_input_src_device;
    float* m_input_resize_device;
    float* m_input_rgb_device;

    // output
    int* m_output_num_device;      // b * 1
    int* m_output_boxes_device;    // b * 1 * 100 * 4
    int* m_output_scores_device;   // b * 1 * 100
    int* m_output_classes_device;  // b * 1 * 100
    
    int* m_output_num_host;        // b * 1
    int* m_output_boxes_host;      // b * 1 * 100 * 4
    int* m_output_scores_host;     // b * 1 * 100
    int* m_output_classes_host;    // b * 1 * 100
};

}

#endif // TENSORRT_ALPHA_ROS_EFFICIENTDET_H