/*********************************************************************************
 *      Project: TensorRT-Alpha-ROS                                              *
 *       Author: @FeiYull (https://github.com/FeiYull)                           *
 *     Modified: Xinran Wei (weixr0605@sina.com)                                 *
 *  Modified on: Apr 27, 2023                                                    *
 *                                                                               *
 *  Copyright (c) 2023, Xinran Wei.                                              *
 *  This code file along with the project is published under MIT LICENCE.        *
 *********************************************************************************/

#ifndef TENSORRT_ALPHA_ROS_LIBFACEDETECTION_H
#define TENSORRT_ALPHA_ROS_LIBFACEDETECTION_H


#include "utils/common_include.h"
#include "utils/utils.h"
#include "utils/kernel_function.h"

namespace TRTAROS{

class LibFaceDet
{
public:
    LibFaceDet(const utils::InitParameter& param);
    ~LibFaceDet();

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

private:
    std::shared_ptr<nvinfer1::ICudaEngine> m_engine;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context;

    //private:
protected:
    utils::InitParameter m_param;
    nvinfer1::Dims m_output_loc_dims;  // 18984*14
    nvinfer1::Dims m_output_conf_dims; // 18984*2
    nvinfer1::Dims m_output_iou_dims;  // 18984*1
    int m_total_objects;  // 18984

    // const params on host 
    const float  m_min_sizes_host[4 * 3] = 
    { 10, 16, 24,  32, 48, FLT_MAX,  64, 96, FLT_MAX,  128, 192, 256 };
    const int m_min_sizes_host_dim[4] = 
    { 3, 2, 2, 3 };
    float* m_feat_hw_host;     // 4 * 3
    float* m_prior_boxes_host;  // 18984 * 4
    const float m_variances_host[2] = { 0.1f, 0.2f };     // 2 * 1

    
    // const params on device
    float* m_min_sizes_device;    // 4 * 3
    float* m_feat_hw_host_device; // 4 * 3
    float* m_prior_boxes_device;  // 18984 * 4
    float* m_variances_device;    // 2 * 1

    std::vector<std::vector<utils::Box>> m_objectss;

    // input
    float* m_input_src_device;
    float* m_input_hwc_device;

    // output
    float* m_output_loc_device;
    float* m_output_conf_device;
    float* m_output_iou_device;
    float* m_output_objects_device;
    float* m_output_objects_host;
    int m_output_objects_width; // 7:left, top, right, bottom, confidence, class, keepflag; 

};

void decodeLibFaceDetDevice(float* minSizes, float* feat_hw, float* priorBoxes, float* variances,
    int srcImgWidth, int srcImgHeight,
    float confThreshold, int batchSize, int srcHeight,
    float* srcLoc, int srcLocWidth,
    float* srcConf, int srcConfWidth,
    float* srcIou, int srcIouWidth,
    float* dst, int dstWidth, int dstHeight);

}

#endif // TENSORRT_ALPHA_ROS_LIBFACEDETECTION_H