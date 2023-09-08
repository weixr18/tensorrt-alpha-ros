/*********************************************************************************
 *      Project: TensorRT-Alpha-ROS                                              *
 *       Author: @FeiYull (https://github.com/FeiYull)                           *
 *     Modified: Xinran Wei (weixr0605@sina.com)                                 *
 *  Modified on: Apr 27, 2023                                                    *
 *                                                                               *
 *  Copyright (c) 2023, Xinran Wei.                                              *
 *  This code file along with the project is published under MIT LICENCE.        *
 *********************************************************************************/

#ifndef TENSORRT_ALPHA_ROS_UFLDV2_H
#define TENSORRT_ALPHA_ROS_UFLDV2_H


#include "utils/common_include.h"
#include "utils/utils.h"
#include "utils/kernel_function.h"

namespace TRTAROS{

struct UFLD_Params : utils::InitParameter
{
    float crop_ratio = 0.6f;
    int resize_width;
    int resize_height;
};

class UFLDV2
{
public:
    UFLDV2(const UFLD_Params& param);
    ~UFLDV2();

public:
    bool init(const std::vector<unsigned char>& trtFile);
    void check();
    void copy(const std::vector<cv::Mat>& imgsBatch);
    void preprocess(const std::vector<cv::Mat>& imgsBatch);
    bool infer();
    void postprocess(const std::vector<cv::Mat>& imgsBatch);
    void reset();

protected:
    std::shared_ptr<nvinfer1::ICudaEngine> m_engine;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context;

protected:
    UFLD_Params m_param;
    nvinfer1::Dims m_output_dims;   
    int m_output_area;              
    int m_total_objects;            
    std::vector<std::vector<utils::Box>> m_objectss;

    // (m_param.dst_h, m_param.dst_w) to (m_param.src_h, m_param.src_w) 
    utils::AffineMat m_dst2src;     // 2*3

    // input
    float* m_input_src_device;
    float* m_input_resize_device;
    float* m_input_crop_device;
    float* m_input_norm_device;
    float* m_input_hwc_device;

    // output
    float* m_output_out0_device; // todo ������onnx��concatһ��!!!!!!!!!!!!!!
    float* m_output_out1_device;
    float* m_output_out2_device;
    float* m_output_out3_device;
};

namespace ufld
{
	// python:dst = src[yLow:yHigh, xLow:xHigh] -> [yLow, yHigh), [xLow, xHigh)
	void cropDevice(int batchSize, int xLow, int xHigh, int yLow, int yHigh,
		float* src, int srcWidth, int srcHeight, 
		float* dst, int dstWidth, int dstHeight);
}


}

#endif // TENSORRT_ALPHA_ROS_UFLDV2_H