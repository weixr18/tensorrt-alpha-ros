/*********************************************************************************
 *      Project: TensorRT-Alpha-ROS                                              *
 *       Author: @FeiYull (https://github.com/FeiYull)                           *
 *     Modified: Xinran Wei (weixr0605@sina.com)                                 *
 *  Modified on: Apr 27, 2023                                                    *
 *                                                                               *
 *  Copyright (c) 2023, Xinran Wei.                                              *
 *  This code file along with the project is published under MIT LICENCE.        *
 *********************************************************************************/


#include "yolo/yolov8.h"

namespace TRTAROS {

YOLOV8::YOLOV8(const utils::InitParameter& param) :YOLO(param)
{
}

YOLOV8::~YOLOV8()
{
    checkRuntime(cudaFree(m_output_src_transpose_device));
}

bool YOLOV8::init(const std::vector<unsigned char>& trtFile)
{
    // 1. init engine & context
    if (trtFile.empty())
    {
        return false;
    }
    // runtime
    std::unique_ptr<nvinfer1::IRuntime> runtime =
        std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));
    if (runtime == nullptr)
    {
        return false;
    }
    // deserializeCudaEngine
    this->m_engine = std::unique_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(trtFile.data(), trtFile.size()));

    if (this->m_engine == nullptr)
    {
        return false;
    }
    // context
    this->m_context = std::unique_ptr<nvinfer1::IExecutionContext>(this->m_engine->createExecutionContext());
    if (this->m_context == nullptr)
    {
        return false;
    }
    // binding dim
    if (m_param.dynamic_batch) // for some models only support static dynamic batch. eg: yolox
    {
        this->m_context->setBindingDimensions(0, nvinfer1::Dims4(m_param.batch_size, 3, m_param.dst_h, m_param.dst_w));
    }

    // 2. get output's dim
    m_output_dims = this->m_context->getBindingDimensions(1);
    m_total_objects = m_output_dims.d[2];
    assert(m_param.batch_size <= m_output_dims.d[0]);
    m_output_area = 1; // 22500 * 85
    for (int i = 1; i < m_output_dims.nbDims; i++)
    {
        if (m_output_dims.d[i] != 0)
        {
            m_output_area *= m_output_dims.d[i];
        }
    }
    // 3. malloc
    checkRuntime(cudaMalloc(&m_output_src_device, m_param.batch_size * m_output_area * sizeof(float)));
    checkRuntime(cudaMalloc(&m_output_src_transpose_device, m_param.batch_size * m_output_area * sizeof(float)));
    // 4. cal affine matrix
    float a = float(m_param.dst_h) / m_param.src_h;
    float b = float(m_param.dst_w) / m_param.src_w;
    float scale = a < b ? a : b;
    cv::Mat src2dst = (cv::Mat_<float>(2, 3) << scale, 0.f, (-scale * m_param.src_w + m_param.dst_w + scale - 1) * 0.5,
        0.f, scale, (-scale * m_param.src_h + m_param.dst_h + scale - 1) * 0.5);
    cv::Mat dst2src = cv::Mat::zeros(2, 3, CV_32FC1);
    cv::invertAffineTransform(src2dst, dst2src);

    m_dst2src.v0 = dst2src.ptr<float>(0)[0];
    m_dst2src.v1 = dst2src.ptr<float>(0)[1];
    m_dst2src.v2 = dst2src.ptr<float>(0)[2];
    m_dst2src.v3 = dst2src.ptr<float>(1)[0];
    m_dst2src.v4 = dst2src.ptr<float>(1)[1];
    m_dst2src.v5 = dst2src.ptr<float>(1)[2];

    return true;
}

void YOLOV8::preprocess(const std::vector<cv::Mat>& imgsBatch)
{
    // 1.resize
    resizeDevice(m_param.batch_size, m_input_src_device, m_param.src_w, m_param.src_h,
        m_input_resize_device, m_param.dst_w, m_param.dst_h, 114, m_dst2src);

#if 0 // valid
    {
        float* phost = new float[3 * m_param.dst_h * m_param.dst_w];
        float* pdevice = m_input_resize_device;
        for (size_t j = 0; j < imgsBatch.size(); j++)
        {
            checkRuntime(cudaMemcpy(phost, pdevice + j * 3 * m_param.dst_h * m_param.dst_w,
                sizeof(float) * 3 * m_param.dst_h * m_param.dst_w, cudaMemcpyDeviceToHost));
            cv::Mat ret(m_param.dst_h, m_param.dst_w, CV_32FC3, phost);
            ret.convertTo(ret, CV_8UC3, 1.0, 0.0);
            cv::namedWindow("ret", cv::WINDOW_NORMAL);
            cv::imshow("ret", ret);
            cv::waitKey(1);
        }
        delete[] phost;
    }
#endif // 0

    // 2. bgr2rgb
    bgr2rgbDevice(m_param.batch_size, m_input_resize_device, m_param.dst_w, m_param.dst_h,
        m_input_rgb_device, m_param.dst_w, m_param.dst_h);

#if 0 // valid
    {
        float* phost = new float[3 * m_param.dst_h * m_param.dst_w];
        float* pdevice = m_input_rgb_device;
        for (size_t j = 0; j < imgsBatch.size(); j++)
        {
            checkRuntime(cudaMemcpy(phost, pdevice + j * 3 * m_param.dst_h * m_param.dst_w,
                sizeof(float) * 3 * m_param.dst_h * m_param.dst_w, cudaMemcpyDeviceToHost));
            cv::Mat ret(m_param.dst_h, m_param.dst_w, CV_32FC3, phost);
            ret.convertTo(ret, CV_8UC3, 1.0, 0.0);
            cv::namedWindow("ret", cv::WINDOW_NORMAL);
            cv::imshow("ret", ret);
            cv::waitKey(1);
        }
        delete[] phost;
    }
#endif // 0

    // 3. norm:scale mean std
    normDevice(m_param.batch_size, m_input_rgb_device, m_param.dst_w, m_param.dst_h,
        m_input_norm_device, m_param.dst_w, m_param.dst_h, m_param);

#if 0 // valid
    {
        float* phost = new float[3 * m_param.dst_h * m_param.dst_w];
        float* pdevice = m_input_norm_device;
        for (size_t j = 0; j < imgsBatch.size(); j++)
        {
            checkRuntime(cudaMemcpy(phost, pdevice + j * 3 * m_param.dst_h * m_param.dst_w,
                sizeof(float) * 3 * m_param.dst_h * m_param.dst_w, cudaMemcpyDeviceToHost));
            cv::Mat ret(m_param.dst_h, m_param.dst_w, CV_32FC3, phost);
            for (size_t y = 0; y < ret.rows; y++)
            {
                for (size_t x = 0; x < ret.cols; x++)
                {
                    for (size_t c = 0; c < 3; c++)
                    {
                        //
                        ret.at<cv::Vec3f>(y, x)[c]
                            = m_param.scale * (ret.at<cv::Vec3f>(y, x)[c] * m_param.stds[c] + m_param.means[c]);
                    }

                }
            }
            ret.convertTo(ret, CV_8UC3, 1.0, 0.0);
            cv::imshow("ret", ret);
            cv::waitKey(1);
        }
        delete[] phost;
    }
#endif // 0

    // 4. hwc2chw
    hwc2chwDevice(m_param.batch_size, m_input_norm_device, m_param.dst_w, m_param.dst_h,
        m_input_hwc_device, m_param.dst_w, m_param.dst_h);
#if 0
    {

        float* phost = new float[3 * m_param.dst_h * m_param.dst_w];
        float* pdevice = m_input_hwc_device;
        for (size_t j = 0; j < imgsBatch.size(); j++)
        {
            checkRuntime(cudaMemcpy(phost, pdevice + j * 3 * m_param.dst_h * m_param.dst_w,
                sizeof(float) * 3 * m_param.dst_h * m_param.dst_w, cudaMemcpyDeviceToHost));

            cv::Mat tmp = imgsBatch[j].clone();

            cv::Mat b(m_param.dst_h, m_param.dst_w, CV_32FC1, phost);
            cv::Mat g(m_param.dst_h, m_param.dst_w, CV_32FC1, phost + 1 * m_param.dst_h * m_param.dst_w);
            cv::Mat r(m_param.dst_h, m_param.dst_w, CV_32FC1, phost + 2 * m_param.dst_h * m_param.dst_w);
            std::vector<cv::Mat> bgr{ b, g, r };
            cv::Mat ret;
            cv::merge(bgr, ret);
            ret.convertTo(ret, CV_8UC3, 255, 0.0);
            cv::imshow("ret", ret);

            /* SYSTEMTIME st = { 0 };
             GetLocalTime(&st);
             std::string t = std::to_string(st.wHour) + std::to_string(st.wMinute) + std::to_string(st.wMilliseconds);
             std::string save_path = "F:/Data/temp/";;
             cv::imwrite(save_path + t + ".jpg", ret);*/
            cv::waitKey(1);

            cv::Mat img_ = imgsBatch[j].clone();
        }
        delete[] phost;

    }
#endif

}


void YOLOV8::postprocess(const std::vector<cv::Mat>& imgsBatch)
{
#if 0 // valid
    {
        float* phost = new float[m_param.batch_size * m_output_area];
        float* pdevice = m_output_src_device;
        for (size_t j = 0; j < imgsBatch.size(); j++)
        {
            checkRuntime(cudaMemcpy(phost, pdevice + j * m_output_area, sizeof(float) * m_output_area, cudaMemcpyDeviceToHost));
            //cv::Mat prediction(m_total_objects, m_param.num_class + 4, CV_32FC1, phost);
            cv::Mat prediction(m_param.num_class + 4, m_total_objects, CV_32FC1, phost); // for yolov8
             // write bin
            utils::saveBinaryFile(phost, 84 * m_total_objects, "d:/yolov8n.bin");
        }
        delete[] phost;
    }
#endif // 0
    // transpose
    transposeDevice(m_param, m_output_src_device, m_total_objects, 4 + m_param.num_class, m_total_objects * (4 + m_param.num_class),
        m_output_src_transpose_device, 4 + m_param.num_class, m_total_objects);
#if 0 // valid
    {
        float* phost = new float[m_total_objects * (4 + m_param.num_class)];
        float* pdevice = m_output_src_transpose_device;
        for (size_t j = 0; j < imgsBatch.size(); j++)
        {
            checkRuntime(cudaMemcpy(phost, pdevice + j * m_total_objects * (4 + m_param.num_class),
                sizeof(float) * m_total_objects * (4 + m_param.num_class), cudaMemcpyDeviceToHost));
            cv::Mat img_transpose(m_total_objects, 4 + m_param.num_class, CV_32FC1, phost);
        }
        delete[] phost;
    }
#endif // 0
    // decode
    decodeDevice(m_param, m_output_src_transpose_device, 4 + m_param.num_class, m_total_objects, m_output_area,
        m_output_objects_device, m_output_objects_width, m_param.topK);
#if 0 // valid
    {
        float* phost = new float[1 + m_output_objects_width * m_param.topK];
        float* pdevice = m_output_objects_device;
        for (size_t j = 0; j < imgsBatch.size(); j++)
        {
            /* cv::imshow("srcimg", imgsBatch[j]);
             cv::waitKey(0);*/
            checkRuntime(cudaMemcpy(phost, pdevice + j * (1 + m_output_objects_width * m_param.topK),
                sizeof(float) * (1 + m_output_objects_width * m_param.topK), cudaMemcpyDeviceToHost));
            int num_candidates = phost[0];
            cv::Mat prediction(m_param.topK, m_output_objects_width, CV_32FC1, phost + 1);
        }
        delete[] phost;
    }
#endif // 0

    // nms
    //nmsDeviceV1(m_param, m_output_objects_device, m_output_objects_width, m_param.topK, m_param.topK * m_output_objects_width + 1);
    nmsDeviceV2(m_param, m_output_objects_device, m_output_objects_width, m_param.topK, m_param.topK * m_output_objects_width + 1, m_output_idx_device, m_output_conf_device);
#if 0 // valid
    {
        float* phost = new float[1 + m_output_objects_width * m_param.topK];
        float* pdevice = m_output_objects_device;
        for (size_t j = 0; j < imgsBatch.size(); j++)
        {
            /*cv::imshow("srcimg", imgsBatch[j]);
            cv::waitKey(0);*/
            checkRuntime(cudaMemcpy(phost, pdevice + j * (1 + m_output_objects_width * m_param.topK),
                sizeof(float) * (1 + m_output_objects_width * m_param.topK), cudaMemcpyDeviceToHost));
            int num_candidates_objects = phost[0];
            cv::Mat prediction(m_param.topK, m_output_objects_width, CV_32FC1, phost + 1);
        }
        delete[] phost;
    }
#endif // 0

    // copy result from gpu to cpu
    checkRuntime(cudaMemcpy(m_output_objects_host, m_output_objects_device, m_param.batch_size * sizeof(float) * (1 + 7 * m_param.topK), cudaMemcpyDeviceToHost));

    // transform to source image coordinate,
    for (size_t bi = 0; bi < imgsBatch.size(); bi++)
    {
        int num_boxes = std::min((int)(m_output_objects_host + bi * (m_param.topK * m_output_objects_width + 1))[0], m_param.topK);
        for (size_t i = 0; i < num_boxes; i++)
        {
            float* ptr = m_output_objects_host + bi * (m_param.topK * m_output_objects_width + 1) + m_output_objects_width * i + 1;
            int keep_flag = ptr[6];
            if (keep_flag)
            {
                // yolov35678
                float x_lt = m_dst2src.v0 * ptr[0] + m_dst2src.v1 * ptr[1] + m_dst2src.v2; // left & top
                float y_lt = m_dst2src.v3 * ptr[0] + m_dst2src.v4 * ptr[1] + m_dst2src.v5;
                float x_rb = m_dst2src.v0 * ptr[2] + m_dst2src.v1 * ptr[3] + m_dst2src.v2; // right & bottom
                float y_rb = m_dst2src.v3 * ptr[2] + m_dst2src.v4 * ptr[3] + m_dst2src.v5;
                // yolov4
                //float x_lt = m_dst2src.v0 * ptr[0] * m_param.dst_w + m_dst2src.v1 * ptr[1] * m_param.dst_h + m_dst2src.v2; // left & top
                //float y_lt = m_dst2src.v3 * ptr[0] * m_param.dst_w + m_dst2src.v4 * ptr[1] * m_param.dst_h + m_dst2src.v5;
                //float x_rb = m_dst2src.v0 * ptr[2] * m_param.dst_w + m_dst2src.v1 * ptr[3] * m_param.dst_h + m_dst2src.v2; // right & bottom
                //float y_rb = m_dst2src.v3 * ptr[2] * m_param.dst_w + m_dst2src.v4 * ptr[3] * m_param.dst_h + m_dst2src.v5;

                m_objectss[bi].emplace_back(x_lt, y_lt, x_rb, y_rb, ptr[4], (int)ptr[5]);
            }
        }

    }
}

void YOLOV8::setParameters(utils::InitParameter& initParameters)
{
	initParameters.class_names = utils::dataSets::coco80;
	//initParameters.class_names = utils::dataSets::voc20;
	initParameters.num_class = 80; // for coco
	//initParameters.num_class = 20; // for voc2012

	initParameters.batch_size = 8;
	initParameters.dst_h = 640;
	initParameters.dst_w = 640;
	initParameters.input_output_names = { "images",  "output0" };
	initParameters.conf_thresh = 0.25f;
	initParameters.iou_thresh = 0.45f;
	initParameters.save_path = "";
}

void YOLOV8::task(const utils::InitParameter& param, std::vector<cv::Mat>& imgsBatch, 
	const int& delayTime, const int& batchi, const bool& isShow, const bool& isSave)
{
	utils::DeviceTimer d_t0; this->copy(imgsBatch);	      float t0 = d_t0.getUsedTime();
	utils::DeviceTimer d_t1; this->preprocess(imgsBatch);  float t1 = d_t1.getUsedTime();
	utils::DeviceTimer d_t2; this->infer();				  float t2 = d_t2.getUsedTime();
	utils::DeviceTimer d_t3; this->postprocess(imgsBatch); float t3 = d_t3.getUsedTime();
	sample::gLogInfo << 
		//"copy time = " << t0 / param.batch_size << "; "
		"preprocess time = " << t1 / param.batch_size << "; "
		"infer time = " << t2 / param.batch_size << "; "
		"postprocess time = " << t3 / param.batch_size << std::endl;

	if(isShow)
		utils::show(this->getObjectss(), param.class_names, delayTime, imgsBatch);
	if(isSave)
		utils::save(this->getObjectss(), param.class_names, param.save_path, imgsBatch, param.batch_size, batchi);
	this->reset();
}

}