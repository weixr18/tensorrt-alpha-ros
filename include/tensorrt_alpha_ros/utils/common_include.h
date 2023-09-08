/*********************************************************************************
 *      Project: TensorRT-Alpha-ROS                                              *
 *       Author: @FeiYull (https://github.com/FeiYull)                           *
 *     Modified: Xinran Wei (weixr0605@sina.com)                                 *
 *  Modified on: Apr 27, 2023                                                    *
 *                                                                               *
 *  Copyright (c) 2023, Xinran Wei.                                              *
 *  This code file along with the project is published under MIT LICENCE.        *
 *********************************************************************************/

#ifndef TENSORRT_ALPHA_ROS_COMMON_INCLUDE_H
#define TENSORRT_ALPHA_ROS_COMMON_INCLUDE_H


// tensorrt
#include <NvInfer.h>
#include <logger.h>
#include <parserOnnxConfig.h>


// cuda
#include <cuda_runtime.h>
#include <thrust/sort.h>
// #include <thrust/device_vector.h>

// opencv
#include <opencv2/opencv.hpp>

// cpp std
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <chrono>
//#include <math.h>

//ros
#include <ros/ros.h>
#include <sensor_msgs/image_encodings.h>

#endif // TENSORRT_ALPHA_ROS_COMMON_INCLUDE_H