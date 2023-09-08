# TensorRT-Alpha-ROS

## Introducion
This repository is a ROS version of [TensorRT-Alhpa](https://github.com/FeiYull/tensorrt-alpha). It provides accelerated deployment cases of deep learning CV popular models, and cuda c supports of dynamic-batch image process, infer, decode, NMS on ROS.

With this repo, you can optimize your nn model(.onnx) via [TensorRT](https://github.com/onnx/onnx-tensorrt) and communicate with other ROS nodes. You can download some popular models directly from [@FeiYull](https://github.com/FeiYull/)'s network drives: [weiyun](https://share.weiyun.com/3T3mZKBm) or [google driver](https://drive.google.com/drive/folders/1-8phZHkx_Z274UVqgw6Ma-6u5AKmqCOv?usp=sharing).


## Acknowledgement

Thanks to [@FeiYull](https://github.com/FeiYull/)'s [TensorRT-Alhpa](https://github.com/FeiYull/tensorrt-alpha) project, on which most of the main code of this project was modified. This project has been open-sourced through the MIT protocol, and any comments and suggestions are welcome!


## Installation
The following environments have been tested：

+ Ubuntu 20.04 LTS
  + ROS noetic
  + cuda 11.4
  + cudnn 8.2.4
  + gcc 9.4.0
  + tensorrt 8.2.0 EA
  + opencv 3.x or 4.x
  + cmake 3.16.3


```bash
# install miniconda, ROS and TensorRT first
conda create -n tensorrt-alpha python==3.8 -y
conda activate tensorrt-alpha
sudo apt-get install python3-catkin-tools
mkdir ~/rt_catkin_ws && cd ~/rt_catkin_ws && mkdir src
cd src && catkin_init_workspace
git clone https://github.com/weixr18/tensorrt-alpha-ros
cd .. && pip install -r requirements.txt  
catkin make
```

## Quick Start

### Ubuntu 20.04

1. set your TensorRT_ROOT path and camera topic

```bash
cd tensorrt-alpha-ros/src/
vim CMakeLists.txt
# set var TensorRT_ROOT to your path in line 20, eg:
# set(TensorRT_ROOT /root/TensorRT-8.2.0.6)
cd launch
vim tensorrt_alpha.launch
# set param `cam_topic`, `cam_input_w` and `cam_input_h` to your own camera settings.
```

2. get and compile onnx to trt

See @FeiYull's documents. For example: [yolov7](https://github.com/FeiYull/tensorrt-alpha/tree/main/yolov7/README.md). You only need to follow **step 1-3**, then you'll get your .trt file. Bingo.

3. run

```bash
roslaunch tensorrt_alpha_ros tensorrt_alpha.launch
```

The result will show as a ROS image topic `/tensorrt_alpha_node/detect_image`. You can use `image_view` to se the real-time detection results.


## Onnx
At present, some of the models have been implemented, and some onnx files of them are organized as follows:

<div align='center'>

| model | tesla v100(32G) |weiyun |google driver |
  :-: | :-: | :-: | :-: |
|[yolov4](yolov4/README.md)| |[weiyun](https://share.weiyun.com/3T3mZKBm)| [google driver](https://drive.google.com/drive/folders/1-8phZHkx_Z274UVqgw6Ma-6u5AKmqCOv?usp=sharing)|
|[yolov5](yolov5/README.md)| |[weiyun](https://share.weiyun.com/3T3mZKBm)| [google driver](https://drive.google.com/drive/folders/1-8phZHkx_Z274UVqgw6Ma-6u5AKmqCOv?usp=sharing)|         
|[yolov7](yolov7/README.md)| |[weiyun](https://share.weiyun.com/3T3mZKBm)| [google driver](https://drive.google.com/drive/folders/1-8phZHkx_Z274UVqgw6Ma-6u5AKmqCOv?usp=sharing)|     
|[yolov8](yolov8/README.md)| |[weiyun](https://share.weiyun.com/3T3mZKBm)| [google driver](https://drive.google.com/drive/folders/1-8phZHkx_Z274UVqgw6Ma-6u5AKmqCOv?usp=sharing)|  
|more...(🚀: I will be back soon!)    |      |          |
</div>  

## Customization

To change models, you just change the trt file. Edit the variable `engine_file` in `tensorrt_alpha.launch`.

To use your own models, inherit class `TRTAROS::Network` and implement these interfaces:

```cpp
virtual bool init(const std::vector<unsigned char>& trtFile);
virtual void check();
virtual void copy(const std::vector<cv::Mat>& imgsBatch);
virtual void preprocess(const std::vector<cv::Mat>& imgsBatch);
virtual bool infer();
virtual void postprocess(const std::vector<cv::Mat>& imgsBatch);
virtual void reset();
virtual void task(const utils::InitParameter& param, std::vector<cv::Mat>& imgsBatch, 
     const int& delayTime, const int& batchi, const bool& isShow, const bool& isSave) = 0;
```
