cmake_minimum_required(VERSION 3.0.2)
project(tensorrt_alpha_ros)

##################### Basic flags #####################

add_compile_options(-std=c++11)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")


##################### Find packages #####################

# ROS
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  roscpp
  sensor_msgs
) 

# CUDA
find_package(CUDA REQUIRED) 
list(APPEND ALL_LIBS 
  ${CUDA_LIBRARIES} 
  ${CUDA_cublas_LIBRARY} 
  ${CUDA_nppc_LIBRARY} ${CUDA_nppig_LIBRARY} ${CUDA_nppidei_LIBRARY} ${CUDA_nppial_LIBRARY}
)
list(APPEND INCLUDE_DIRS ${CUDA_INCLUDE_DIRS})

# OpenCV
find_package(OpenCV REQUIRED)
list(APPEND ALL_LIBS ${OpenCV_LIBRARIES})
list(APPEND INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS})

# TensorRT 
set(TensorRT_ROOT /opt/TensorRT-8.2.0.6)
#################################################################
### ↑↑↑ SET TensorRT_ROOT TO YOUR OWN TensorRT's PATH !!!!!   ###
#################################################################
find_library(TRT_NVINFER NAMES nvinfer HINTS ${TensorRT_ROOT} PATH_SUFFIXES lib lib64 lib/x64)
find_library(TRT_NVINFER_PLUGIN NAMES nvinfer_plugin HINTS ${TensorRT_ROOT} PATH_SUFFIXES lib lib64 lib/x64)
find_library(TRT_NVONNX_PARSER NAMES nvonnxparser HINTS ${TensorRT_ROOT} PATH_SUFFIXES lib lib64 lib/x64)
find_library(TRT_NVCAFFE_PARSER NAMES nvcaffe_parser HINTS ${TensorRT_ROOT} PATH_SUFFIXES lib lib64 lib/x64)
find_path(TENSORRT_INCLUDE_DIR NAMES NvInfer.h HINTS ${TensorRT_ROOT} PATH_SUFFIXES include)
list(APPEND ALL_LIBS ${TRT_NVINFER} ${TRT_NVINFER_PLUGIN} ${TRT_NVONNX_PARSER} ${TRT_NVCAFFE_PARSER})
list(APPEND INCLUDE_DIRS ${TENSORRT_INCLUDE_DIR})
set(SAMPLES_COMMON_DIR ${TensorRT_ROOT}/samples/common)
list(APPEND INCLUDE_DIRS ${SAMPLES_COMMON_DIR})



############# Catkin specific configuration #############

catkin_package(
 INCLUDE_DIRS include
 LIBRARIES tensorrt_alpha_ros
 CATKIN_DEPENDS cv_bridge roscpp sensor_msgs
#  DEPENDS system_lib
)


##################### CPP source files #####################

# message(STATUS ALL_LIBS = ${ALL_LIBS})
file(GLOB CPPS 
  ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/yolo/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp  
  ${TensorRT_ROOT}/samples/common/logger.cpp 
  ${TensorRT_ROOT}/samples/common/sampleOptions.cpp 
)
file(GLOB CUS 
${CMAKE_CURRENT_SOURCE_DIR}/src/utils/*.cu
${CMAKE_CURRENT_SOURCE_DIR}/src/yolo/*.cu
${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu  
)
list(APPEND CPPS ${CUS})


list (LENGTH CPPS length)
message(*********************************************************)
message(STATUS TOTAL_Source_File_Number = ${length}) 
message(*********************************************************)


##################### Build #####################

include_directories(
  include/tensorrt_alpha_ros/
  ${INCLUDE_DIRS} 
  ${catkin_INCLUDE_DIRS}
)

cuda_add_library(${PROJECT_NAME} SHARED ${CPPS})
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 50 61 72 75)
target_compile_options(${PROJECT_NAME} PUBLIC 
   $<$<COMPILE_LANGUAGE:CUDA>:--default-stream per-thread -lineinfo; --use_fast_math --disable-warnings>)
target_link_libraries(${PROJECT_NAME} ${ALL_LIBS})

cuda_add_executable(${PROJECT_NAME}_node src/tensorrt_alpha_ros_node.cpp src/tensorrt_alpha_ros.cpp)
target_link_libraries(${PROJECT_NAME}_node ${PROJECT_NAME} ${ALL_LIBS} ${catkin_LIBRARIES})


#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# catkin_install_python(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables for installation
install(TARGETS ${PROJECT_NAME}_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Mark libraries for installation
install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)

## Mark cpp header files for installation
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
  PATTERN ".svn" EXCLUDE
)

## Mark other files for installation (e.g. launch and bag files, etc.)
install(FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/launch/tensorrt_alpha.launch
  # myfile2
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_tensorrt_alpha_ros.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
