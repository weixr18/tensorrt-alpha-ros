/*********************************************************************************
 *      Project: TensorRT-Alpha-ROS                                              *
 *       Author: @FeiYull (https://github.com/FeiYull)                           *
 *     Modified: Xinran Wei (weixr0605@sina.com)                                 *
 *  Modified on: Apr 27, 2023                                                    *
 *                                                                               *
 *  Copyright (c) 2023, Xinran Wei.                                              *
 *  This code file along with the project is published under MIT LICENCE.        *
 *********************************************************************************/


#include "yolo/yolov4.h"

namespace TRTAROS {

__global__ void decode_YOLOV4_device_kernel(int batch_size, int  num_class, int topK, float conf_thresh,
									float* src, int srcWidth, int srcHeight, int srcArea, 
									float* dst, int dstWidth, int dstHeight, int dstArea)
{
	int dx = blockDim.x * blockIdx.x + threadIdx.x; // "srcArea" dim
	int dy = blockDim.y * blockIdx.y + threadIdx.y; // "batch size" dim
	if (dx >= srcHeight || dy >= batch_size)
	{
		return;
	}
	float* pitem = src + dy * srcArea + dx * srcWidth;
	//float objectness = pitem[4]; //  Pr(Object)
	//if (objectness < conf_thresh)
	//{
	//	return;
	//}
	// find max Pr(Classi/Object)
	//float* class_confidence = pitem + 5;  // Pr(Class0/Object)
	float* class_confidence = pitem + 4;    // Pr(Class0/Object)
	float confidence = *class_confidence++; // Pr(Class1/Object)
	int label = 0;
	for (int i = 1; i < num_class; ++i, ++class_confidence)
	{
		if (*class_confidence > confidence)
		{
			confidence = *class_confidence;
			label = i;
		}
	}
	//confidence *= objectness; // Pr(Class0/Object) * Pr(Object)
	if (confidence < conf_thresh)
	{
		return;
	}
	
	// parray:count, box1, box2, box3(count:)
	// parray[0]:count
	// atomicAdd -> count += 1
	// atomicAdd: return old_count
	//int index = atomicAdd(dst + dy * dstArea, 1);
	//assert(dy == 1);
	int index = atomicAdd(dst + dy * dstArea, 1);
	//int index = atomicAdd(&(dst + dy * dstWidth)[0], 1);
	if (index >= topK)
	{
		return;
	}
	//printf("count = %f \n", (dst + dy * dstArea)[0]);
	// xywh -> xyxy
	float cx = *pitem++;
	float cy = *pitem++;
	float width = *pitem++;
	float height = *pitem++;

	/*float left = cx - width * 0.5f;
	float top = cy - height * 0.5f;
	float right = cx + width * 0.5f;
	float bottom = cy + height * 0.5f;*/

	float left = cx;
	float top = cy;
	float right = width;
	float bottom = height;
	// 
	//float* pout_item = dst + dy * dstArea + 1 + index * dstWidth;
	float* pout_item = dst + dy * dstArea + 1 + index * dstWidth;
	*pout_item++ = left; // todo
	*pout_item++ = top;
	*pout_item++ = right;
	*pout_item++ = bottom;

	/**pout_item++ = *pitem++;
	*pout_item++ = *pitem++;
	*pout_item++ = *pitem++;
	*pout_item++ = *pitem++;*/

	*pout_item++ = confidence;
	*pout_item++ = label;
	*pout_item++ = 1;// 1 = keep, 0 = ignore
	//*pout_item = 1;// 1 = keep, 0 = ignore
}

static __device__ float box_iou(
	float aleft, float atop, float aright, float abottom,
	float bleft, float btop, float bright, float bbottom
) {
	float cleft = max(aleft, bleft);
	float ctop = max(atop, btop);
	float cright = min(aright, bright);
	float cbottom = min(abottom, bbottom);

	float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
	if (c_area == 0.0f)
		return 0.0f;

	float a_area = max(0.0f, aright - aleft) * max(0.0f, abottom - atop);
	float b_area = max(0.0f, bright - bleft) * max(0.0f, bbottom - btop);
	return c_area / (a_area + b_area - c_area);
}

void YOLOV4::decodeDevice(utils::InitParameter param, 
	float* src, int srcWidth, int srcHeight, int srcArea, 
	float* dst, int dstWidth, int dstHeight)
{
	dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid_size((srcHeight + BLOCK_SIZE - 1) / BLOCK_SIZE,
		(param.batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int dstArea = 1 + dstWidth * dstHeight;
	
	decode_YOLOV4_device_kernel << < grid_size, block_size, 0, nullptr >> >(
		param.batch_size, param.num_class, param.topK, param.conf_thresh,
		src, srcWidth, srcHeight, srcArea, 
		dst, dstWidth, dstHeight, dstArea
	);
}


}