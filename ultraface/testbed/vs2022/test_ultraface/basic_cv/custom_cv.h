#ifndef CUSTOM_CV_H
#define CUSTOM_CV_H

// 定义表示BGR图像的结构体
struct CustomImage {
	int width;         // 图像宽度
	int height;        // 图像高度
	unsigned char* data;    // 指向图像像素数据的指针  排布为bgrbgr
	int channels;      // 图像通道数
	int bytesPerRow;   // 每行像素数据的字节数
};
bool CustomWarpBGR(const CustomImage* src, CustomImage* dst, const double affineArray[6]);

#endif