#include<cmath>
#include<algorithm>
#include <iostream>
#include"custom_cv.h"


// 辅助函数：将值限制在最小值和最大值之间
template <typename T>
T clamp(T value, T minVal, T maxVal) {
	return std::max(minVal, std::min(value, maxVal));
}

// 仿射变换函数：CustomWarpBGR
bool CustomWarpBGR(const CustomImage* src, CustomImage* dst, const double affineArray[6]) {
	// 检查源图像和目标图像是否有效
	if (src == nullptr || src->data == nullptr || dst == nullptr || dst->data == nullptr) {
		return false;
	}

	// 循环遍历目标图像像素并应用逆向仿射变换
	for (int y = 0; y < dst->height; y++) {
		for (int x = 0; x < dst->width; x++) {
			// 计算源图像中对应的坐标,注意这里和opencv的仿射参数是互逆的，这里是算dst点在原图的位置
			double srcX = affineArray[0] * x + affineArray[1] * y + affineArray[2];
			double srcY = affineArray[3] * x + affineArray[4] * y + affineArray[5];
			;
			// 双线性插值获取源坐标处的颜色值
			int srcX0 = static_cast<int>(std::floor(srcX));
			int srcX1 = srcX0 + 1;
			int srcY0 = static_cast<int>(std::floor(srcY));
			int srcY1 = srcY0 + 1;

			// 边界检查，确保不会超出图像范围
			srcX0 = clamp(srcX0, 0, src->width - 1);
			srcX1 = clamp(srcX1, 0, src->width - 1);
			srcY0 = clamp(srcY0, 0, src->height - 1);
			srcY1 = clamp(srcY1, 0, src->height - 1);

			double dx = srcX - srcX0;
			double dy = srcY - srcY0;

			// 获取四个相邻像素的BGR值
			const unsigned char* p00 = src->data + (srcY0 * src->bytesPerRow) + (srcX0 * src->channels);
			const unsigned char* p01 = src->data + (srcY0 * src->bytesPerRow) + (srcX1 * src->channels);
			const unsigned char* p10 = src->data + (srcY1 * src->bytesPerRow) + (srcX0 * src->channels);
			const unsigned char* p11 = src->data + (srcY1 * src->bytesPerRow) + (srcX1 * src->channels);

			// 双线性插值计算BGR值
			for (int channel = 0; channel < src->channels; channel++) {
				double value = p00[channel] * (1 - dx) * (1 - dy) + p01[channel] * dx * (1 - dy) +
					p10[channel] * (1 - dx) * dy + p11[channel] * dx * dy;

				// 获取目标图像中当前通道像素的索引
				int dstIndex = (y * dst->bytesPerRow) + (x * dst->channels) + channel;

				// 将插值得到的颜色值赋给目标图像像素
				dst->data[dstIndex] = static_cast<unsigned char>(value);
			}
		}
	}
	return true;
}