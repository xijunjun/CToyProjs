#include<cmath>
#include<algorithm>
#include <iostream>
#include"custom_cv.h"


// ������������ֵ��������Сֵ�����ֵ֮��
template <typename T>
T clamp(T value, T minVal, T maxVal) {
	return std::max(minVal, std::min(value, maxVal));
}

// ����任������CustomWarpBGR
bool CustomWarpBGR(const CustomImage* src, CustomImage* dst, const double affineArray[6]) {
	// ���Դͼ���Ŀ��ͼ���Ƿ���Ч
	if (src == nullptr || src->data == nullptr || dst == nullptr || dst->data == nullptr) {
		return false;
	}

	// ѭ������Ŀ��ͼ�����ز�Ӧ���������任
	for (int y = 0; y < dst->height; y++) {
		for (int x = 0; x < dst->width; x++) {
			// ����Դͼ���ж�Ӧ������,ע�������opencv�ķ�������ǻ���ģ���������dst����ԭͼ��λ��
			double srcX = affineArray[0] * x + affineArray[1] * y + affineArray[2];
			double srcY = affineArray[3] * x + affineArray[4] * y + affineArray[5];
			;
			// ˫���Բ�ֵ��ȡԴ���괦����ɫֵ
			int srcX0 = static_cast<int>(std::floor(srcX));
			int srcX1 = srcX0 + 1;
			int srcY0 = static_cast<int>(std::floor(srcY));
			int srcY1 = srcY0 + 1;

			// �߽��飬ȷ�����ᳬ��ͼ��Χ
			srcX0 = clamp(srcX0, 0, src->width - 1);
			srcX1 = clamp(srcX1, 0, src->width - 1);
			srcY0 = clamp(srcY0, 0, src->height - 1);
			srcY1 = clamp(srcY1, 0, src->height - 1);

			double dx = srcX - srcX0;
			double dy = srcY - srcY0;

			// ��ȡ�ĸ��������ص�BGRֵ
			const unsigned char* p00 = src->data + (srcY0 * src->bytesPerRow) + (srcX0 * src->channels);
			const unsigned char* p01 = src->data + (srcY0 * src->bytesPerRow) + (srcX1 * src->channels);
			const unsigned char* p10 = src->data + (srcY1 * src->bytesPerRow) + (srcX0 * src->channels);
			const unsigned char* p11 = src->data + (srcY1 * src->bytesPerRow) + (srcX1 * src->channels);

			// ˫���Բ�ֵ����BGRֵ
			for (int channel = 0; channel < src->channels; channel++) {
				double value = p00[channel] * (1 - dx) * (1 - dy) + p01[channel] * dx * (1 - dy) +
					p10[channel] * (1 - dx) * dy + p11[channel] * dx * dy;

				// ��ȡĿ��ͼ���е�ǰͨ�����ص�����
				int dstIndex = (y * dst->bytesPerRow) + (x * dst->channels) + channel;

				// ����ֵ�õ�����ɫֵ����Ŀ��ͼ������
				dst->data[dstIndex] = static_cast<unsigned char>(value);
			}
		}
	}
	return true;
}