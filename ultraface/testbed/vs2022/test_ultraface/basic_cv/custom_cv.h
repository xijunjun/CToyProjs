#ifndef CUSTOM_CV_H
#define CUSTOM_CV_H

// �����ʾBGRͼ��Ľṹ��
struct CustomImage {
	int width;         // ͼ����
	int height;        // ͼ��߶�
	unsigned char* data;    // ָ��ͼ���������ݵ�ָ��  �Ų�Ϊbgrbgr
	int channels;      // ͼ��ͨ����
	int bytesPerRow;   // ÿ���������ݵ��ֽ���
};
bool CustomWarpBGR(const CustomImage* src, CustomImage* dst, const double affineArray[6]);

#endif