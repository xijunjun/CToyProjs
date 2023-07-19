// basic_cv.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include "custom_math.h"
#include <vector>
#include "custom_cv.h"
#include "custom_math.h"

//// 定义一个点的结构体
//struct Point {
//	float x;
//	float y;
//};

using namespace std;


//512模板
float face_template_512[10] = { 192.98138, 239.94708,318.90277, 240.1936,256.63416, 314.01935,201.26117, 371.41043,313.08905, 371.15118 };


struct Point {
	float x, y; // Source points
	float u, v; // Destination points
};

cv::Mat limit_window_auto(cv::Mat img)
{
	cv::Mat img_vis;
#define WINDOW_IMSHOW_HEIGHT  1000.0
#define WINDOW_IMSHOW_WIDTH  1900.0
	int imgw = img.cols;
	int imgh = img.rows;
	float scale_ratio = 1.0;
	if (1.0 * imgw / imgh > WINDOW_IMSHOW_WIDTH / WINDOW_IMSHOW_HEIGHT) {
		scale_ratio = WINDOW_IMSHOW_WIDTH / imgw;
	}
	else {
		scale_ratio = WINDOW_IMSHOW_HEIGHT / imgh;
	}
	if (scale_ratio >= 1)return img.clone();
	cv::resize(img, img_vis, cv::Size(), scale_ratio, scale_ratio);
	return img_vis;
}




void get_affparam_opencv(float* pts_src, float* pts_dst)
{
	// 5个源点和目标点的坐标
	float x1 = pts_src[0], y1 = pts_src[1], u1 = pts_dst[0], v1 = pts_dst[1];
	float x2 = pts_src[2], y2 = pts_src[3], u2 = pts_dst[2], v2 = pts_dst[3];
	float x3 = pts_src[4], y3 = pts_src[5], u3 = pts_dst[4], v3 = pts_dst[5];
	float x4 = pts_src[6], y4 = pts_src[7], u4 = pts_dst[6], v4 = pts_dst[7];
	float x5 = pts_src[8], y5 = pts_src[9], u5 = pts_dst[8], v5 = pts_dst[9];
	// 两个点集的对应关系
	std::vector<cv::Point2f> src_points = { cv::Point2f(x1, y1),cv::Point2f(x2, y2),cv::Point2f(x3, y3),cv::Point2f(x4, y4),cv::Point2f(x5, y5) };
	std::vector<cv::Point2f> dst_points = { cv::Point2f(u1, v1),cv::Point2f(u2, v2),cv::Point2f(u3, v3),cv::Point2f(u4, v4),cv::Point2f(u5, v5) };

	// 估计仿射变换参数矩阵
	cv::Mat affine_matrix = cv::estimateAffinePartial2D(dst_points, src_points, cv::noArray(), cv::LMEDS);

	// 输出仿射变换矩阵
	std::cout << "Affine Matrix1: " << std::endl << affine_matrix << std::endl;
	affine_matrix = cv::estimateAffinePartial2D(src_points, dst_points, cv::noArray(), cv::LMEDS);
	std::cout << "Affine Matrix2: " << std::endl << affine_matrix << std::endl;
}




int main()
{
	string imgpath = "D:\\workspace\\data\\pics\\temp\\facetemp.jpg";
	cv::Mat imgface = cv::imread(imgpath);
	float faceland5[10] = { 290,362,433,363,363,453,300,515,414,517 };
	cv::Mat imgvis = cv::Mat::zeros(512, 512, CV_8UC3);
	cv::Scalar color(0, 0, 255);


	//get_affparam(faceland5, face_template_512);

	get_affparam_opencv(faceland5, face_template_512);

	for (int i = 0; i < 5; i++)
	{
		cv::circle(imgvis, cv::Point(int(face_template_512[i * 2]), int(face_template_512[i * 2 + 1])), 8, color, -1);
	}


	float* pts_src = faceland5;
	float* pts_dst = face_template_512;
	float result[4];
	//double result[4];

	double alignparam[6],  alignparam_inv[6];

	customMathGetAlignParam5pts(pts_src, pts_dst,  alignparam,alignparam_inv);

	//customMathGetAlignParam5pts( pts_dst, pts_src,result);

	// 从数组初始化仿射变换矩阵
	cv::Mat affineMatrix = cv::Mat(2, 3, CV_64F, alignparam);
	cv::Mat affineMatrixInv = cv::Mat(2, 3, CV_64F, alignparam_inv);
	cout << "affineMatrix:" << affineMatrix << endl;
	cout << "affineMatrixInv:" << affineMatrixInv << endl;


	// 进行仿射变换
	cv::Mat outputImage;
	cv::warpAffine(imgface,  outputImage, affineMatrix, cv::Size(512, 512));

	CustomImage face_origin, face_algned;
	face_origin.width = imgface.cols;
	face_origin.height = imgface.rows;
	face_origin.channels = imgface.channels();
	face_origin.bytesPerRow = imgface.step;
	face_origin.data = imgface.data;

	face_algned.width = 512;
	face_algned.height = 512;
	face_algned.channels = 3;
	face_algned.bytesPerRow = 512*3;
	face_algned.data = (uchar*)malloc(512*512*3);

	CustomWarpBGR(&face_origin, &face_algned, alignparam_inv);

	cv::Mat face_algned_cv(512, 512, CV_8UC3, face_algned.data);

	//cv::Mat face_algned_cv( face_origin.height, face_origin.width, CV_8UC3, face_origin.data);

	std::cout << outputImage.cols << " " << outputImage.rows << endl;


	cv::imshow("imgvis", limit_window_auto(imgvis));
	cv::imshow("imgface", limit_window_auto(imgface));
	cv::imshow("outputImage", limit_window_auto(outputImage));
	cv::imshow("face_algned_cv",limit_window_auto(face_algned_cv));

	//cv::imwrite("outputImage.jpg", limit_window_auto(outputImage));
	//cv::imwrite("face_algned_cv.jpg", limit_window_auto(face_algned_cv));


	cv::waitKey(0);


	std::cout << "Hello World!\n";
	//system("pause");
}

