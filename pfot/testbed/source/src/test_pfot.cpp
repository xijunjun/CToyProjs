

#include <iostream>
#include "libalgo_pfot.h"
#include "libalgo_ultraface.h"
#include "algo_basicdef.h"
#include<string>
#include <opencv2/opencv.hpp>
#include "tinydir.h"

using namespace std;
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

void extendFaceRect(AlgoFaceRect* pfacerect_src, AlgoFaceRect* pfacerect_dst, float wratio, float hratio)
{
	float ctx = 0.5 * (pfacerect_src->tl.x + pfacerect_src->br.x);
	float cty = 0.5 * (pfacerect_src->tl.y + pfacerect_src->br.y);
	pfacerect_dst->tl.x = pfacerect_src->tl.x - ctx;
	pfacerect_dst->br.x = pfacerect_src->br.x - ctx;
	pfacerect_dst->tl.y = pfacerect_src->tl.y - cty;
	pfacerect_dst->br.y = pfacerect_src->br.y - cty;
	pfacerect_dst->tl.x *= wratio;
	pfacerect_dst->br.x *= wratio;
	pfacerect_dst->tl.y *= hratio;
	pfacerect_dst->br.y *= hratio;

	pfacerect_dst->tl.x += ctx;
	pfacerect_dst->br.x += ctx;
	pfacerect_dst->tl.y += cty;
	pfacerect_dst->br.y += cty;
}

void getSimAffParamOpencv(float* pts_src, float* pts_dst, double* alignparam, double* alignparam_inv)
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

	cout << "src_points:" << src_points << endl;
	cout << "dst_points:" << dst_points << endl;

	// 估计仿射变换参数矩阵
	cv::Mat affine_matrix = cv::estimateAffinePartial2D(src_points, dst_points, cv::noArray(), cv::LMEDS);
	cv::Mat affine_matrix_inv = affine_matrix.clone();
	cv::invertAffineTransform(affine_matrix, affine_matrix_inv);
	memcpy(alignparam, affine_matrix.ptr<double>(0), 6 * sizeof(double));
	memcpy(alignparam_inv, affine_matrix_inv.ptr<double>(0), 6 * sizeof(double));
}


void getFullAffParamOpencv(float* pts_src, float* pts_dst, double* alignparam, double* alignparam_inv)
{
	// 5个源点和目标点的坐标
	float x1 = pts_src[0], y1 = pts_src[1], u1 = pts_dst[0], v1 = pts_dst[1];
	float x2 = pts_src[2], y2 = pts_src[3], u2 = pts_dst[2], v2 = pts_dst[3];
	float x3 = pts_src[4], y3 = pts_src[5], u3 = pts_dst[4], v3 = pts_dst[5];

	cv::Point2f src[3];
	src[0] = cv::Point2f(x1, y1);
	src[1] = cv::Point2f(x2, y2);
	src[2] = cv::Point2f(x3, y3);

	cv::Point2f dst[3];
	dst[0] = cv::Point2f(u1, v1);
	dst[1] = cv::Point2f(u2, v2);
	dst[2] = cv::Point2f(u3, v3);


	// 估计仿射变换参数矩阵
	cv::Mat affine_matrix = cv::getAffineTransform(src, dst);
	cv::Mat affine_matrix_inv = affine_matrix.clone();
	cv::invertAffineTransform(affine_matrix, affine_matrix_inv);
	memcpy(alignparam, affine_matrix.ptr<double>(0), 6 * sizeof(double));
	memcpy(alignparam_inv, affine_matrix_inv.ptr<double>(0), 6 * sizeof(double));
}



void RectTo5pts(AlgoFaceRect* pfacerect_src, float pts_dst[10])
{
	float ctx = 0.5 * (pfacerect_src->tl.x + pfacerect_src->br.x);
	float cty = 0.5 * (pfacerect_src->tl.y + pfacerect_src->br.y);
	pts_dst[0] = pfacerect_src->tl.x;
	pts_dst[1] = pfacerect_src->tl.y;
	pts_dst[2] = pfacerect_src->br.x;
	pts_dst[3] = pfacerect_src->tl.y;
	pts_dst[4] = ctx;
	pts_dst[5] = cty;
	pts_dst[6] = pfacerect_src->tl.x;
	pts_dst[7] = pfacerect_src->br.y;
	pts_dst[8] = pfacerect_src->br.x;
	pts_dst[9] = pfacerect_src->br.y;
}

void RectTo3pts(AlgoFaceRect* pfacerect_src, float pts_dst[6])
{
	float ctx = 0.5 * (pfacerect_src->tl.x + pfacerect_src->br.x);
	float cty = 0.5 * (pfacerect_src->tl.y + pfacerect_src->br.y);
	pts_dst[0] = pfacerect_src->tl.x;
	pts_dst[1] = pfacerect_src->tl.y;
	pts_dst[2] = pfacerect_src->br.x;
	pts_dst[3] = pfacerect_src->tl.y;
	pts_dst[4] = pfacerect_src->br.x;
	pts_dst[5] = pfacerect_src->br.y;
}


int main() {

	//cv::Mat image = ultraface_process();
	ParamUltraFace ParamUltraFace;
	ParamUltraFace.input_height = 240;
	ParamUltraFace.input_width = 320;
	ParamUltraFace.iou_threshold_ = 0.3;
	ParamUltraFace.score_threshold_ = 0.7;
	ParamUltraFace.num_thread_ = 4;
	ParamUltraFace.mnn_path = "D:\\workspace\\CToyProjs\\CToyProjs\\ultraface\\source\\models\\mnn\\version-RFB\\RFB-320.mnn";


	UltraFaceDetect ultra_face_detect(&ParamUltraFace); // config model input

	ParamPfot param_pfot;
	param_pfot.mnn_path = "D:\\workspace\\CToyProjs\\CToyProjs\\pfot\\source\\models\\mnn\\slim_160_latest.mnn";
	param_pfot.num_thread = 4;
	void* pfotHandle = pfotCreateHandle(&param_pfot);



	tinydir_dir dir;
	int i;
	//string dirpath= "D:\\workspace\\data\\pics\\imgs\\";
	//string dirpath = "D:\\workspace\\data\\pics\\temp\\";
	//string dirpath = "D:\\workspace\\data\\pics\\facehd\\";
	string dirpath ="Z:\\Dataset\\hair\\image_unsplash";
	tinydir_open_sorted(&dir, dirpath.c_str());

	for (i = 0; i < dir.n_files; i++)
	{
		tinydir_file file;
		tinydir_readfile_n(&dir, &file, i);

		printf("%s", file.name);
		if (file.is_dir)
		{
			printf("/");
			continue;
		}
		//	printf("\n");
		//}

		//tinydir_close(&dir);







		//string image_file = "D:\\workspace\\data\\pics\\imgs\\2.jpg";

		string image_file = dirpath + "\\" + file.name;

		cout << "Processing " << image_file << endl;

		cv::Mat image = cv::imread(image_file);
		cv::resize(image, image,cv::Size(),0.25,0.25);


		auto start = chrono::steady_clock::now();
		vector<FaceInfo> face_info;
		ultra_face_detect.detect(image, face_info);

		for (auto face : face_info) {
			cv::Point pt1(face.x1, face.y1);
			cv::Point pt2(face.x2, face.y2);
			//cv::rectangle(image, pt1, pt2, cv::Scalar(0, 255, 0), 2);

			//AlgoFaceRect algofacerct;
			//algofacerct.tl.x = face.x1;
			//algofacerct.tl.y = face.y1;
			//algofacerct.br.x = face.x2;
			//algofacerct.br.y = face.y2;

			//AlgoFaceRect algofacerct_extend;
			//extendFaceRect(&algofacerct, &algofacerct_extend, 1.4, 1.4);
			//cv::Point pt1ex((int)algofacerct_extend.tl.x, (int)algofacerct_extend.tl.y);
			//cv::Point pt2ex((int)algofacerct_extend.br.x, (int)algofacerct_extend.br.y);

			//AlgoFaceRect algofacerct_target;
			//algofacerct_target.tl.x = 0;
			//algofacerct_target.tl.y = 0;
			//algofacerct_target.br.x = 160;
			//algofacerct_target.br.y = 160;


			////cv::Point pt1ex((int)algofacerct.tl.x, (int)algofacerct.tl.y);
			////cv::Point pt2ex((int)algofacerct.br.x, (int)algofacerct.br.y);
			//cv::rectangle(image, pt1ex, pt2ex, cv::Scalar(0, 255, 0), 2);

			//float rct_pts3_src[6], rct_pts3_dst[6];


			//RectTo5pts(&algofacerct_extend, rct_pts3_src);
			//RectTo5pts(&algofacerct_target, rct_pts3_dst);



			//double alignparam[6], alignparam_inv[6];
			//getFullAffParamOpencv(rct_pts3_src, rct_pts3_dst, alignparam, alignparam_inv);

			////getFullAffParamCustom(rct_pts3_src, rct_pts3_dst, alignparam, alignparam_inv);
			////getFullAffParamCustom(rct_pts3_dst, rct_pts3_src, alignparam, alignparam_inv);

			//cv::Mat affineMatrix = cv::Mat(2, 3, CV_64F, alignparam);
			//cv::Mat cropedImage;
			//cv::warpAffine(image, cropedImage, affineMatrix, cv::Size(160, 160));
			//cv::imshow("cropedImage", cropedImage);
			////if(cv::waitKey(0)==27)break;
		}

		auto end = chrono::steady_clock::now();
		chrono::duration<double> elapsed = end - start;
		cout << "all time: " << elapsed.count() << " s" << endl;



		//人脸关键点检测
		PAlgoFaceLandList palgofacelandlist;

		int numface = face_info.size();
		AlgoFaceRectList face_rect_list;
		face_rect_list.pfaces = (AlgoFaceRect*)malloc(numface * sizeof(AlgoFaceRect));
		face_rect_list.num_face_list = numface;
		for (int n = 0; n < numface; n++) {
			AlgoFaceRect* pthisFace = NULL;
			pthisFace = face_rect_list.pfaces + n;
			pthisFace->tl.x = face_info[n].x1;
			pthisFace->tl.y = face_info[n].y1;
			pthisFace->br.x = face_info[n].x2;
			pthisFace->br.y = face_info[n].y2;
		}

		pfotProcess(pfotHandle, image, &face_rect_list, &palgofacelandlist);


		cv::Scalar color(0, 0, 255);
		for (int n = 0; n < numface; n++) {
			AlgoPointF* pthis_faceland = palgofacelandlist->p_face_land + n * PFOT_NUMLAND;
			for (int k = 0; k < PFOT_NUMLAND; k++) {
				cv::circle(image, cv::Point(int(pthis_faceland[k].x), int(pthis_faceland[k].y)), 2, color, -1);
			}
		}

		for (auto face : face_info) {
			cv::Point pt1(face.x1, face.y1);
			cv::Point pt2(face.x2, face.y2);
			cv::rectangle(image, pt1, pt2, cv::Scalar(0, 255, 0), 2);
		}

		//

		cv::imshow("Image", limit_window_auto(image));
		if (27 == cv::waitKey(0)) return 0;

		printf("\n");
	}

	tinydir_close(&dir);

	pfotDestroyHandle(pfotHandle);

	//free(face_rect_list.pfaces);
	return 0;
}