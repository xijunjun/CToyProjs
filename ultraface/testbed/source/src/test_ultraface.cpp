

#include <iostream>
#include "libalgo_ultraface.h"
#include<string>
#include <opencv2/opencv.hpp>

using namespace std;
int main() {

	//cv::Mat image = ultraface_process();
	ParamUltraFace ParamUltraFace;
	ParamUltraFace.input_height = 240;
	ParamUltraFace.input_width = 320;
	ParamUltraFace.iou_threshold_ = 0.3;
	ParamUltraFace.score_threshold_=0.7;
	ParamUltraFace.num_thread_ = 4;
	ParamUltraFace.mnn_path= "D:\\workspace\\CToyProjs\\CToyProjs\\ultraface\\source\\models\\mnn\\version-RFB\\RFB-320.mnn";


	UltraFaceDetect ultra_face_detect(&ParamUltraFace); // config model input


	string image_file = "D:\\workspace\\data\\pics\\imgs\\2.jpg";
	cout << "Processing " << image_file << endl;

	cv::Mat image = cv::imread(image_file);
	auto start = chrono::steady_clock::now();
	vector<FaceInfo> face_info;
	ultra_face_detect.detect(image, face_info);

	for (auto face : face_info) {
		cv::Point pt1(face.x1, face.y1);
		cv::Point pt2(face.x2, face.y2);
		cv::rectangle(image, pt1, pt2, cv::Scalar(0, 255, 0), 2);
	}

	auto end = chrono::steady_clock::now();
	chrono::duration<double> elapsed = end - start;
	cout << "all time: " << elapsed.count() << " s" << endl;




	// 如果图像成功读取
	if (!image.empty()) {
		// 创建窗口
		cv::namedWindow("Image", cv::WINDOW_NORMAL);
		// 调整窗口大小以适应图像
		cv::resizeWindow("Image", image.cols, image.rows);
		// 显示图像
		cv::imshow("Image", image);
		// 等待按下任意键继续执行
		cv::waitKey(0);
		// 关闭窗口
		cv::destroyAllWindows();

		system("pause");
	}
}