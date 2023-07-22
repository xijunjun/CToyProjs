
#include "libalgo_ultraface.h"
#include <limits.h>
#include <utility>

#include "ultraface.h"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;

cv::Mat ultraface_process() {

  string mnn_path = "D:\\workspace\\CToyProjs\\CToyProjs\\ultraface\\source\\models\\mnn\\version-RFB\\RFB-320.mnn";
  UltraFace ultraface(mnn_path, 320, 240, 4, 0.65); // config model input


    string image_file = "D:\\workspace\\data\\pics\\imgs\\2.jpg";
    cout << "Processing " << image_file << endl;

    cv::Mat frame = cv::imread(image_file);
    auto start = chrono::steady_clock::now();
    vector<FaceInfo> face_info;
    ultraface.detect(frame, face_info);

    for (auto face : face_info) {
      cv::Point pt1(face.x1, face.y1);
      cv::Point pt2(face.x2, face.y2);
      cv::rectangle(frame, pt1, pt2, cv::Scalar(0, 255, 0), 2);
    }

    auto end = chrono::steady_clock::now();
    chrono::duration<double> elapsed = end - start;
    cout << "all time: " << elapsed.count() << " s" << endl;

    return frame;

}