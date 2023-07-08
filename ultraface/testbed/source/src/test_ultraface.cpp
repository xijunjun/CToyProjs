

#include <iostream>
#include "libalgo_ultraface.h"

#include <opencv2/opencv.hpp>

int main() {

  cv::Mat image = ultraface_process();

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