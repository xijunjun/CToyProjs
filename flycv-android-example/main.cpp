#include "flycv.h"

int main(int argc, char **argv) {
  fcv::Mat dst;
  fcv::Mat src = fcv::imread("user.jpg");
  fcv::resize(src, dst, fcv::Size(src.width() / 2, src.height() / 2));
  fcv::imwrite("resize.jpg", dst);

  return 0;
}