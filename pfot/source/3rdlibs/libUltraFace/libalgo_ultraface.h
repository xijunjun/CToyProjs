#ifndef LIBALGO_ULTRAFACE_H
#define LIBALGO_ULTRAFACE_H

#include <opencv2/opencv.hpp>

#ifdef ULTRAFACEDETECT_EXPORTS
#define ULTRAFACEDETECT_API __declspec(dllexport)
#else
#define ULTRAFACEDETECT_API __declspec(dllimport)
#endif

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

typedef struct FaceInfo {
  float x1;
  float y1;
  float x2;
  float y2;
  float score;

} FaceInfo;

struct ParamUltraFace {
  std::string mnn_path;
  int input_width;
  int input_height;
  int num_thread_;
  float score_threshold_;
  float iou_threshold_;
};

class ULTRAFACEDETECT_API UltraFaceDetect {
public:
  UltraFaceDetect(ParamUltraFace *pParamUltraFace);
  ~UltraFaceDetect();
  int detect(cv::Mat &raw_image, std::vector<FaceInfo> &face_list);

private:
  void *p_ultraface_paramdict;
};

#endif 