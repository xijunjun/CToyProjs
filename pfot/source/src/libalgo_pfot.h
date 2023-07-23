#ifndef LIBALGO_PFOT_H
#define LIBALGO_PFOT_H

#define WINDOWS

#ifdef WINDOWS
#define PFOT_API __declspec(dllexport)
#else
#define PFOT_API __declspec(dllimport)
#endif

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include "algo_basicdef.h"


#define PFOT_NUMLAND 68
#define PFOT_MAXFACENUM 100

struct ParamPfot {
  std::string mnn_path;
  //AlgoFaceRectList* p_face_rect_list;
  int num_thread;
};



PFOT_API void *pfotCreateHandle(ParamPfot* p_param_pfot);
PFOT_API int pfotProcess(void* pfotHandle_, cv::Mat& raw_image, AlgoFaceRectList* p_face_rect_list, PAlgoFaceLandList* ppland_result);
PFOT_API void pfotDestroyHandle(void* pfotHandle_);

//class ULTRAFACEDETECT_API UltraFaceDetect {
//public:
//  UltraFaceDetect(ParamUltraFace *pParamUltraFace);
//  ~UltraFaceDetect();
//  int detect(cv::Mat &raw_image, std::vector<FaceInfo> &face_list);
//
//private:
//  void *p_ultraface_paramdict;
//};

#endif 