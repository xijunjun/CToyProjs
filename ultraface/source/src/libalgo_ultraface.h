#ifndef LIBALGO_ULTRAFACE_H
#define LIBALGO_ULTRAFACE_H

#include <opencv2/opencv.hpp>

#ifdef MATHLIBRARY_EXPORTS
#define MATHLIBRARY_API __declspec(dllexport)
#else
#define MATHLIBRARY_API __declspec(dllimport)
#endif



MATHLIBRARY_API cv::Mat ultraface_process();
                                               

#endif /* MNNDefine_h */