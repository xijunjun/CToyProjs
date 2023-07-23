#ifndef LIBALGO_BASICDEF_H
#define LIBALGO_BASICDEF_H

#include <opencv2/opencv.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

typedef struct AlgoPointF {
	float x;
	float y;
};

typedef struct AlgoFaceRect {
	AlgoPointF tl;
	AlgoPointF br;

} AlgoFaceRect;

typedef struct AlgoFaceRectList {
	AlgoFaceRect* pfaces;
	int num_face_list;
} AlgoFaceRectList;


typedef struct AlgoFaceLand {
	AlgoPointF p_face_land[200];
	int num_face_land;
} AlgoFaceLand,* PAlgoFaceLand;

typedef struct AlgoFaceLandList {
	AlgoPointF* p_face_land;
	int num_land_oneface;
	int num_face_list;
} AlgoFaceLandList,*PAlgoFaceLandList;


#endif