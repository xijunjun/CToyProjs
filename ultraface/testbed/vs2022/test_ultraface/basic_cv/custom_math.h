#ifndef CUSTOM_MATH_H
#define CUSTOM_MATH_H

#include<cmath>
#include<algorithm>

bool customMathGetAlignParam5pts(float pts_src_[10], float pts_dst_[10], double* alignparam, double* alignparam_inv);

bool getFullAffParamCustom(float* pts_src, float* pts_dst, double* alignparam, double* alignparam_inv);

#endif