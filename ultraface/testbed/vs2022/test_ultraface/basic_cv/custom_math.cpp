
#include "custom_math.h"

bool hasInverseMatrix(double src[4][4]) {
	double det =
		src[0][0] * (src[1][1] * (src[2][2] * src[3][3] - src[2][3] * src[3][2]) -
			src[1][2] * (src[2][1] * src[3][3] - src[2][3] * src[3][1]) +
			src[1][3] * (src[2][1] * src[3][2] - src[2][2] * src[3][1])) -
		src[0][1] * (src[1][0] * (src[2][2] * src[3][3] - src[2][3] * src[3][2]) -
			src[1][2] * (src[2][0] * src[3][3] - src[2][3] * src[3][0]) +
			src[1][3] * (src[2][0] * src[3][2] - src[2][2] * src[3][0])) +
		src[0][2] * (src[1][0] * (src[2][1] * src[3][3] - src[2][3] * src[3][1]) -
			src[1][1] * (src[2][0] * src[3][3] - src[2][3] * src[3][0]) +
			src[1][3] * (src[2][0] * src[3][1] - src[2][1] * src[3][0])) -
		src[0][3] * (src[1][0] * (src[2][1] * src[3][2] - src[2][2] * src[3][1]) -
			src[1][1] * (src[2][0] * src[3][2] - src[2][2] * src[3][0]) +
			src[1][2] * (src[2][0] * src[3][1] - src[2][1] * src[3][0]));

	return (det != 0.0);
}

// 定义高斯消元法求解逆矩阵
void gaussianElimination(float src[4][8]) {
	int n = 4;

	for (int i = 0; i < n; ++i) {
		// 将当前列的对角线元素缩放为 1
		float scale = src[i][i];
		for (int j = 0; j < 2 * n; ++j) {
			src[i][j] /= scale;
		}

		// 将其他行的对应列元素消为 0
		for (int k = 0; k < n; ++k) {
			if (k != i) {
				float factor = src[k][i];
				for (int j = 0; j < 2 * n; ++j) {
					src[k][j] -= factor * src[i][j];
				}
			}
		}
	}
}

// 4x4 矩阵逆的计算函数
bool inverseMatrix4x4(double src[4][4], double dst[4][4]) {
	if (!hasInverseMatrix(src)){
		return false;
	}

	// 将输入矩阵和单位矩阵合并为增广矩阵
	float augmentedMatrix[4][8];
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			augmentedMatrix[i][j] = src[i][j];
			augmentedMatrix[i][j + 4] = (i == j) ? 1.0 : 0.0;
		}
	}

	// 使用高斯消元法求解逆矩阵
	gaussianElimination(augmentedMatrix);

	// 将计算得到的逆矩阵写入 dst
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			dst[i][j] = augmentedMatrix[i][j + 4];
		}
	}
	return true;
}

void matrixMultiply4x4(const double matrix[4][4], const double vector[4], float result[4]) {
	for (int i = 0; i < 4; ++i) {
		result[i] = 0.0;
		for (int j = 0; j < 4; ++j) {
			result[i] += matrix[i][j] * vector[j];
		}
	}
}

// 计算3x3矩阵的逆矩阵
bool inverseMatrix3x3(double param_src[3][3], double param_dst[3][3]) {
	double detA = param_src[0][0] * (param_src[1][1] * param_src[2][2] - param_src[1][2] * param_src[2][1]) -
		param_src[0][1] * (param_src[1][0] * param_src[2][2] - param_src[1][2] * param_src[2][0]) +
		param_src[0][2] * (param_src[1][0] * param_src[2][1] - param_src[1][1] * param_src[2][0]);

	// 检查行列式是否为0，如果为0则无法求逆
	if (detA == 0.0) {
		return false;
	}

	double invDetA = 1.0 / detA;

	param_dst[0][0] = (param_src[1][1] * param_src[2][2] - param_src[1][2] * param_src[2][1]) * invDetA;
	param_dst[0][1] = (param_src[0][2] * param_src[2][1] - param_src[0][1] * param_src[2][2]) * invDetA;
	param_dst[0][2] = (param_src[0][1] * param_src[1][2] - param_src[0][2] * param_src[1][1]) * invDetA;
	param_dst[1][0] = (param_src[1][2] * param_src[2][0] - param_src[1][0] * param_src[2][2]) * invDetA;
	param_dst[1][1] = (param_src[0][0] * param_src[2][2] - param_src[0][2] * param_src[2][0]) * invDetA;
	param_dst[1][2] = (param_src[0][2] * param_src[1][0] - param_src[0][0] * param_src[1][2]) * invDetA;
	param_dst[2][0] = (param_src[1][0] * param_src[2][1] - param_src[1][1] * param_src[2][0]) * invDetA;
	param_dst[2][1] = (param_src[0][1] * param_src[2][0] - param_src[0][0] * param_src[2][1]) * invDetA;
	param_dst[2][2] = (param_src[0][0] * param_src[1][1] - param_src[0][1] * param_src[1][0]) * invDetA;
	return true;
}







bool customMathGetAlignParam5pts(float pts_src_[10], float pts_dst_[10], double* alignparam, double* alignparam_inv)
{
	//只支持5个点的放射参数
	double P[3] = { 0 }, Q[4] = { 0 };

	double ATA_INV[4][4] = { 0 };
	for (int i = 0; i < 5; i++)
	{
		int xind = i * 2, yind = i * 2 + 1;
		P[0] += pts_src_[xind] * pts_src_[xind] + pts_src_[yind] * pts_src_[yind];
		P[1] += pts_src_[xind];
		P[2] += pts_src_[yind];
		Q[0] += pts_src_[xind] * pts_dst_[xind] + pts_src_[yind] * pts_dst_[yind];
		Q[1] += pts_dst_[xind] * pts_src_[yind] - pts_src_[xind] * pts_dst_[yind];
		Q[2] += pts_dst_[xind];
		Q[3] += pts_dst_[yind];
	}
	double ATA[4][4] = {
		{P[0],0,P[1],P[2]},
		{0,P[0],P[2],-P[1]},
		{P[1],P[2],5,0},
		{P[2],-P[1],0,5 } };

	bool hasinverse = inverseMatrix4x4(ATA, ATA_INV);
	if (!hasinverse) {
		return false;
	}
	float result[6];
	matrixMultiply4x4(ATA_INV, Q, result);

	
	double param_src[3][3] = { {result[0], result[1], result[2]}, { -result[1], result[0], result[3] }, { 0,0,1 }};
	double param_inv[3][3] = {0};
	inverseMatrix3x3(param_src,  param_inv);

	double affineArray[6] = { result[0], result[1], result[2], -result[1], result[0], result[3] };
	double affineArrayInv[6] = { param_inv[0][0],param_inv[0][1],param_inv[0][2] ,param_inv[1][0] ,param_inv[1][1] ,param_inv[1][2] };


	memcpy(alignparam, affineArray,6*sizeof(double));
	memcpy(alignparam_inv, affineArrayInv, 6*sizeof(double));

	return true;
}

bool getFullAffParamCustom(float* pts_src, float* pts_dst, double* alignparam, double* alignparam_inv)
{
	double A[3][3] = {
		{pts_src[0],pts_src[1],1},
		{pts_src[2],pts_src[3],1},
		{pts_src[4],pts_src[5],1}
	};
	double A_inv[3][3] = { 0 };
	double B[3][2]{
		{pts_dst[0],pts_dst[1]},
		{pts_dst[2],pts_dst[3]},
		{pts_dst[4],pts_dst[5]},
	};
	inverseMatrix3x3(A, A_inv);

	double result[3][2] = {0};
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 2; ++j) {
			result[i][j] = 0.0;
			for (int k = 0; k < 3; ++k) {
				result[i][j] += A_inv[i][k] * B[k][j];
			}
		}
	}

	double param_src[3][3] = { {result[0][0], result[1][0], result[2][0]}, {result[0][1], result[1][1], result[2][1]}, {0,0,1}};
	double param_inv[3][3] = { 0 };
	inverseMatrix3x3(param_src, param_inv);

	double affineArray[6] = { result[0][0], result[1][0], result[2][0],result[0][1], result[1][1], result[2][1] };
	double affineArrayInv[6] = { param_inv[0][0],param_inv[0][1],param_inv[0][2] ,param_inv[1][0] ,param_inv[1][1] ,param_inv[1][2] };

	memcpy(alignparam, affineArray, 6 * sizeof(double));
	memcpy(alignparam_inv, affineArrayInv, 6 * sizeof(double));

	return true;
}



/////////////完整的仿射变换，尚未验证
//#include <iostream>
//#include <vector>
//
//struct Point {
//	double x;
//	double y;
//};
//
//// 求解仿射参数
//bool solveAffineParameters(const std::vector<Point>& srcPoints,
//	const std::vector<Point>& dstPoints,
//	double alignparam[6]) {
//	int n = srcPoints.size();
//
//	if (n < 3 || dstPoints.size() != n) {
//		std::cerr << "Error: Invalid number of points or mismatch between srcPoints and dstPoints." << std::endl;
//		return false;
//	}
//
//	double sum_x_src = 0.0, sum_y_src = 0.0, sum_x_dst = 0.0, sum_y_dst = 0.0;
//	double sum_x_src_dst = 0.0, sum_y_src_dst = 0.0;
//	double sum_x_src_sq = 0.0, sum_y_src_sq = 0.0;
//
//	for (int i = 0; i < n; i++) {
//		sum_x_src += srcPoints[i].x;
//		sum_y_src += srcPoints[i].y;
//		sum_x_dst += dstPoints[i].x;
//		sum_y_dst += dstPoints[i].y;
//		sum_x_src_dst += srcPoints[i].x * dstPoints[i].x;
//		sum_y_src_dst += srcPoints[i].y * dstPoints[i].y;
//		sum_x_src_sq += srcPoints[i].x * srcPoints[i].x;
//		sum_y_src_sq += srcPoints[i].y * srcPoints[i].y;
//	}
//
//	double denominator = n * sum_x_src_sq - sum_x_src * sum_x_src;
//
//	if (denominator == 0.0) {
//		std::cerr << "Error: Singular matrix, cannot compute affine parameters." << std::endl;
//		return false;
//	}
//
//	alignparam[0] = (n * sum_x_src_dst - sum_x_src * sum_x_dst) / denominator;
//	alignparam[1] = (sum_x_src_sq * sum_x_dst - sum_x_src * sum_x_src_dst) / denominator;
//
//	alignparam[2] = (n * sum_y_src_dst - sum_y_src * sum_y_dst) / denominator;
//	alignparam[3] = (sum_y_src_sq * sum_y_dst - sum_y_src * sum_y_src_dst) / denominator;
//
//	alignparam[4] = (sum_x_dst - alignparam[0] * sum_x_src - alignparam[1] * sum_y_src) / n;
//	alignparam[5] = (sum_y_dst - alignparam[2] * sum_x_src - alignparam[3] * sum_y_src) / n;
//
//	return true;
//}


