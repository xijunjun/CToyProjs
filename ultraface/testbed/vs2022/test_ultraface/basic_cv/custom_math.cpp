
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

// �����˹��Ԫ����������
void gaussianElimination(float src[4][8]) {
	int n = 4;

	for (int i = 0; i < n; ++i) {
		// ����ǰ�еĶԽ���Ԫ������Ϊ 1
		float scale = src[i][i];
		for (int j = 0; j < 2 * n; ++j) {
			src[i][j] /= scale;
		}

		// �������еĶ�Ӧ��Ԫ����Ϊ 0
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

// 4x4 ������ļ��㺯��
bool inverseMatrix4x4(double src[4][4], double dst[4][4]) {
	if (!hasInverseMatrix(src)){
		return false;
	}

	// ���������͵�λ����ϲ�Ϊ�������
	float augmentedMatrix[4][8];
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			augmentedMatrix[i][j] = src[i][j];
			augmentedMatrix[i][j + 4] = (i == j) ? 1.0 : 0.0;
		}
	}

	// ʹ�ø�˹��Ԫ����������
	gaussianElimination(augmentedMatrix);

	// ������õ��������д�� dst
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

// ����3x3����������
bool inverseMatrix3x3(double param_src[3][3], double param_dst[3][3]) {
	double detA = param_src[0][0] * (param_src[1][1] * param_src[2][2] - param_src[1][2] * param_src[2][1]) -
		param_src[0][1] * (param_src[1][0] * param_src[2][2] - param_src[1][2] * param_src[2][0]) +
		param_src[0][2] * (param_src[1][0] * param_src[2][1] - param_src[1][1] * param_src[2][0]);

	// �������ʽ�Ƿ�Ϊ0�����Ϊ0���޷�����
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
	//ֻ֧��5����ķ������
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
