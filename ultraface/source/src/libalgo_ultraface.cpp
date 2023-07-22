
#include "libalgo_ultraface.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include "Tensor.hpp"
#include "ImageProcess.hpp"
#include "Interpreter.hpp"
#include "MNNDefine.h"

#include "ultraface.h"
#define num_featuremap 4
#define hard_nms 1
#define blending_nms                                                           \
  2 /* mix nms was been proposaled in paper blaze face, aims to minimize the   \
       temporal jitter*/
#define clip(x, y) (x < 0 ? 0 : (x > y ? y : x))

using namespace std;
struct UltraFaceParamDict;

void
nms(UltraFaceParamDict* ptemp,
	std::vector<FaceInfo>& input,
	std::vector<FaceInfo>& output,
	int type);
void
generateBBox(UltraFaceParamDict* ptemp,
	std::vector<FaceInfo>& bbox_collection,
	MNN::Tensor* scores,
	MNN::Tensor* boxes);

struct UltraFaceParamDict
{
	std::shared_ptr<MNN::Interpreter> ultraface_interpreter;
	MNN::Session* ultraface_session = nullptr;
	MNN::Tensor* input_tensor = nullptr;

	int num_thread;
	int image_w;
	int image_h;

	int in_w;
	int in_h;
	int num_anchors;

	float score_threshold;
	float iou_threshold;

	const float mean_vals[3] = { 127, 127, 127 };
	const float norm_vals[3] = { 1.0 / 128, 1.0 / 128, 1.0 / 128 };

	const float center_variance = 0.1;
	const float size_variance = 0.2;
	const std::vector<std::vector<float>> min_boxes = {
	  { 10.0f, 16.0f, 24.0f },
	  { 32.0f, 48.0f },
	  { 64.0f, 96.0f },
	  { 128.0f, 192.0f, 256.0f }
	};
	const std::vector<float> strides = { 8.0, 16.0, 32.0, 64.0 };
	std::vector<std::vector<float>> featuremap_size;
	std::vector<std::vector<float>> shrinkage_size;
	std::vector<int> w_h_list;

	std::vector<std::vector<float>> priors = {};
};

UltraFaceDetect::UltraFaceDetect(ParamUltraFace* pParamUltraFace)
{
	//p_ultraface_paramdict = malloc(sizeof(UltraFaceParamDict));
	p_ultraface_paramdict = new UltraFaceParamDict;
	if (!p_ultraface_paramdict) {
		return;
	}
	UltraFaceParamDict* ptemp = (UltraFaceParamDict*)p_ultraface_paramdict;
	ptemp->image_w = pParamUltraFace->input_width;
	ptemp->image_h = pParamUltraFace->input_height;
	ptemp->num_thread = pParamUltraFace->num_thread_;
	ptemp->iou_threshold = pParamUltraFace->iou_threshold_;
	ptemp->score_threshold = pParamUltraFace->score_threshold_;


	//
	int num_thread = ptemp->num_thread;
	float score_threshold = ptemp->score_threshold;
	float iou_threshold = ptemp->iou_threshold;
	ptemp->in_w = ptemp->image_w;
	ptemp->in_h = ptemp->image_h;
	std::vector<int> w_h_list = { ptemp->in_w, ptemp->in_h };

	cout << "ptemp->strides.size():" << ptemp->strides.size() << endl;;

	for (auto size : w_h_list) {
		std::vector<float> fm_item;
		for (float stride : ptemp->strides) {
			fm_item.push_back(ceil(size / stride));
		}
		ptemp->featuremap_size.push_back(fm_item);
	}

	for (auto size : w_h_list) {
		ptemp->shrinkage_size.push_back(ptemp->strides);
	}
	/* generate prior anchors */
	for (int index = 0; index < num_featuremap; index++) {
		float scale_w = ptemp->in_w / ptemp->shrinkage_size[0][index];
		float scale_h = ptemp->in_h / ptemp->shrinkage_size[1][index];
		for (int j = 0; j < ptemp->featuremap_size[1][index]; j++) {
			for (int i = 0; i < ptemp->featuremap_size[0][index]; i++) {
				float x_center = (i + 0.5) / scale_w;
				float y_center = (j + 0.5) / scale_h;

				for (float k : ptemp->min_boxes[index]) {
					float w = k / ptemp->in_w;
					float h = k / ptemp->in_h;
					ptemp->priors.push_back(
						{ clip(x_center, 1), clip(y_center, 1), clip(w, 1), clip(h, 1) });
				}
			}
		}
	}

	ptemp->num_anchors = ptemp->priors.size();

	ptemp->ultraface_interpreter = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(pParamUltraFace->mnn_path.c_str()));
	MNN::ScheduleConfig config;
	config.numThread = num_thread;
	MNN::BackendConfig backendConfig;
	backendConfig.precision = (MNN::BackendConfig::PrecisionMode)2;
	config.backendConfig = &backendConfig;
	ptemp->ultraface_session = ptemp->ultraface_interpreter->createSession(config);
	ptemp->input_tensor = ptemp->ultraface_interpreter->getSessionInput(ptemp->ultraface_session, nullptr);

}

UltraFaceDetect::~UltraFaceDetect()
{
	UltraFaceParamDict* ptemp = (UltraFaceParamDict*)p_ultraface_paramdict;
	ptemp->ultraface_interpreter->releaseModel();
	ptemp->ultraface_interpreter->releaseSession(ptemp->ultraface_session);
	delete ptemp;
}

int
UltraFaceDetect::detect(cv::Mat& raw_image, std::vector<FaceInfo>& face_list)
{
	if (raw_image.empty()) {
		std::cout << "image is empty ,please check!" << std::endl;
		return -1;
	}
	UltraFaceParamDict* ptemp = (UltraFaceParamDict*)p_ultraface_paramdict;
	ptemp->image_h = raw_image.rows;
	ptemp->image_w = raw_image.cols;
	cv::Mat image;
	cv::resize(raw_image, image, cv::Size(ptemp->in_w, ptemp->in_h));

	ptemp->ultraface_interpreter->resizeTensor(
		ptemp->input_tensor, { 1, 3, ptemp->in_h, ptemp->in_w });
	ptemp->ultraface_interpreter->resizeSession(ptemp->ultraface_session);
	std::shared_ptr<MNN::CV::ImageProcess> pretreat(MNN::CV::ImageProcess::create(
		MNN::CV::BGR, MNN::CV::RGB, ptemp->mean_vals, 3, ptemp->norm_vals, 3));
	pretreat->convert(
		image.data, ptemp->in_w, ptemp->in_h, image.step[0], ptemp->input_tensor);

	auto start = chrono::steady_clock::now();

	// run network
	ptemp->ultraface_interpreter->runSession(ptemp->ultraface_session);

	// get output data

	string scores = "scores";
	string boxes = "boxes";
	MNN::Tensor* tensor_scores = ptemp->ultraface_interpreter->getSessionOutput(
		ptemp->ultraface_session, scores.c_str());
	MNN::Tensor* tensor_boxes = ptemp->ultraface_interpreter->getSessionOutput(
		ptemp->ultraface_session, boxes.c_str());

	MNN::Tensor tensor_scores_host(tensor_scores,
		tensor_scores->getDimensionType());

	tensor_scores->copyToHostTensor(&tensor_scores_host);

	MNN::Tensor tensor_boxes_host(tensor_boxes, tensor_boxes->getDimensionType());

	tensor_boxes->copyToHostTensor(&tensor_boxes_host);

	std::vector<FaceInfo> bbox_collection;

	auto end = chrono::steady_clock::now();
	chrono::duration<double> elapsed = end - start;
	cout << "inference time:" << elapsed.count() << " s" << endl;

	generateBBox(ptemp, bbox_collection, tensor_scores, tensor_boxes);
	nms(ptemp, bbox_collection, face_list, hard_nms);
	return 0;
}

void
nms(UltraFaceParamDict* ptemp,
	std::vector<FaceInfo>& input,
	std::vector<FaceInfo>& output,
	int type)
{
	std::sort(
		input.begin(), input.end(), [](const FaceInfo& a, const FaceInfo& b) {
		return a.score > b.score;
	});

	int box_num = input.size();

	std::vector<int> merged(box_num, 0);

	for (int i = 0; i < box_num; i++) {
		if (merged[i])
			continue;
		std::vector<FaceInfo> buf;

		buf.push_back(input[i]);
		merged[i] = 1;

		float h0 = input[i].y2 - input[i].y1 + 1;
		float w0 = input[i].x2 - input[i].x1 + 1;

		float area0 = h0 * w0;

		for (int j = i + 1; j < box_num; j++) {
			if (merged[j])
				continue;

			float inner_x0 = input[i].x1 > input[j].x1 ? input[i].x1 : input[j].x1;
			float inner_y0 = input[i].y1 > input[j].y1 ? input[i].y1 : input[j].y1;

			float inner_x1 = input[i].x2 < input[j].x2 ? input[i].x2 : input[j].x2;
			float inner_y1 = input[i].y2 < input[j].y2 ? input[i].y2 : input[j].y2;

			float inner_h = inner_y1 - inner_y0 + 1;
			float inner_w = inner_x1 - inner_x0 + 1;

			if (inner_h <= 0 || inner_w <= 0)
				continue;

			float inner_area = inner_h * inner_w;

			float h1 = input[j].y2 - input[j].y1 + 1;
			float w1 = input[j].x2 - input[j].x1 + 1;

			float area1 = h1 * w1;

			float score;

			score = inner_area / (area0 + area1 - inner_area);

			if (score > ptemp->iou_threshold) {
				merged[j] = 1;
				buf.push_back(input[j]);
			}
		}
		switch (type) {
		case hard_nms: {
			output.push_back(buf[0]);
			break;
		}
		case blending_nms: {
			float total = 0;
			for (int i = 0; i < buf.size(); i++) {
				total += exp(buf[i].score);
			}
			FaceInfo rects;
			memset(&rects, 0, sizeof(rects));
			for (int i = 0; i < buf.size(); i++) {
				float rate = exp(buf[i].score) / total;
				rects.x1 += buf[i].x1 * rate;
				rects.y1 += buf[i].y1 * rate;
				rects.x2 += buf[i].x2 * rate;
				rects.y2 += buf[i].y2 * rate;
				rects.score += buf[i].score * rate;
			}
			output.push_back(rects);
			break;
		}
		default: {
			printf("wrong type of nms.");
			exit(-1);
		}
		}
	}
}

void
generateBBox(UltraFaceParamDict* ptemp,
	std::vector<FaceInfo>& bbox_collection,
	MNN::Tensor* scores,
	MNN::Tensor* boxes)
{
	for (int i = 0; i < ptemp->num_anchors; i++) {
		if (scores->host<float>()[i * 2 + 1] > ptemp->score_threshold) {
			FaceInfo rects;
			float x_center = boxes->host<float>()[i * 4] * ptemp->center_variance *
				ptemp->priors[i][2] +
				ptemp->priors[i][0];
			float y_center = boxes->host<float>()[i * 4 + 1] *
				ptemp->center_variance * ptemp->priors[i][3] +
				ptemp->priors[i][1];
			float w = exp(boxes->host<float>()[i * 4 + 2] * ptemp->size_variance) *
				ptemp->priors[i][2];
			float h = exp(boxes->host<float>()[i * 4 + 3] * ptemp->size_variance) *
				ptemp->priors[i][3];

			rects.x1 = clip(x_center - w / 2.0, 1) * ptemp->image_w;
			rects.y1 = clip(y_center - h / 2.0, 1) * ptemp->image_h;
			rects.x2 = clip(x_center + w / 2.0, 1) * ptemp->image_w;
			rects.y2 = clip(y_center + h / 2.0, 1) * ptemp->image_h;
			rects.score = clip(scores->host<float>()[i * 2 + 1], 1);
			bbox_collection.push_back(rects);
		}
	}
}

