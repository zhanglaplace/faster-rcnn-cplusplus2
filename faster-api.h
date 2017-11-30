#ifndef _FASTER_RCNN_API_H
#define _FASTER_RCNN_API_H
#define INPUT_SIZE_NARROW  600
#define INPUT_SIZE_LONG  1000

#include <string>
#include <caffe/net.hpp>
#include <caffe/common.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <memory>
#include <map>

typedef struct abox
{
	float x1;
	float y1;
	float x2;
	float y2;
	float score;
	bool operator <(const abox&tmp) const{
		return score < tmp.score;
	}
}abox;
void nms(std::vector<abox>& input_boxes,float nms_thresh);
cv::Mat bbox_tranform_inv(cv::Mat, cv::Mat);
using namespace std;

class ObjectDetector
{
public:

      ObjectDetector(const std::string &model_file, const std::string &weights_file);  //构造函数
    //对一张图片，进行检测，将结果保存进map数据结构里,分别表示每个类别对应的目标框，如果需要分数信息，则计算分数
      map<int,vector<cv::Rect> > detect(const cv::Mat& image, map<int,vector<float> >* score=NULL);

private:
    boost::shared_ptr< caffe::Net<float> > net_;
    int class_num_;     //类别数+1   ,官方给的demo 是20+1类
};
#endif
