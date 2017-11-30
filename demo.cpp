#include "faster-api.h"
#include<opencv2/opencv.hpp>
#include <chrono>
#include<iostream>
#include<sstream>
using namespace cv;
using namespace std;
string num2str(float i){
	stringstream ss;
	ss<<i;
	return ss.str();
}

int main(int argc,char **argv){
  ::google::InitGoogleLogging(argv[0]);
#ifdef CPU_ONLY
  cout<<"Use CPU\n";
#else
  cout<<"Use GPU\n";
#endif

  ObjectDetector detect("../models/test.prototxt","../models/VGG16_faster_rcnn_final.caffemodel");
  ifstream fin("../testImage.txt",std::ios_base::in);
  string lines;
  while(getline(fin,lines)){
    cout<<lines<<endl;
    std::chrono::time_point<std::chrono::system_clock> p0 = std::chrono::system_clock::now();
    Mat img = imread(lines.c_str());
    if(img.empty()){
        cout<<"input image error: "<<lines<<endl;
        continue; 
     }
     map<int,vector<float> > score;
     map<int,vector<Rect> > label_objs=detect.detect(img,&score);  //目标检测,同时保存每个框的置信度

    for(map<int,vector<Rect> >::iterator it=label_objs.begin();it!=label_objs.end();it++){
        int label=it->first;  //标签
	    vector<Rect> rects=it->second;  //检测框
	    for(int j=0;j<rects.size();j++){
		      rectangle(img,rects[j],Scalar(0,0,255),2);   //画出矩形框
		     string txt=num2str(label)+" : "+num2str(score[label][j]);
		     putText(img,txt,Point(rects[j].x,rects[j].y),CV_FONT_HERSHEY_SIMPLEX,0.5,Scalar(0,255,0)); //标记 类别：置信度
	     }
    }
    std::chrono::time_point<std::chrono::system_clock> p1 = std::chrono::system_clock::now();
    cout << "detection time:" << (float)std::chrono::duration_cast<std::chrono::microseconds>(p1 - p0).count() / 1000 << "ms" << endl;
    imshow("", img);
    waitKey();
  }
  fin.close();
  return 0;
}
