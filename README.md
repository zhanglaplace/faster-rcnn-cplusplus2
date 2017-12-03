# faster-rcnn-cplusplus2

  Faster-rcnn cplusplus2 with python model ;
  
  the project is according to the http://blog.csdn.net/zxj942405301/article/details/72775463

  the c++ faster-rcnn with matlab model see:https://github.com/zhanglaplace/Faster_rcnn_Cplusplus_vs2013
# Platform

## Windows

  You need a VC compiler to build these project, Visual Studio 2013 Community should be fine. You can download from https://www.visualstudio.com/downloads/.

## Ubuntu
    mkdir build 
	
	cd build
	
	cmake ..
	
	make .
	
Of course ,you shoule put your model and protoxt file in models directory

# Caffe 
  You should build caffe with RoiPooling layer and rpn_layer
	
- rpn_layer
	- add rpn_layer.cpp to $Caffe/src/caffe/layers/

	- add rpn_layer.hpp to $Caffe/include/caffe/layers/

- caffe.proto
```cpp
	optional RPNParameter rpn_param = 158;
	message RPNParameter {  
	  optional uint32 feat_stride = 1;  
	  optional uint32 basesize = 2;  
	  repeated uint32 scale = 3;  
	  repeated float ratio = 4;  
	  optional uint32 boxminsize =5;  
	  optional uint32 per_nms_topn = 9;  
	  optional uint32 post_nms_topn = 11;  
	  optional float nms_thresh = 8;  
	}  
```
 
- protoxt modify
```cpp
	modify test.prototxt 
	layer {  
	   name: 'proposal'  
	   type: 'Python'  
	   bottom: 'rpn_cls_prob_reshape'  
	   bottom: 'rpn_bbox_pred'  
	   bottom: 'im_info'  
	   top: 'rois'  
	   python_param {  
		 module: 'rpn.proposal_layer'  
		 layer: 'ProposalLayer'  
		param_str: "'feat_stride': 16"  
	   }  
	}  
	to:
	layer {  
	   name: "proposal"  
	   type: "RPN"  
	   bottom: "rpn_cls_prob_reshape"  
	   bottom: "rpn_bbox_pred"  
	   bottom: "im_info"  
	   top: "rois"  
	   rpn_param {  
		   feat_stride : 16  
		   basesize : 16  
		   scale : 8  
		   scale : 16  
		   scale : 32  
		   ratio : 0.5  
		   ratio : 1  
		   ratio : 2  
		   boxminsize :16  
		   per_nms_topn : 0;  
		   post_nms_topn : 0;  
		   nms_thresh : 0.3  
	   }  
	}  
```

# Result
  it's  cost much time the py-faster-rcnn
In My compute(GTX1080Ti GPU) a picture of size(375\*500\*3)cost 361ms .
![image](https://github.com/zhanglaplace/faster-rcnn-cplusplus2/blob/master/imgs/speed.png)
![image](https://github.com/zhanglaplace/faster-rcnn-cplusplus2/blob/master/imgs/result_004545.jpg)
![image](https://github.com/zhanglaplace/faster-rcnn-cplusplus2/blob/master/imgs/result_001150.jpg)
![image](https://github.com/zhanglaplace/faster-rcnn-cplusplus2/blob/master/imgs/result_000456.jpg)
![image](https://github.com/zhanglaplace/faster-rcnn-cplusplus2/blob/master/imgs/result_000542.jpg)
![image](https://github.com/zhanglaplace/faster-rcnn-cplusplus2/blob/master/imgs/result_001763.jpg)

# Something else
 
  if it's helpful to you ,please give me a star thanks~


  

