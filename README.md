# YOLOv3
This is only around 1000 lines code version of YOLOv3. But it is a very good start to do more research on it. I implemented training, inference, bounding box visualization, pretrained darknet weights loading and several evaluation methods of YOLOv3, while NMS may have some problems. I will update it later.

The model architecture is based on Aladdin Persson's code https://github.com/aladdinpersson/, which is very concise and understandable.  I followed the main thoughts of simplification and implemented it in a more compact way and also implemented all helper functions mostly only using tensor calculation, e.g. mean average precision calculation, non max suppression, etc, which are all in utils.py.

Here is my ugly hand drawing architecture. I hope it is legible:)

![image](https://raw.githubusercontent.com/XingZeng307/YOLOv3/main/results/YOLOv3_architecture.jpeg) 

# Result on PASCAL VOC data

mAP 0.5|Obj accuracy|Class accuracy
--|:--:|--:
82.23%|97.97%|78.55%

This is trained from scratch and evaluated on 98 epochs, iou 0.4 and confidence score 0.2.

![image](https://raw.githubusercontent.com/XingZeng307/YOLOv3/main/results/pascal_voc.png)

Also did training on some fashion data to detect different necklines.

![image](https://raw.githubusercontent.com/XingZeng307/YOLOv3/main/results/fashion_data.png)


# Training

python train.py --img-size 416 --epoches 100 --batch-size 32 --iou-thre 0.4 --conf-thre 0.2

and also change the num_classes and training data info in train.py code, then you can run it. 
I put my pre-trained weights link here. https://drive.google.com/drive/folders/1f8A6aSUK_rmsfGCUx1Xtnepbx4dEnHcZ?usp=sharing

# Reference
[1] Joseph Redmon and Ali Farhadi. Yolov3: An incremental improvement. arXiv preprint arXiv:1804.02767, 2018

[2] https://github.com/pjreddie/darknet/tree/master/src (source code)

[3] https://github.com/ultralytics/yolov3/

[4] https://github.com/aladdinpersson/

[5] https://github.com/yqyao/YOLOv3_Pytorch/

[6] https://github.com/ethanyanjiali/deep-vision/tree/master/YOLO/tensorflow

# Contact
xingzeng@kth.se
