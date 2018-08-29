## 基于Tensorflow的YOLO算法

代码基于[YOLO](https://arxiv.org/pdf/1506.02640.pdf), 包含训练和测试部分 

1. 下在训练数据以及模型参数：
>在下载的源码文件夹中新建文件夹：data
>在data中分别新建文件夹pascal_VOC、weights。目录结构如下图所示:


2. 下载[YOLO_small](https://drive.google.com/file/d/0B5aC8pI-akZUNVFZMmhmcVRpbTA/view?usp=sharing)
将其放在weights文件夹中

4. 可以在yolo文件夹中的config.py文件里调整模型训练参数`

5. 训练
	```Shell
	$ python train.py
	```

6. 测试
	```Shell
	$ python test.py
	```

### 所用的库版本
1. Tensorflow 1.14.1

2. OpenCV 3.0
