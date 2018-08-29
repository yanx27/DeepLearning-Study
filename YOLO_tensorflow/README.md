## 基于Tensorflow的YOLO算法

* 代码基于[You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/pdf/1506.02640.pdf), 包含训练和测试部分 

* 下在训练数据以及模型参数：
在下载的源码文件夹中新建文件夹：data
在data中分别新建文件夹pascal_VOC、weights。

* 在 weights 放入模型参数：

链接: https://pan.baidu.com/s/1htt9YBE 密码: ehw2


* 在pascal_voc放入训练数据集，下载链接：

链接: https://pan.baidu.com/s/1kWshVhl 密码: 89iu


* 可以在yolo文件夹中的config.py文件里调整模型训练参数


* 训练
	```Shell
	$ python train.py
	```

	* 测试
	```Shell
	$ python test.py
	```

### 所用的库版本
1. Tensorflow 1.14.1

2. OpenCV 3.0
