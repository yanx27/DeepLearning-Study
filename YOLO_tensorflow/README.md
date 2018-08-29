## YOLO-Tensorflow

* 代码基于[You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/pdf/1506.02640.pdf), 包含训练和测试部分 

* 下在训练数据以及模型参数：
在下载的源码文件夹中新建文件夹：data
在data中分别新建文件夹pascal_VOC、weights。

* 在 weights 放入模型参数：

    链接: https://pan.baidu.com/s/1htt9YBE 密码: ehw2


* 在pascal_voc放入训练数据集，下载链接：

    链接: https://pan.baidu.com/s/1kWshVhl 密码: 89iu


* 可以在yolo文件夹中的config.py文件里调整模型训练参数
若出现错误

```
ResourceExhaustedError (see above for traceback): OOM when allocating tensor with shape[48,28,28,1024]
	 [[Node: yolo/conv_20/Conv2D = Conv2D[T=DT_FLOAT, data_format="NHWC", padding="SAME", strides=[1, 1, 1, 1], use_cudnn_on_gpu=true, _device="/job:localhost/replica:0/task:0/device:GPU:0"](yolo/conv_19/leaky_relu/Maximum, yolo/conv_20/weights/read/_201)]]
	 [[Node: gradients/AddN_12/_323 = _Recv[client_terminated=false, recv_device="/job:localhost/replica:0/task:0/device:CPU:0", send_device="/job:localhost/replica:0/task:0/device:GPU:0", send_device_incarnation=1, tensor_name="edge_3737_gradients/AddN_12", tensor_type=DT_FLOAT, _device="/job:localhost/replica:0/task:0/device:CPU:0"]()]]
```
则需要调小minibatch的大小


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
