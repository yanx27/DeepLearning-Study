YOLO算法原理（YOU ONLY LOOK ONCE）
====  
#
* 参考文献为 [You only look once unified real-time object detection](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf) 
#
1、创新
-------
* 作者在YOLO算法中把物体检测（object detection）问题处理成回归问题，用一个卷积神经网络结构就可以从输入图像直接预测bounding box和类别概率。
* YOLO算法的优点：
>* 1、YOLO的速度非常快。在Titan X GPU上的速度是45 fps（frames per second），加速版的YOLO差不多是150fps。
>* 2、YOLO是基于图像的全局信息进行预测的。这一点和基于sliding window以及region proposal等检测算法不一样。与Fast R-CNN相比，YOLO在误检测（将背景检测为物体）方面的错误率能降低一半多。
>* 3、YOLO可以学到物体的generalizable representations。可以理解为泛化能力强。
>* 4、准确率高，有实验证明。* 在论文中，通过引入一个深度残差学习框架来解决退化问题（degradation，就是指非常深的网络的性能反而比较浅的网络差，而且越深越差）。

#
2、核心思想
-------
* YOLO将输入图像分成 SxS个格子（grid cell），每个格子负责检测落入该格子的物体。若某个物体的中心位置的坐标落入到某个格子，那么这个格子就负责检测出这个物体。如下图所示，图中物体狗的中心点（红色原点）落入第5行、第2列的格子内，所以这个格子负责预测图像中的物体狗。每个格子输出B个bounding box（包含物体的矩形区域）信息，以及C个物体属于哪个类别的概率信息。<br>
![](https://pic1.zhimg.com/80/v2-4b3c159386ae24809aa6721cf307df30_hd.jpg)
* 每个格子都预测C个假定类别的概率。在本文中作者取S=7（最终输出7x7的格子），B=2（使用两种bounding box进行计算），C=20（因为PASCAL VOC有20个类别），所以最后全连接层的输出有7x7x30个tensor。<br>
![](https://github.com/yanx27/DeepLearning-Study/blob/master/yolo_tf/principle%20of%20the%20yolo%20algorithm/principle1.jpg)
* Bounding box信息包含5个数据值，分别是x,y,w,h,和confidence。其中x,y是指当前格子预测得到的物体的bounding box的中心位置的坐标（其将每个格子的左上角定义为(0,0)，右下角定义为(1,1)，故x和y是取值为0和1之间的数）。w,h是bounding box的宽度和高度。（其将w和h归一化，用其与每个格子的宽度之比来表示，因此若其bounding box宽度或高度大于格子的宽度，其可能大于1）
* confidence反映当前bounding box是否包含物体以及物体位置的准确性，计算方式如下：<br>


>     confidence = Pr(object) * IOU
> 其中，若bounding box包含物体，则Pr(object)=1；否则Pr(object)=0。IOU (intersection over union)为预测bounding box与物体真实区域的交集面积占并集面积的比例。

* 每个bounding box都对应一个confidence score，如果grid cell里面没有object，confidence就是0；如果有，则confidence score等于预测的box包含物体与否的值和其box中IOU值的乘积，见上面公式。如果一个object的中心点坐标在一个grid cell中，那么这个grid cell就是包含这个object，也就是说这个object的预测就由该grid cell负责。 每个grid cell都预测C个类别概率，表示一个grid cell在包含object的条件下属于某个类别的概率
