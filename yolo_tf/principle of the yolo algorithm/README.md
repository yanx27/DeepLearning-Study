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
> 其中，若bounding box包含物体，则Pr(object)=1；否则Pr(object)=0。交并比IOU (intersection over union)为预测bounding box与物体真实区域的交集面积占并集面积的比例，一般在目标检测任务中，约定如果 IoU⩾0.5 ，那么就说明检测正确。当然标准越大，则对目标检测算法越严格。得到的IoU值越大越好。

* 每个bounding box都对应一个confidence score，如果grid cell里面没有object，confidence就是0（在其为0的时候，损失函数不关注x,y,w,h，故它们的值不重要）；如果有，则confidence score等于预测的box包含物体与否的值和其box中IOU值的乘积，见上面公式。如果一个object的中心点坐标在一个grid cell中，那么这个grid cell就是包含这个object，也就是说这个object的预测就由该grid cell负责。 
* 同样，每个grid cell都预测C个类别概率，表示一个grid cell在包含object的条件下属于某个类别的概率。因此其得到每个bounding box属于哪一类的confidence score（若其 Pr(object)=0，则不关心其概率的大小），也就是说最后会得到20x（7x7x2）=20x98个score矩阵，括号里面2是bounding box的种类数（应用一个横的和一个竖的好识别不同形状的物体），7x7是grid cell的数量，20代表类别数。作者开源出的YOLO代码中，全连接层输出特征向量各维度对应内容如下：<br>
![](https://pic3.zhimg.com/80/v2-1098c1152f55d73a859f20bae3d9bb1e_hd.jpg)

#
3、YOLO算法最终如何进行目标检测
-------
* 这里先以DeepLearning.ai视频中的为例子：
![](https://github.com/yanx27/DeepLearning-Study/blob/master/yolo_tf/principle%20of%20the%20yolo%20algorithm/picture2.png)
* 从上图可以看出，我们假设是做一个只有车辆、行人和摩托车的三对象识别问题（C=3），grid cell个数为3x3，bounding box的种类为2，右边的向量就是每一个grid cell将要对应的值。向量的第一位Pc对应着confidence，bx,by,bh,bh代表着这个bonding box的范围，c1,c2,c3对应着这个grid cell的object是每一个类别的概率值，而整个前八个数值，代表着第一个bounding box，下面的第九到第十六个数的含义与第一个bounding box相同，代表着第二个bonding box的结果。
* 从右边的图可以看出，当这个grid cell没有对象的时候（即Pr(object)=0），底下的那些数值我们并不关心；当底下（绿色格子）中出现了与bouding box 2形状相一致的物体时，向量中代表第二个bounding box的位置有数字，而代表bounding box 1的那些位置Pc为0，故剩下的位置我们不关心。
* 模型预测时也是同理，输出每个格子的向量，其形成的3x3x16的三维矩阵如图中间所示，这边是对于每一张图片其所对应的训练输入或测试输出。
![](https://github.com/yanx27/DeepLearning-Study/blob/master/yolo_tf/principle%20of%20the%20yolo%20algorithm/picture3.png)
