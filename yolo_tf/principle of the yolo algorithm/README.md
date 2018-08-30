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
3、YOLO算法的最终输入与输出
-------
* 这里先以[DeepLearning.ai](https://mooc.study.163.com/smartSpec/detail/1001319001.htm)视频中的为例子：
![](https://github.com/yanx27/DeepLearning-Study/blob/master/yolo_tf/principle%20of%20the%20yolo%20algorithm/picture2.png)
* 从上图可以看出，我们假设是做一个只有车辆、行人和摩托车的三对象识别问题（C=3），grid cell个数为3x3，bounding box的种类为2，右边的向量就是每一个grid cell将要对应的值。向量的第一位Pc对应着confidence，bx,by,bh,bh代表着这个bonding box的范围，c1,c2,c3对应着这个grid cell的object是每一个类别的概率值，而整个前八个数值，代表着第一个bounding box，下面的第九到第十六个数的含义与第一个bounding box相同，代表着第二个bonding box的结果。
* 从右边的图可以看出，当这个grid cell没有对象的时候（即Pr(object)=0），底下的那些数值我们并不关心；当底下（绿色格子）中出现了与bouding box 2形状相一致的物体时，向量中代表第二个bounding box的位置有数字，而代表bounding box 1的那些位置Pc为0，故剩下的位置我们不关心。
* 模型预测时也是同理，输出每个格子的向量，其形成的3x3x16的三维矩阵如图中间所示，这便是对于每一张图片其所对应的训练输入或测试输出。
![](https://github.com/yanx27/DeepLearning-Study/blob/master/yolo_tf/principle%20of%20the%20yolo%20algorithm/picture3.png)

#
4、非最大值抑制（NMS）
-------
* 针对某一类别，选择得分最大的bounding box，然后计算它和其它bounding box的IOU值，如果IOU大于0.5（论文中是0.5，实际上也可以设置得更高），说明重复率较大，该得分设为0，如果不大于0.5，则不改；这样一轮后，再选择剩下的score里面最大的那个bounding box，然后计算该bounding box和其它bounding box的IOU，重复以上过程直到最后。最后每个bounding box的C个score取最大的score，如果这个score大于0，那么这个bounding box就是这个socre对应的类别（矩阵的行），如果小于0，说明这个bounding box里面没有物体，跳过即可。<br>
![](https://github.com/yanx27/DeepLearning-Study/blob/master/yolo_tf/principle%20of%20the%20yolo%20algorithm/picture4.png)
![](https://github.com/yanx27/DeepLearning-Study/blob/master/yolo_tf/principle%20of%20the%20yolo%20algorithm/picture5.png)

>注：
>* 由于输出层为全连接层，因此在检测时，YOLO训练模型只支持与训练图像相同的输入分辨率。
>* 虽然每个格子可以预测B个bounding box，但是最终只选择只选择IOU最高的bounding box作为物体检测输出，即每个格子最多只预测出一个物体。当物体占画面比例较小，如图像中包含畜群或鸟群时，每个格子包含多个物体，但却只能检测出其中一个。这是YOLO方法的一个缺陷。
#
5、网络结构
-------
* YOLO检测网络网络方面主要采用GoogLeNet，包括24个卷积层和2个全连接层，卷积层主要用来提取特征，全连接层主要用来预测类别概率和坐标。最后的输出是7x7x30，这里30=20+2x(4+1)，由20个类别,4个bounding box的决定系数和一个Pc得出，7x7是grid cell的数量。这里注意下实现的细节可能人人都不大一样，比如对inception的改动，最后几层的全连接层的改动等等，但是重点在于最后一层的输出是7x7x30。网络结构如图所示：<br>
![](https://pic1.zhimg.com/80/v2-2c4e8576b987236de47f91ad594bf36d_hd.jpg)
> 另外两个小细节：
>* 1、作者先在ImageNet数据集上预训练网络，而且网络只采用fig3的前面20个卷积层，输入是224x224大小的图像。然后在检测的时候再加上随机初始化的4个卷积层和2个全连接层，同时输入改为更高分辨率的448X448。
>* 2、Relu层改为leakyRelu，即当x<0时，激活值是0.1xX，而不是传统的0。
#
6、损失函数
-------
* YOLO使用均方和误差作为loss函数来优化模型参数，即网络输出的SxSx(Bx5+C)维向量与真实图像的对应SxSx(Bx5+C)维向量的均方和误差。如下式所示。<br>
![](https://www.zhihu.com/equation?tex=loss%3D%5Csum_%7Bi%3D0%7D%5E%7BS%5E%7B2%7D+%7D%7BcoordError+%2B+iouError+%2B+classError%7D+)

* YOLO对上式loss的计算进行了如下修正:<br>
![](https://pic2.zhimg.com/80/v2-c629e12fb112f0e3c36b0e5dca60103a_hd.jpg)

* 在loss function中，前面两行表示localization error(即坐标误差)，第一行是box中心坐标(x,y)的预测，第二行为宽和高的预测。这里注意用宽和高的开根号代替原来的宽和高，这样做主要是因为相同的宽和高误差对于小的目标精度影响比大的目标要大。举个例子，原来w=10，h=20，预测出来w=8，h=22，跟原来w=3，h=5，预测出来w1，h=7相比，其实前者的误差要比后者小，但是如果不加开根号，那么损失都是一样：4+4=8，但是加上根号后，变成0.15和0.7。 
* 第三、四行表示bounding box的confidence损失，就像前面所说的，分成grid cell包含与不包含object两种情况。这里注意下因为每个grid cell包含两个bounding box，所以只有当ground truth 和该网格中的某个bounding box的IOU值最大的时候，才计算这项。 
* 第五行表示预测类别的误差，注意前面的系数只有在grid cell包含object的时候才为1。

#
7、损失函数的代码实现
-------
* 训练的时候：输入N个图像，每个图像包含M个object，每个object包含4个坐标（x，y，w，h）和1个label。然后通过网络得到7x7x30大小的三维矩阵。每个1x30的向量前5个元素表示第一个bounding box的4个坐标和1个confidence，第6到10元素表示第二个bounding box的4个坐标和1个confidence。最后20个表示这个grid cell所属类别。注意这30个都是预测的结果。然后就可以计算损失函数的第一、二 、五行。至于第二三行，confidence可以根据ground truth和预测的bounding box计算出的IOU和是否有object的0,1值相乘得到。真实的confidence是0或1值，即有object则为1，没有object则为0。 这样就能计算出loss function的值了。

* 测试的时候：输入一张图像，跑到网络的末端得到7x7x30的三维矩阵，这里虽然没有计算IOU，但是由训练好的权重已经直接计算出了bounding box的confidence。然后再跟预测的类别概率相乘就得到每个bounding box属于哪一类的概率。


#
参考资料
-------
1、https://docs.google.com/presentation/d/1aeRvtKG21KHdD5lg6Hgyhx5rPq_ZOsGjG5rJ1HP7BbA/pub?start=false&loop=false&delayms=3000&slide=id.p 

2、http://blog.csdn.net/tangwei2014/article/details/50915317 

3、https://zhuanlan.zhihu.com/p/25236464 

4、https://zhuanlan.zhihu.com/p/24916786
