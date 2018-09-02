ResNet (谷歌人脸识别系统)
====  
#
* 参考文献为Florian Schroff的 [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832) 

#
1、简介：
-------
* 近年来，人脸识别技术取得了飞速的进展，但是人脸验证和识别在自然条件中应用仍然存在困难。本文中，作者开发了一个新的人脸识别系统：FaceNet，可以直接将人脸图像映射到欧几里得空间，空间距离的长度代表了人脸图像的相似性。只要该映射空间生成，人脸识别，验证和聚类等任务就可以轻松完成。文章的方法是基于深度卷积神经网络。FaceNet在LFW数据集上，准确率为0.9963，在YouTube Faces DB数据集上，准确率为0.9512。
* FaceNet是一个通用的系统，可以用于人脸验证（是否是同一人？），识别（这个人是谁？）和聚类（寻找类似的人？）。FaceNet采用的方法是通过卷积神经网络学习将图像映射到欧几里得空间。空间距离直接和图片相似度相关：同一个人的不同图像在空间距离很小，不同人的图像在空间中有较大的距离。只要该映射确定下来，相关的人脸识别任务就变得很简单。
* 当前存在的基于深度神经网络的人脸识别模型使用了分类层（classification layer）：中间层为人脸图像的向量映射，然后以分类层作为输出层。这类方法的弊端是不直接和效率低。
* 与当前方法不同，FaceNet直接使用基于triplets的LMNN（最大边界近邻分类）的loss函数训练神经网络，网络直接输出为128维度的向量空间。我们选取的triplets（三联子）包含两个匹配脸部缩略图和一个非匹配的脸部缩略图，loss函数目标是通过距离边界区分正负类，如图所示:<br>
![](https://pic2.zhimg.com/80/v2-ced157b8ca1fa96603c30b651eb2e1e0_hd.jpg)

#
2、方法概述：
-------
上图步骤可以描述为：

* 前面部分采用一个CNN结构提取特征，
* CNN之后接一个特征归一化（使其特征的||f(x)||2=1,这样子，所有图像的特征都会被映射到一个超球面上)，
* 再接入一个embedding层(嵌入函数)，嵌入过程可以表达为一个函数，即把图像x通过函数f映射到d维欧式空间。
* 此外，作者对嵌入函数f(x)的值，即值阈，做了限制。使得x的映射f(x)在一个超球面上。
* 接着，再去优化这些特征，而文章这里提出了一个新的损失函数，triplet损失函数(优化函数），而这也是文章最大的特点所在。

#
3.1、Triplet Loss(三元组损失函数)
-------
* 什么是Triplet Loss呢？故名思意，也就是有三张图片输入的Loss（之前的都是Double Loss或者是SingleLoss）。本文通过LDA思想训练分类模型，使得类内特征间隔小，类间特征间隔大。为了保证目标图像 与类内图片(正样本)特征距离小，与类间图片(负样本)特征距离大。需要Triplet损失函数来实现。<br>
![](https://pic2.zhimg.com/80/v2-e97dea2c74c31b53803925294983b7c8_hd.jpg)
* 根据上文，可以构建一个约束条件：<br>
![](https://pic2.zhimg.com/80/v2-fb0de06aa80bfd4bb6eb9a24f9855c6b_hd.jpg)
![](https://pic3.zhimg.com/80/v2-89f6cb30446edc2f7748ed0541d1aeba_hd.jpg)

> 其中，alpha为positive/negtive的边界, (1)表示：左边类内的距离（加上边际）要小于右边类间的距离，这个约束需要在所有的三元组上都成立。

#
3.2、triplets筛选
-------
* 在上面中，如果严格的按照上式来进行学习的话，它的T(穷举所有的图像3元组)是非常大的。举个例子：在一个1000个人，每人有20张图片的情况下，其T=(1000x20)x19x(20x999)(总图片数x每个图片类内组合x每个图片类间组合),也就是O(T)=N^2 ，所以，穷举是不大现实的。那么，我们只能从这所有的 N^2个中选择部分来进行训练。现在问题来了，怎么从这么多的图像中挑选呢？答案是选择最难区分的图像对。

* 给定一张人脸图片，我们要挑选：<br>
    1.一张hard positive：即在类内的另外19张图像中，跟它最不相似的图片。(正样本里面最差的样本)<br>
    2.一张hard negative：即在类间的另外20x999图像中，跟它最为相似的图片。(负样本里面最差的样本)<br>
    挑选hard positive 和hard negative有两种方法，offline和online方法，具体的差别只是在训练上。<br>
* triplets 的选择对模型的收敛非常重要。如公式1所示，对于<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;x_{i}^{a}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\inline&space;x_{i}^{a}" title="x_{i}^{a}" /></a> ，我们我们需要选择同一个体的不同图片<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;x_{i}^{p}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\inline&space;x_{i}^{p}" title="x_{i}^{p}" /></a> ，使<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;argmax_{x_{i}^{p}&space;}&space;\left|&space;\left|&space;f(x_{i}^{a}&space;)-f(x_{i}^{p}&space;)\right|&space;\right|&space;_{2}^{2}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\inline&space;argmax_{x_{i}^{p}&space;}&space;\left|&space;\left|&space;f(x_{i}^{a}&space;)-f(x_{i}^{p}&space;)\right|&space;\right|&space;_{2}^{2}" title="argmax_{x_{i}^{p} } \left| \left| f(x_{i}^{a} )-f(x_{i}^{p} )\right| \right| _{2}^{2}" /></a> ；同时，还需要选择不同个体的图片<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;x_{i}^{n}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\inline&space;x_{i}^{n}" title="x_{i}^{n}" /></a> ，使得<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;argmin_{x_{i}^{n}&space;}&space;\left|&space;\left|&space;f(x_{i}^{a}&space;)-f(x_{i}^{n}&space;)\right|&space;\right|&space;_{2}^{2}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\inline&space;argmin_{x_{i}^{n}&space;}&space;\left|&space;\left|&space;f(x_{i}^{a}&space;)-f(x_{i}^{n}&space;)\right|&space;\right|&space;_{2}^{2}" title="argmin_{x_{i}^{n} } \left| \left| f(x_{i}^{a} )-f(x_{i}^{n} )\right| \right| _{2}^{2}" /></a>。在实际训练中，跨越所有训练样本来计算argmin和argmax是不现实的，还会由于错误标签图像导致训练收敛困难。
* 在整个训练集上寻找argmax和argmin是困难的。如果找不到，会使训练变得困难,难以收敛，例如错误的打标签。因此需要采取两种显而易见的方法避免这个问题：：<br>
一，离线更新三元组(每隔n步)，采用最近的网络模型的检测点 并计算数据集的子集的argmin和argmax(局部最优)。。<br>
二，在线更新三元组。在mini-batch上 选择不好的正(类内)/负(类间)训练模型。(一个mini-batch可以训练出一个子模型)。<br>
本文中，我们采用在线生成triplets的方法。我们选择了大样本的mini-batch（1800样本/batch）来增加每个batch的样本数量。每个mini-batch中，我们对单个个体选择40张人脸图片作为正样本，随机筛选其它人脸图片作为负样本。
* 实际采用方法：<br>
    1.采用在线的方式 (作者说，在线+不在线方法结果不确定)<br>
    2.在mini-batch中挑选所有的anchor positive 图像对 (因为实际操作时，发现这样训练更稳定)<br>
    3.依然选择最为困难的anchor negative图像对 (可以提前发现不好的局部最小值)<br>
* 选择最为困难的负样本，在实际当中，容易导致在训练中很快地陷入局部最优，或者说整个学习崩溃f(x)=0。为了避免，我们采用如下公式来帮助筛选负样本：<br>
![](https://pic3.zhimg.com/80/v2-2af5cd0d92a4ab587cf44db2b463241a_hd.jpg)
> 左边：Positive pair的欧式距离; 右边：negative pair的欧式距离,把这一个约束叫作semi-hard(半序关系)。因为虽然这些negative pair的欧式距离远小于 Positive pair的欧式距离，但是negative pair的欧式距离的平方接近于Positive pair的欧式距离的平方。
* 总结：以上所有过程博概括为：为了快速收敛模型-->需要找到训练的不好的mini-batch上的差模型(负样本)-->从而找到不满足约束条件/使损失增大的三元组
