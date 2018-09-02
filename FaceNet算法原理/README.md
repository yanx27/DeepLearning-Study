ResNet (谷歌人脸识别系统)
====  
#
* 参考文献为Florian Schroff, Dmitry Kalenichenko, James Philbin的 [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832) 

#
简介：
-------
* 近年来，人脸识别技术取得了飞速的进展，但是人脸验证和识别在自然条件中应用仍然存在困难。本文中，作者开发了一个新的人脸识别系统：FaceNet，可以直接将人脸图像映射到欧几里得空间，空间距离的长度代表了人脸图像的相似性。只要该映射空间生成，人脸识别，验证和聚类等任务就可以轻松完成。文章的方法是基于深度卷积神经网络。FaceNet在LFW数据集上，准确率为0.9963，在YouTube Faces DB数据集上，准确率为0.9512。

