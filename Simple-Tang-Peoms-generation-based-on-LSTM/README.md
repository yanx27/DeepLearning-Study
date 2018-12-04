基于Tensorflow唐诗及歌词生成
====  
#
训练数据
-------
>* 为txt格式的歌词和诗歌，每一行为一个样本

参数设置
-------
>* 测试参数：<br> 
```
python main.py -w poem --no-train
```
>* 训练参数：<br> 
```
python main.py -w poem --train
```

存在问题
-------
>* 为了解决不同诗歌长度不一的问题，在训练前引用空格来填充，导致在预测过程中有可能会预测出空格，从而使得无法输出诗歌

训练结果
-------
>* 输入第一个字，程序自动输出整首诗歌<br> 
```
[INFO] write tang poem...
输入起始字:颜
runing
runing
runing
runing
runing
runing
runing
runing
runing
runing
runing
颜疏到头者，笑子屐难是。
有人须载水，此日如迷息。
```
