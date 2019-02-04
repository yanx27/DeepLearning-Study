'''Keras处理文本数据'''
'''用 Keras 实现单词级的 one-hot 编码'''
from keras.preprocessing.text import Tokenizer
samples = ['The cat sat on the mat.', 'The dog ate my homework.']
tokenizer = Tokenizer(num_words=1000) # 创建一个分词器（tokenizer），设置为只考虑前 1000 个最常见的单词
tokenizer.fit_on_texts(samples) # 构建单词索引
sequences = tokenizer.texts_to_sequences(samples) # 将字符串转换为整数索引组成的列表
one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary') # 也可以直接得到 one-hot 二进制表示。这个分词器也支持除 one-hot 编码外的其他向量化模式
word_index = tokenizer.word_index # 找回单词索引
print('Found %s unique tokens.' % len(word_index))

'''词嵌入'''
# 将一个 Embedding 层实例化，Embedding 层至少需要两个参数：标记的个数（这里是 1000，即最大单词索引 +1）和嵌入的维度（这里是 64）
from keras.layers import Embedding
embedding_layer = Embedding(1000, 64)
'''
我们将这个想法应用于你熟悉的 IMDB 电影评论情感预测任务。首先，我们需要快速准备
数据。将电影评论限制为前 10 000 个最常见的单词（第一次处理这个数据集时就是这么做的），
然后将评论长度限制为只有 20 个单词。对于这 10 000 个单词，网络将对每个词都学习一个 8
维嵌入，将输入的整数序列（二维整数张量）转换为嵌入序列（三维浮点数张量），然后将这个
张量展平为二维，最后在上面训练一个 Dense 层用于分类。
'''
# 加载 IMDB 数据，准备用于 Embedding 层
from keras.datasets import imdb # 数据集
from keras import preprocessing
max_features = 10000 # 作为特征的单词个数
maxlen = 20 # 在这么多单词后截断文本（这些单词都属于前 max_features 个最常见的单词）
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features) # 将数据加载为整数列表
# 将整数列表转换成形状为 (samples,maxlen) 的二维整数张量
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)
'''在 IMDB 数据上使用 Embedding 层和分类器'''
from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding
model = Sequential()
model.add(Embedding(10000, 8, input_length=maxlen)) #指定 Embedding 层的最大输入长度，以便后面将嵌入输入展平。 Embedding 层激活的形状为 (samples, maxlen, 8)
model.add(Flatten()) # 将三维的嵌入张量展平成形状为 (samples, maxlen * 8) 的二维张量
model.add(Dense(1, activation='sigmoid')) # 在上面添加分类器
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()
history = model.fit(x_train, y_train,epochs=10,batch_size=32,validation_split=0.2)

'''使用预训练的词嵌入—— GloVe和Word2Vec'''
'''下载原始 IMDB 数据集并解压,将训练评论转换成字符串列表，每个字符串对应一条评论'''
import os
imdb_dir = 'E:\\深度学习\\KerasLearning\\aclImdb'
train_dir = os.path.join(imdb_dir, 'train')
labels = []
texts = []
for label_type in ['neg', 'pos']:
    dir_name = os.path.join(train_dir, label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname),encoding='UTF-8')
        texts.append(f.read())
        f.close()
        if label_type == 'neg':
            labels.append(0)
        else:
            labels.append(1)
'''对 IMDB 原始数据的文本进行分词'''
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
maxlen = 100 # 在 100 个单词后截断评论
training_samples = 200 # 在 200 个样本上训练
validation_samples = 10000 # 词在 200 个样本上训练
max_words = 10000 # 只考虑数据集中前 10 000 个最常见的单词
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=maxlen)
labels = np.asarray(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)
indices = np.arange(data.shape[0]) # 将数据划分为训练集和验证集，但首先要打乱数据，因为一开始数据中的样本是排好序的（所有负面评论都在前面，然后是所有正面评论）
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]

'''解析 GloVe 词嵌入文件'''
glove_dir = 'E:\\深度学习\\KerasLearning\\glove.6B'
embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'),encoding='UTF-8') # xxd代表output的维度
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))

'''
准备 GloVe 词嵌入矩阵

接下来，需要构建一个可以加载到 Embedding 层中的嵌入矩阵。它必须是一个形状为
(max_words, embedding_dim) 的矩阵，对于单词索引（在分词时构建）中索引为 i 的单词，
这个矩阵的元素 i 就是这个单词对应的 embedding_dim 维向量。注意，索引 0 不应该代表任何
单词或标记，它只是一个占位符。
'''
embedding_dim = 100
embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector # 嵌入索引（ embeddings_index ）中找不到的词，其嵌入向量全为 0

'''模型架构　'''
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

'''在模型中加载 GloVe 嵌入'''
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False
'''训练与评估'''
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_val, y_val))
model.save_weights('pre_trained_glove_model.h5')