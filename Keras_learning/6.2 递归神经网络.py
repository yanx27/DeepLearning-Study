'''RNN对电影影评进行分类'''
'''准备 IMDB 数据'''
from keras.datasets import imdb
from keras.preprocessing import sequence
max_features = 10000 # 作为特征的单词个数
maxlen = 500 # 在这么多单词之后截断文本（这些单词都属于前 max_features 个最常见的单词）
batch_size = 32
print('Loading data...')
(input_train, y_train), (input_test, y_test) = imdb.load_data(
num_words=max_features)
print(len(input_train), 'train sequences')
print(len(input_test), 'test sequences')
print('Pad sequences (samples x time)')
input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)
print('input_train shape:', input_train.shape)
print('input_test shape:', input_test.shape)

'''用 Embedding 层和 SimpleRNN 层来训练模型'''
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN
'''
一种是返回每个时间步连续输出的完整序列，即形状为 (batch_size, timesteps, output_features)
的三维张量；另一种是只返回每个输入序列的最终输出，即形状为 (batch_size, output_features) 
的二维张量。这两种模式由 return_sequences 这个构造函数参数来控制
'''
model = Sequential()
model.add(Embedding(max_features, 32))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(input_train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2)

''' LSTM 层和 GRU 层'''
from keras.layers import LSTM
model = Sequential()
model.add(Embedding(max_features, 32))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(input_train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2)
