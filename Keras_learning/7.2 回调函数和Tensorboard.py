'''Callback和Tensorboard'''

'''ModelCheckpoint 与 EarlyStopping'''
import keras
callbacks_list = [keras.callbacks.EarlyStopping(monitor='acc', # 如果acc不再改善，就中断训练
                                                patience=1,), # 如果精度在多于一轮的时间（即两轮）内不再改善，中断训练
                  # 在每轮过后保存当前权重
                  keras.callbacks.ModelCheckpoint(filepath='my_model.h5', #保存路径
                                                  # 这两个参数的含义是，如果 val_loss 没有改善，那么不需要覆
                                                  # 盖模型文件。这就可以始终保存在训练过程中见到的最佳模型
                                                  monitor='val_loss',
                                                  save_best_only=True,)
                  ]
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
model.fit(x, y,
          epochs=10,
          batch_size=32,
          # 由于回调函数要监控验证损失和验证精度，
          # 所以在调用 fit 时需要传入 validation_data （验证数据）
          callbacks=callbacks_list,
          validation_data=(x_val, y_val)
          )

'''ReduceLROnPlateau——如果验证损失不再改善，你可以使用这个回调函数来降低学习率'''
import keras
callbacks_list = [keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                    factor=0.1,#触发时将学习率除以 10
                                                    patience=10) #如果验证损失在 10 轮内都没有改善，那么就触发这个回调函数)
                  ]
model.fit(x, y,epochs=10,batch_size=32,
          callbacks=callbacks_list,
          validation_data=(x_val, y_val))


'''使用了 TensorBoard 的文本分类模型'''\
'''建立模型'''
import keras
from keras import layers
from keras.datasets import imdb
from keras.preprocessing import sequence
max_features = 2000
max_len = 500
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)
model = keras.models.Sequential()
model.add(layers.Embedding(max_features, 128,input_length=max_len,name='embed'))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(1))
model.summary()
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
'''使用一个 TensorBoard 回调函数来训练模型'''
callbacks = [
keras.callbacks.TensorBoard(log_dir='my_log_dir',
                            histogram_freq=1,#每一轮之后记录激活直方图
                            embeddings_freq=1 #每一轮之后记录嵌入数据
                            )
]
history = model.fit(x_train, y_train,epochs=20,batch_size=128,
                    validation_split=0.2,
                    callbacks=callbacks)
'''将模型绘制为层组成的图'''
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
from keras.utils import plot_model
plot_model(model, show_shapes=True, to_file='model.png')
