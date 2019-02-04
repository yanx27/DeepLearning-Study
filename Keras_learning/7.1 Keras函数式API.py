'''函数式 API 简介'''
'''
将 Model 对象实例化只用了一个输入张量和一个输出张量。Keras 会在后台检索从 input_tensor 到 
output_tensor 所包含的每一层，并将这些层组合成一个类图的数据结构，即一个 Model。 
'''
from keras.models import Sequential, Model
from keras import layers
from keras import Input
input_tensor = Input(shape=(64,))
x = layers.Dense(32, activation='relu')(input_tensor)
x = layers.Dense(32, activation='relu')(x)
output_tensor = layers.Dense(10, activation='softmax')(x)
model = Model(input_tensor, output_tensor)
model.summary()

'''多输入模型'''
'''参考文本+问题 --> 回答'''
from keras.models import Model
import keras
from keras import layers
from keras import Input
text_vocabulary_size = 10000
question_vocabulary_size = 10000
answer_vocabulary_size = 500
text_input = Input(shape=(None,), dtype='int32', name='text') # 文本输入是一个长度可变的整数序列。注意，你可以选择对输入进行命名
embedded_text = layers.Embedding(text_vocabulary_size, 64)(text_input) # 将输入嵌入长度为 64 的向量
encoded_text = layers.LSTM(32)(embedded_text) #利用 LSTM 将向量编码为单个向量
question_input = Input(shape=(None,),dtype='int32',name='question') #对问题进行相同的处理（使用不同的层实例）
embedded_question = layers.Embedding(question_vocabulary_size, 32)(question_input)
encoded_question = layers.LSTM(16)(embedded_question)
concatenated = layers.concatenate([encoded_text, encoded_question],axis=-1) #将编码后的问题和文本连接起来
answer = layers.Dense(answer_vocabulary_size,activation='softmax')(concatenated)
model = Model([text_input, question_input], answer) #在模型实例化时，指定两个输入和输出
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['acc'])

'''用函数式 API 实现双输入问答模型'''
import numpy as np
num_samples = 1000
max_length = 100
text = np.random.randint(1, text_vocabulary_size,size=(num_samples, max_length)) # 生成虚构的 Numpy数据
question = np.random.randint(1, question_vocabulary_size,size=(num_samples, max_length))
answers = np.random.randint(answer_vocabulary_size, size=(num_samples))
answers = keras.utils.to_categorical(answers, answer_vocabulary_size) # 回答是 one-hot 编码的，不是整数
model.fit([text, question], answers, epochs=10, batch_size=128) # 使用输入组成的列表来拟合
model.fit({'text': text, 'question': question}, answers,epochs=10, batch_size=128) #使用输入组成的字典来拟合（只有对输入进行命名之后才能用这种方法）

'''多输出模型'''
from keras import layers
from keras import Input
from keras.models import Model
vocabulary_size = 50000
num_income_groups = 10
posts_input = Input(shape=(None,), dtype='int32', name='posts')
embedded_posts = layers.Embedding(256, vocabulary_size)(posts_input)
x = layers.Conv1D(128, 5, activation='relu')(embedded_posts)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dense(128, activation='relu')(x)
# 输出层都具有名称
age_prediction = layers.Dense(1, name='age')(x)
income_prediction = layers.Dense(num_income_groups,activation='softmax',name='income')(x)
gender_prediction = layers.Dense(1, activation='sigmoid', name='gender')(x)
model = Model(posts_input,[age_prediction, income_prediction, gender_prediction])

'''多输出模型的编译选项：多重损失'''
# 与上述写法等效（只有输出层具有名称时才能采用这种写法）
model.compile(optimizer='rmsprop',loss=['mse', 'categorical_crossentropy', 'binary_crossentropy'])
model.compile(optimizer='rmsprop',loss={'age': 'mse','income': 'categorical_crossentropy','gender': 'binary_crossentropy'})
'''多输出模型的编译选项：损失加权'''
model.compile(optimizer='rmsprop',
              loss=['mse', 'categorical_crossentropy', 'binary_crossentropy'],
              loss_weights=[0.25, 1., 10.])
model.compile(optimizer='rmsprop',
              loss={'age': 'mse','income': 'categorical_crossentropy','gender': 'binary_crossentropy'},
              loss_weights={'age': 0.25,'income': 1.,'gender': 10.})

'''将数据输入到多输出模型中

#假设 age_targets 、income_targets 和gender_targets 都是 Numpy 数组
model.fit(posts, [age_targets, income_targets, gender_targets],
          epochs=10,
          batch_size=64)
#与上述写法等效（只有输出层具有名称时才能采用这种写法）
model.fit(posts, {'age': age_targets,'income': income_targets,'gender': gender_targets},
          epochs=10,
          batch_size=64)
'''

''' Inception 模块'''
from keras import layers
def Inception_block(x):
    '''完整的Inception V3架构内置于Keras中，位置在 keras.applications.inception_v3.
       InceptionV3 ，其中包括在 ImageNet 数据集上预训练得到的权重。'''
    branch_a = layers.Conv2D(128, 1, activation='relu', strides=2)(x)
    branch_b = layers.Conv2D(128, 1, activation='relu')(x)
    branch_b = layers.Conv2D(128, 3, activation='relu', strides=2)(branch_b)
    branch_c = layers.AveragePooling2D(3, strides=2)(x)
    branch_c = layers.Conv2D(128, 3, activation='relu')(branch_c)
    branch_d = layers.Conv2D(128, 1, activation='relu')(x)
    branch_d = layers.Conv2D(128, 3, activation='relu')(branch_d)
    branch_d = layers.Conv2D(128, 3, activation='relu', strides=2)(branch_d)
    output = layers.concatenate([branch_a, branch_b, branch_c, branch_d], axis=-1)
    return output

'''残差连接'''
def residul_block(x):
    from keras import layers
    y = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    y = layers.Conv2D(128, 3, activation='relu', padding='same')(y)
    y = layers.Conv2D(128, 3, activation='relu', padding='same')(y)
    # 使用 1×1 卷积，将原始 x 张量线性下采样为与 y 具有相同的形状
    residual = layers.Conv2D(128, 1, strides=2, padding='same')(x)
    y = layers.add([y, residual]) #将残差张量与输出特征相加
    return y


'''将模型化为层——使用双摄像头作为输入的视觉模型：两个平行的摄像头，可以感知深度'''
from keras import layers
from keras import applications
from keras import Input
# 图像处理基础模型是 Xception 网络（只包括卷积基）
xception_base = applications.Xception(weights=None,include_top=False)
left_input = Input(shape=(250, 250, 3))
right_input = Input(shape=(250, 250, 3))
# 对相同的视觉模型调用两次
left_features = xception_base(left_input)
right_input = xception_base(right_input)
# 合并后的特征包含来自左右两个视觉输入中的信息
merged_features = layers.concatenate([left_features, right_input], axis=-1)

