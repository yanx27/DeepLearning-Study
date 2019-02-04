'''小数据问题'''

'''将图像复制到训练、验证和测试的目录'''
import os, shutil
original_dataset_dir = 'E:\\深度学习\\KerasLearning\\kaggle_original_data\\train' # 原始数据集解压目录的路径
base_dir = 'E:\\深度学习\\KerasLearning\\cats_and_dogs_small' # 保存较小数据集的目录
os.mkdir(base_dir)
# 分别对应划分后的训练、验证和测试的目录
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)
train_cats_dir = os.path.join(train_dir, 'cats') #猫的训练图像目录
os.mkdir(train_cats_dir)
train_dogs_dir = os.path.join(train_dir, 'dogs') # 狗的训练图像目录
os.mkdir(train_dogs_dir)
validation_cats_dir = os.path.join(validation_dir, 'cats') # 猫的验证图像目录
os.mkdir(validation_cats_dir)
validation_dogs_dir = os.path.join(validation_dir, 'dogs') # 狗的验证图像目录
os.mkdir(validation_dogs_dir)
test_cats_dir = os.path.join(test_dir, 'cats') # 猫的测试图像目录
os.mkdir(test_cats_dir)
test_dogs_dir = os.path.join(test_dir, 'dogs') #狗的测试图像目录
os.mkdir(test_dogs_dir)
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)] # 将前 1000 张猫的图像复制到 train_cats_dir
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)

'''构建网络'''
from keras import layers
from keras import models
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2),padding='same'))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()
from keras import optimizers
model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(lr=1e-4),metrics=['acc'])

'''
数据预处理
(1) 读取图像文件。
(2) 将 JPEG 文件解码为 RGB 像素网格。
(3) 将这些像素网格转换为浮点数张量。
(4) 将像素值（0~255 范围内）缩放到 [0, 1] 区间
（正如你所知，神经网络喜欢处理较小的输入值）。
这些步骤可能看起来有点吓人，但幸运的是，Keras 拥有自动完成这些步骤的工具。Keras
有一个图像处理辅助工具的模块，位于 keras.preprocessing.image 。特别地，它包含
ImageDataGenerator 类，可以快速创建 Python 生成器，能够将硬盘上的图像文件自动转换
为预处理好的张量批量。
'''
from keras.preprocessing.image import ImageDataGenerator
# 将所有图像乘以 1/255 缩放
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
# 将所有图像的大小调整为 150×150(因为使用了 binary_crossentropy 损失，所以需要用二进制标签）
train_generator = train_datagen.flow_from_directory(train_dir,target_size=(150, 150),batch_size=20,class_mode='binary')
validation_generator = test_datagen.flow_from_directory(validation_dir,target_size=(150, 150),batch_size=20,class_mode='binary')
'''
利用批量生成器拟合模型

    利用生成器，我们让模型对数据进行拟合。我们将使用 fit_generator 方法来拟合，它
在数据生成器上的效果和 fit 相同。它的第一个参数应该是一个 Python 生成器，可以不停地生
成输入和目标组成的批量，比如 train_generator 。因为数据是不断生成的，所以 Keras 模型
要知道每一轮需要从生成器中抽取多少个样本。这是 steps_per_epoch 参数的作用：从生成
器中抽取 steps_per_epoch 个批量后（即运行了 steps_per_epoch 次梯度下降），拟合过程
将进入下一个轮次。本例中，每个批量包含 20 个样本，所以读取完所有 2000 个样本需要 100
个批量。
'''
history = model.fit_generator(train_generator,steps_per_epoch=100,epochs=30,validation_data=validation_generator,validation_steps=50)

'''保存模型'''
model.save('cats_and_dogs_small_1.h5')

'''绘制训练过程中的损失曲线和精度曲线'''
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

'''ImageDataGenerator 来设置数据增强'''
datagen = ImageDataGenerator(
rotation_range=40, #是角度值（在 0~180 范围内），表示图像随机旋转的角度范围
width_shift_range=0.2, # 是图像在水平方向上平移的范围（相对于总宽度的比例）
height_shift_range=0.2, # 是图像在垂直方向上平移的范围（相对于总高度的比例）
shear_range=0.2, # 是随机错切变换的角度
zoom_range=0.2, # 是图像随机缩放的范围
horizontal_flip=True, # 是随机将一半图像水平翻转
fill_mode='nearest')  # 是用于填充新创建像素的方法，这些新像素可能来自于旋转或宽度/高度平移

'''显示几个随机增强后的训练图像'''
from keras.preprocessing import image # 图像预处理工具的模块
fnames = [os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)]
img_path = fnames[3] # 选择一张图像进行增强
img = image.load_img(img_path, target_size=(150, 150)) # 读取图像并调整大小
x = image.img_to_array(img) # 将其转换为形状 (150, 150, 3) 的 Numpy 数组
x = x.reshape((1,) + x.shape) # 将其形状改变为 (1, 150, 150, 3)
i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    plt.show()
    if i % 4 == 0:
        break #生成随机变换后的图像批量。循环是无限的，因此你需要在某个时刻终止循环

'''定义一个包含 dropout 的新卷积神经网络'''
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
optimizer=optimizers.RMSprop(lr=1e-4),
metrics=['acc'])

'''利用数据增强生成器训练卷积神经网络'''
train_datagen = ImageDataGenerator(
rescale=1./255,
rotation_range=40,
width_shift_range=0.2,
height_shift_range=0.2,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True,)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(train_dir,target_size=(150, 150),batch_size=32,class_mode='binary')
validation_generator = test_datagen.flow_from_directory(validation_dir,target_size=(150, 150),batch_size=32,class_mode='binary')
history = model.fit_generator(train_generator,steps_per_epoch=100,epochs=100,validation_data=validation_generator,validation_steps=50)
model.save('cats_and_dogs_small_2.h5')

'''使用预训练的卷积神经网络'''
# 将 VGG16 卷积基实例化
from keras.applications import VGG16
conv_base = VGG16(weights='imagenet', # 指定模型初始化的权重检查点
                  include_top=False, # 指定模型最后是否包含密集连接分类器。默认情况下，这个密集连接分类器对应于 ImageNet 的 1000 个类别。因为我们打算使用自己的密集连接分类器（只有两个类别： cat 和 dog ），所以不需要包含它。
                  input_shape=(150, 150, 3)) # 是输入到网络中的图像张量的形状。这个参数完全是可选的，如果不传入这个参数，那么网络能够处理任意形状的输入。
conv_base.summary()

'''在卷积基上添加一个密集连接分类器'''
from keras import models
from keras import layers
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()
# 在 Keras 中，冻结网络的方法是将其 trainable 属性设为 False
conv_base.trainable = False
'''利用冻结的卷积基端到端地训练模型'''
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(train_dir,target_size=(150, 150),batch_size=20,class_mode='binary')
validation_generator = test_datagen.flow_from_directory(validation_dir,target_size=(150, 150),batch_size=20,class_mode='binary')
model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(lr=2e-5),metrics=['acc'])
history = model.fit_generator(train_generator,steps_per_epoch=100,epochs=30,validation_data=validation_generator,validation_steps=50)

'''
微调模型
微调网络的步骤如下。
(1) 在已经训练好的基网络（base network）上添加自定义网络。
(2) 冻结基网络。
(3) 训练所添加的部分。
(4) 解冻基网络的一些层。
(5) 联合训练解冻的这些层和添加的部分。
你在做特征提取时已经完成了前三个步骤。我们继续进行第四步：先解冻 conv_base ，然
后冻结其中的部分层。
'''
# 冻结直到某一层的所有层: block5_conv1
conv_base.trainable = True
set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False
# 重新训练
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-5),
              metrics=['acc'])
history = model.fit_generator(train_generator,
                              steps_per_epoch=100,
                              epochs=100,
                              validation_data=validation_generator,
                              validation_steps=50)

'''最后你可以在测试数据上最终评估这个模型了'''
test_generator = test_datagen.flow_from_directory(
test_dir,
target_size=(150, 150),
batch_size=20,
class_mode='binary')
test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
print('test acc:', test_acc) # 97%