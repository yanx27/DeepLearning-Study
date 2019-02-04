''' 第一个神经网络示例'''

'''加载 Keras 中的 MNIST 数据集'''
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
'''网络架构'''
from keras import models
from keras import layers
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))
'''编译步骤'''
network.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

'''
准备图像数据:
在开始训练之前，我们将对数据进行预处理，将其变换为网络要求的形状，并缩放到所有值都在 [0, 1] 区间。
比如，之前训练图像保存在一个 uint8 类型的数组中，其形状为(60000, 28, 28) ，取值区间为 [0, 255] 。
我们需要将其变换为一个 float32 数组，其形状为 (60000, 28 * 28) ，取值范围为 0~1
'''
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

'''准备标签'''
from keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

'''在 Keras 中这一步是通过调用网络的 fit 方法来完成的我们在训练数据上拟合（fit）模型'''
network.fit(train_images, train_labels, epochs=5, batch_size=128)

'''我们很快就在训练数据上达到了 0.989（98.9%）的精度。现在我们来检查一下模型在测试集上的性能'''
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)

'''显示第 4 个数字'''
digit = train_images[4]
import matplotlib.pyplot as plt
plt.imshow(digit.reshape((28,28)), cmap=plt.cm.binary)
plt.show()