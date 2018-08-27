import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
import tensorflow as tf




def load_dataset():
    os.chdir('E:\\深度学习\\吴恩达深度学习课程\\代码\\卷积神经网络\\CNN Application\\datasets')
    train_dataset = h5py.File('train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
# Example of a picture
index = np.random.randint(0, X_train_orig.shape[0]-1)
plt.imshow(X_train_orig[index])
plt.show()
print ("y = " + str(np.squeeze(Y_train_orig[:, index])))

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y
X_train = X_train_orig/255.
X_test = X_test_orig/255.
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T
print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape)) #(1080, 64, 64, 3)
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

#每个批次的大小
batch_size = 128
#计算一共有多少个批次
n_batch = X_train.shape[0] // batch_size
# mini-batch的建立
def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    m = X.shape[0]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :, :, :]
    shuffled_Y = Y[permutation, :]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = int(np.floor(m / mini_batch_size))  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m, :, :, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

def create_placeholders(n_H0,n_W0,n_C0,ny):
    X = tf.placeholder(tf.float32,shape=[None,n_H0,n_W0,n_C0])
    Y = tf.placeholder(tf.float32,shape=[None,ny])
    return X,Y
def weight_variabkle(shape):
    initial = tf.truncated_normal(shape,stddev = 0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
def conv2d(x,W,stride=1):
    return tf.nn.conv2d(x,W,strides=[1,stride,stride,1],padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
def average_pool_6x6(x):
    return tf.nn.avg_pool(x,ksize=[1,6,6,1],strides=[1,6,6,1],padding='SAME')
# 占位符
X, Y = create_placeholders(64,64,3,6) #(1080, 64, 64, 3)
# 第一层卷积层的变量
W_conv1 = weight_variabkle([5,5,3,16])
b_conv1 = bias_variable([16])
# 第一层卷积层和池化层
h_conv1 = tf.nn.relu(conv2d(X,W_conv1,stride=2) + b_conv1) #(1080,64,64,16)
h_pool1 = max_pool_2x2(h_conv1)
# 第二层卷积层的变量
W_conv2 = weight_variabkle([3,3,16,32])
b_conv2 = bias_variable([32])
# 第二层卷积层和池化层
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2) #(1080,32,32,32)
h_pool2 = max_pool_2x2(h_conv2)
# 第三层卷积层的变量
W_conv3 = weight_variabkle([3,3,32,64])
b_conv3 = bias_variable([64])
# 第三层卷积层和池化层
h_conv3 = tf.nn.relu(conv2d(h_pool2,W_conv3) + b_conv3) #(1080,16,16,64)
h_pool3 = max_pool_2x2(h_conv3)
# 将卷积层的输出拉成一列向量
h_flatten = tf.contrib.layers.flatten(h_pool3) #(1080,16*16*64)
# 利用keep_prob来存放dropout中神经元的存留概率
keep_prob = tf.placeholder(tf.float32)
# 第一层全连接层
h_fc1 = tf.contrib.layers.fully_connected(h_flatten, 500, activation_fn=tf.nn.relu)
h_fc1_dropout = tf.nn.dropout(h_fc1,keep_prob) # dropout
# 第二层全连接层
h_fc2 = tf.contrib.layers.fully_connected(h_fc1_dropout, 100, activation_fn=tf.nn.relu)
h_fc2_dropout = tf.nn.dropout(h_fc2,keep_prob) # dropout
# 第三层全连接层
h_fc3 = tf.contrib.layers.fully_connected(h_fc2_dropout, 6, activation_fn = tf.nn.softmax)
# 交叉熵代价函数
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=h_fc3,labels=Y))
# 利用Adam进行优化
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# 结果存放在一个Bool型的列表中
correct_prediction = tf.equal(tf.argmax(h_fc3,1),tf.argmax(Y,1)) #返回张量中最大值所在的位置
accurracy = tf.reduce_mean(tf.cast(correct_prediction,'float'))
# 保存模型
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    loss = []
    accurracy_list = []
    for epoch in range(1000):
        minibatches = random_mini_batches(X_train, Y_train, batch_size)
        for minibatch in minibatches:
            minibatch_X, minibatch_Y = minibatch
            sess.run(train_step,feed_dict={X:minibatch_X,Y:minibatch_Y,keep_prob:0.7})
        train_acc = sess.run(accurracy, feed_dict={X: X_train, Y: Y_train,keep_prob:1})
        test_acc = sess.run(accurracy, feed_dict={X: X_test, Y: Y_test,keep_prob:1})
        accurracy_list.append(sess.run(accurracy, feed_dict={X: X_test, Y: Y_test,keep_prob:1}))
        loss.append(sess.run(cross_entropy,feed_dict={X: X_train, Y: Y_train,keep_prob:1}))
        print('Iteration'+ str(epoch) +', Train accuracy: %.2f%%' %(train_acc*100) +', Testing accuracy: %.2f%%' %(test_acc*100))
        # saver.save(sess,'E:\\深度学习\\吴恩达深度学习课程\\代码\\卷积神经网络\\CNN Application\\model/')
ax1 = plt.subplot(211)
ax1.plot(loss,'r--')
plt.ylabel('Loss')
plt.title('Change of Training Loss')
ax2 = plt.subplot(212)
ax2.plot(accurracy_list,'b')
plt.xlabel('Iteration number')
plt.ylabel('Accurracy')
plt.title('Change of Training Accuracy')
plt.show()










