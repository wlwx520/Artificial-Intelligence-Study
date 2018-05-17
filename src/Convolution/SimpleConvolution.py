import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets('./../../data/mnist',one_hot=True)
sess=tf.InteractiveSession()

#权重附带标准差为0.1的截断正态分布的噪音
def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.1))

#偏执增加一点正值以免节点死亡
def bias_variable(shape):
    return tf.Variable(tf.constant(0.1,shape=shape))

#卷积
def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')

#池化
def max_pool_2_2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#开始搭建网络
x=tf.placeholder(tf.float32,[None,784])
y_=tf.placeholder(tf.float32,[None,10])
x_image=tf.reshape(x,[-1,28,28,1])

#第一层卷积,卷积尺寸5*5，颜色1个通道，32个卷积核
w_conv1=weight_variable([5,5,1,32])
b_conv1=bias_variable([32])
h_conv1=tf.nn.relu(conv2d(x_image,w_conv1)+b_conv1)
h_pool1=max_pool_2_2(h_conv1)

#第二层卷积,卷积尺寸5*5，输入为上层的32核，64个卷积核
w_conv2=weight_variable([5,5,32,64])
b_conv2=bias_variable([64])
h_conv2=tf.nn.relu(conv2d(h_pool1,w_conv2)+b_conv2)
h_pool2=max_pool_2_2(h_conv2)

#经过两次池化28*28变成了7*7，而且最后一次核数为64，我们给的下一个隐含层为1024个节点
h_pool2_out=tf.reshape(h_pool2,[-1,7*7*64])
w_fc1=weight_variable([7*7*64,1024])
b_fc1=bias_variable([1024])

#就第二次卷积池化后的用relu激活进入下一层
h_fc1=tf.nn.relu(tf.matmul(h_pool2_out,w_fc1)+b_fc1)

#减小过拟合，加入dropout
keep_prob=tf.placeholder(tf.float32)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

#然后再加入一层softmax进行概率化
w_fc2=weight_variable([1024,10])
b_fc2=bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop,w_fc2)+b_fc2)

#定义损失
cross=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_conv),reduction_indices=[1]))

#使用小的学习速率
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross)

correct_prediction=tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

tf.global_variables_initializer().run()

for i in range(20000):
    batch=mnist.train.next_batch(50)
    if i%100==0:
        train_acc=accuracy.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})
        print("step %d , trainning accuracy %g"%(i,train_acc))

    train_step.run(feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})

print("train over %g"%accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1}))