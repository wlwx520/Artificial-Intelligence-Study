import numpy as np
import tensorflow as tf
from data.cifar10 import cifar10_input
import time

max_steps=2000
batch_size=128
data_dir=''

def variabel_with_weight_loss(shape,stddev,w1):
    var = tf.Variable(tf.truncated_normal(shape,stddev=stddev))
    if w1 is not None:
        weight_loss=tf.multiply(tf.nn.l2_loss(var),w1,name='weight_loss')
        tf.add_to_collection('losses',weight_loss)
    return var

image_train,labels_train=cifar10_input.distorted_inputs(data_dir=data_dir,batch_size=batch_size)

image_test,labels_test=cifar10_input.inputs(eval_data=True,data_dir=data_dir,batch_size=batch_size)

x=tf.placeholder(tf.float32,[batch_size,24,24,3])
y_=tf.placeholder(tf.float32,[batch_size])

w1=variabel_with_weight_loss(shape=[5,5,3,64],stddev=5e-2,w1=0.0)
k1=tf.nn.conv2d(x,w1,padding='SAME')
b1=tf.Variable(tf.constant(0.0,shape=[64]))
c1=tf.nn.relu(tf.nn.bias_add(k1,b1))
p1=tf.nn.max_pool(c1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')
n1=tf.nn.lrn(p1,4,bias=1.0,alpha=0.001/9.0,beta=0.75)

w2=variabel_with_weight_loss(shape=[5,5,64,64],stddev=5e-2,w1=0.0)
k2=tf.nn.conv2d(n1,w2,padding='SAME')
b2=tf.Variable(tf.constant(0.0,shape=[64]))
c2=tf.nn.relu(tf.nn.bias_add(k2,b2))
n2=tf.nn.lrn(c2,bias=1.0,alpha=0.001/9.0,beta=0.75)
p2=tf.nn.max_pool(n2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')

reshape=tf.reshape(p2,[batch_size,-1])
dim=reshape.get_shape()[1].value

w3=variabel_with_weight_loss(shape=[dim,384],stddev=0.04,w1=0.004)
b3=tf.Variable(tf.constant(0.1,[384]))
l3=tf.nn.relu(tf.matmul(reshape,w3)+b3)

w4=variabel_with_weight_loss([384,192],stddev=0.04,w1=0.004)
b4=tf.Variable(tf.constant(0.1,[192]))
l4=tf.nn.relu(tf.matmul(l3,w4)+b4)

w5=variabel_with_weight_loss([192,10],stddev=1/192.0,w1=0)
b5=tf.constant(0.0,[10])
logits=tf.add(tf.matmul(l4,w5),b5)

def loss(logits,y_):
    y_=tf.cast(y_,tf.int64)
    cross=tf.nn.sparse_softmax_cross_entropy_with_logits(logits,y_,name='cross_per')
    cross_mean=tf.reduce_mean(cross,name='cross')
    tf.add_to_collection('losses',cross_mean)
    return tf.add_n(tf.get_collection('losses'),name='taotal_loss')

loss=loss(logits,y_)

train=tf.train.AdamOptimizer(1e-3).minimize(loss)

top_n=tf.nn.in_top_k(logits,y_,1)

sess=tf.InteractiveSession()

tf.global_variables_initializer().run()

tf.train.start_queue_runners()

for step in range(max_steps):
    image_batch,labels_batch=sess.run([image_train,labels_train])
    _,loss_value=sess.run([train],loss,feed_dict={x:image_batch,y_:labels_batch})








