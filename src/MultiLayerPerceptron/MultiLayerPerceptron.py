import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets('./../../data/mnist',one_hot=True)
sess=tf.InteractiveSession()

in_units=784
hidden_units=300

#隐含层权重为截断性正态分布，标准差为0.1,因为softmax的激活函数在0附近梯度最大最敏感，所以其他参数初始化为0
w1=tf.Variable(tf.truncated_normal([in_units,hidden_units],stddev=0.1))
b1=tf.Variable(tf.zeros([hidden_units]))
w2=tf.Variable(tf.zeros([hidden_units,10]))
b2=tf.Variable(tf.zeros([10]))

#输入层
x=tf.placeholder(tf.float32,[None,in_units])
#有多少数据不进行dropout
keep_prob=tf.placeholder(tf.float32)

#广播机制不需要tf.add，丢弃部分数据
hidden=tf.nn.relu(tf.matmul(x,w1)+b1)
hidden_drop=tf.nn.dropout(hidden,keep_prob)

#输出层用softmax激活
y=tf.nn.softmax(tf.matmul(hidden_drop,w2)+b2)

#期望
y_=tf.placeholder(tf.float32,[None,10])

#损失函数
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))

train_step=tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)

tf.global_variables_initializer().run()

for i in range(3000):
    batch_xs,batch_ys=mnist.train.next_batch(100)
    train_step.run({x:batch_xs,y_:batch_ys,keep_prob:0.75})

correct_prediction=tf.equal(tf.arg_max(y,1),tf.arg_max(y_,1))

accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

print(accuracy.eval({x:mnist.test.images,y_:mnist.test.labels,keep_prob:1}))



