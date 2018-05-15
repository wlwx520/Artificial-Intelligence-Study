import numpy as np
from sklearn import preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 定义标准分布的Xaiver初始化器
def xavier_init(fan_in,fan_out,constant=1):
    #这样的初始化器正好均值为0，方差为(max-min)的平方除以12
    low=-constant*np.sqrt(6.0/(fan_in+fan_out))
    high=constant*np.sqrt(6.0/(fan_in+fan_out))
    return tf.random_uniform((fan_in,fan_out),minval=low,maxval=high,dtype=tf.float32)

# 加性高斯自编码器
class AdditiveGaussianNoiseAuteEncoder(object):
    #定义输入层节点数，隐含层节点数，隐含层默认激活函数为softplus，优化器默认为Adam，高斯系数0.1
    def __init__(self,n_input,n_hidden,transfer=tf.nn.softplus,optimizer=tf.train.AdamOptimizer(),scale=0.1):
        self.n_input=n_input
        self.n_hidden=n_hidden
        #隐含层激活函数
        self.transfer=transfer
        #这里的scale将是在第一次训练传入的或者是训练后修正的
        self.scale=tf.placeholder(tf.float32)
        #传给下次训练修改后的scale
        self.training_scale=scale
        #网络的权重用之后定义的初始化权重函数完成
        self.weights=self.init_weights()

        #定义网络结构
        #输入层
        self.x=tf.placeholder(tf.float32,[None,self.n_input])

        #隐含层,将输入层加上高斯噪音,乘以隐含层权重加偏执，并使用传入的transfer激活函数进行处理
        self.hidden=self.transfer(tf.add(tf.matmul(
            self.x+scale*tf.random_normal((n_input)),
            self.weights['w1'],),self.weights['b1']))

        #重构层,不需要激活函数了，直接乘以输出层权重加偏执
        self.reconstruction=tf.add(tf.matmul(self.hidden,self.weights['w2']),self.weights['b2'])

        #损失函数是输入层和重构层的平方差
        self.cost=0.5*tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction,self.x),2.0))
        self.optimizer=optimizer.minimize(self.cost)
        init=tf.global_variables_initializer()
        self.sess=tf.Session()
        self.sess.run(init)

        #网络的定义到此结束，接下来是网络的权重的初始化，用一开始定义好的Xaiver初始化器
        def _init_weights(self):
            weights=dict()
            #隐含层权重用Xaiver初始化器，其他全初始化0即可
            weights['w1']=tf.Variable(xavier_init(self.n_input,self.n_hidden))
            weights['b1']=tf.Variable(tf.zeros([self.n_input],dtype=tf.float32))
            weights['w2']=tf.Variable(tf.zeros([self.n_hidden,self.n_input],dtype=tf.float32))
            weights['b2']=tf.Variable(tf.zeros(self.n_input,dtype=tf.float32))
            return weights

        #定义执行一步的训练函数以及返回损失cost
        def partial_fit(self,X):
            cost,opt=self.sess.run((self.cost,self.optimizer),feed_dict={self.x:X,self.scale:self.training_scale})
            return cost

        #定义只求损失函数不触发训练
        def calc_total_cost(self,X):
            return self.sess.run(self.cost,feed_dict={self.x:X,self.scale:self.training_scale})

        #定义获取数据在隐含层的高阶特征
        def transform(self,X):
            return self.sess.run(self.hidden, feed_dict={self.x: X, self.scale: self.training_scale})

        #定义用高阶特征还原数据
        def generate(self,hidden=None):
            if hidden is None:
                hidden=np.random.normal(size=self.weights['b1'])
            return self.sess.run(self.reconstruction, feed_dict={self.hidden:hidden})

        # 定义整个重构过程
        def reconstruct(self,X):
            return self.sess.run(self.reconstruction,feed_dict={self.x:X,self.scale:self.training_scale})

        #定义获取w1
        def getWeights(self):
            return self.sess.run(self.weights['w1'])

        # 定义获取b1
        def getBiases(self):
            return self.sess.run(self.weights['b1'])

#至此去噪自编码器类定义全部完成接下来将对数据进行标准化处理

#获取数据
mnist=input_data.read_data_sets('./../../data/mnist',one_hot=True)

#标准化即为均值为0，标准差为1的分布数据
#定义标准化函数
def standard_scale(X_train,X_test):
    p=prep.StandardScaler().fit(X_train)
    X_train = p.transform(X_train)
    X_test = p.transform(X_test)
    return X_train,X_test

#标准化数据
X_train,X_test = standard_scale(mnist.train.images,mnist.test.images)

#样本总数
n_samples=int(mnist.train.num_examples)
#训练轮数
training_epochs=20
#每轮训练数据条数
batch_size=128
#每1轮显示cost
display_step=1

#创建加性高斯去噪自编码器实例
autoencoder = AdditiveGaussianNoiseAuteEncoder(n_input=784,n_hidden=200,transfer=tf.nn.softplus,optimizer=tf.train.AdamOptimizer(learning_rate=0.001),scale=0.01)

#在开始训练前定义一个随机获取数据的方法，并且该方法是不放回的抽样
def get_randon(data,batch_size):
    index=np.random.randint(0,len(data)-batch_size)
    return data[index:(index+batch_size)]

#开始训练
for epoch in range(training_epochs):
    avg_cost=0.0
    total_batch=int(n_samples/batch_size)
    for i in range(total_batch):
        batch_xs=get_randon(X_train,batch_size)
        cost=autoencoder.patial_fit(batch_xs)
        avg_cost+=cost/n_samples*batch_size

    #每隔几轮打印损失
    if epoch%display_step==0:
        print("Epoch:",'%04d'%(epoch+1),"cost=","{:.9f}".format(avg_cost))

#使用测试集计算总的损失函数
print("Total cost = "+str(autoencoder.calc_total_cost(X_test)))
