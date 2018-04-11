<h1>前言</h1>
&nbsp;&nbsp;&nbsp;&nbsp;在这个人工智能盛行的时代，希望有更多的小伙伴和我一起，在这条路上分享各自的经验。也请大牛们多多关照，您的宝贵意见一定能促使我成为一名优秀的人工智能开发者。<br>

<h1>Softmax Regression</h1>
&nbsp;&nbsp;&nbsp;&nbsp;作为人工智能学院的小生，跨出了第一步，写了第一个模型。步长平滑衰减，选用信息熵作为损失函数再加上一次L2正则化，加入隐含层偏执成为一个正统的神经网络。(数学原理压力大的还请补补高数吧 — —！)

* **数据集**：github上下载的MNIST数据
* **源码**：源于tensorflow实战书中的第一个例子的扩展
* **笔记**

| | |
| -------- | -------- | 
| ![](./image/Softmax-Regression/Softmax-Regression-1.jpg)  | ![](./image/Softmax-Regression/Softmax-Regression-2.jpg)    |
| ![](./image/Softmax-Regression/Softmax-Regression-3.jpg)  | ![](./image/Softmax-Regression/Softmax-Regression-4.jpg)    | 

* **附**</br>

<a href="https://blog.csdn.net/u012162613/article/details/44261657">L12正则化：</a>就是为了让拟合函数的系数尽量小防止过度拟合，因为拟合函数需要照顾每一个训练集所以往往系数很多，通过正则化就可以让系数变小。在正则化后也会产生损失的，所以总的损失函数也需要加上该损失。</br>
<a href="https://blog.csdn.net/zchang81/article/details/70225220">tf.nn.softmax_cross_entropy_with_logits：</a>把softmax计算与cross entropy计算放到一起,提高计算效率。</br>
<a href="https://blog.csdn.net/zchang81/article/details/70225220">Sigmoid/Tanh/ReLU：</a>不同激活函数有各自的特定。</br>