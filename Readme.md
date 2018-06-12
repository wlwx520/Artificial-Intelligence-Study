# 目录</br>

[一、前言](#1)<br>
[二、Softmax Regression](#2)<br>
[三、去噪自编码器](#3)<br>
[四、多层感知机](#4)<br>
[五、简单的卷积网络](#5)<br>
[六、进阶的卷积网络](#6)<br>
[七、Adam优化算法](#7)<br>
[附、数学知识点补充](#8)<br>

<h1 id='1'>前言</h1>

&nbsp;&nbsp;&nbsp;&nbsp;在这个人工智能盛行的时代，希望有更多的小伙伴和我一起，在这条路上分享各自的经验。也请大牛们多多关照，您的宝贵意见一定能促使我成为一名优秀的人工智能开发者。<br>

<h1 id='2'>Softmax Regression</h1>

* **输入层：** mnist手写数字识别训练集-28*28只有灰度的图片数据展开成一维的向量784</br>
* **隐含层：** 激活函数为Relu，权重w1，偏执b1</br>
* **输出层：** 结果为0-9的数字，长度为10的向量。权重w2，偏执b2</br>
* **损失函数：** 交叉熵</br>
* **其他：** 使用L2正则化防止过度拟合，学习速率平滑递减提高学习效率</br>

* **附**</br>

&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://blog.csdn.net/u012162613/article/details/44261657">L12正则化：</a>就是为了让拟合函数的系数尽量小防止过度拟合，因为拟合函数需要照顾每一个训练集所以往往系数很多，通过正则化就可以让系数变小。在正则化后也会产生损失的，所以总的损失函数也需要加上该损失。</br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://blog.csdn.net/zchang81/article/details/70225220">tf.nn.softmax_cross_entropy_with_logits：</a>把softmax计算与cross entropy计算放到一起,提高计算效率。</br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://blog.csdn.net/zchang81/article/details/70225220">Sigmoid/Tanh/ReLU：</a>不同激活函数有各自的特定。</br>

<h1 id='3'>去噪自编码器</h1>

* **输入层：** 输入数据加上高斯噪音</br>
* **隐含层：** 激活函数为传入，权重w1，偏执b1。该节点数小于输入层节点数</br>
* **输出层：** 通过高阶特征重构源数据。权重w2，偏执b2</br>
* **损失函数：** 平方误差</br>

* **附**</br>

&nbsp;&nbsp;&nbsp;&nbsp;特点：希望输入与输出一致并且使用少量稀疏的高阶特征来重构自身。</br>
&nbsp;&nbsp;&nbsp;&nbsp;噪音：一般情况数据会加入噪音，在重复的学习后，无规律的噪音将被去除，这样我们就能从噪音中学习数据的特征。去噪自编码器中最常用的是<a href="https://blog.csdn.net/u012936765/article/details/53200918">加性高斯噪音自编码器</a>。</br>

<h1 id='4'>多层感知机</h1>

* **输入层：** mnist手写数字识别训练集-28*28只有灰度的图片数据展开成一维的向量784</br>
* **隐含层：** 激活函数relu，再按一定比例dropout</br>
* **输出层：** 使用softmax激活隐含层</br>
* **损失函数：** 交叉熵</br>

* **附**</br>

&nbsp;&nbsp;&nbsp;&nbsp;将一部分训练数据置为0，可避免在训练集上越来越精确，却在测试集上越来越差，整个网络使用dropout减轻过拟合，自适应学习速率Adagrad，Relu激活函数避免梯度弥散。</br>
&nbsp;&nbsp;&nbsp;&nbsp;Relu解决梯度弥散问题:用softmax函数时，在0附近函数梯度较大，而向两边越来越小，导致模型的训练中根据梯度更新权重时效率很低。而且Relu函数还有单侧抑制，稀疏激活等特点。</br>
&nbsp;&nbsp;&nbsp;&nbsp;Relu函数:<a href="https://www.zhihu.com/question/52020211?from=profile_question_card">单侧抑制性和稀疏激活性</a></br>
&nbsp;&nbsp;&nbsp;&nbsp;现在隐含层主要还是用Relu函数以及其变种（<a href="https://blog.csdn.net/u013146742/article/details/51986575">Leaky-ReLU、P-ReLU、R-ReLU等</a>）激活，对应输出层还是sigmoid函数，因为该函数在0-1，最接近概率分布</br>

<h1 id='5'>简单的卷积网络</h1>

* **输入层：** mnist手写数字识别训练集</br>
* **第一层卷积：** 卷积尺寸5*5，颜色1个通道，32个卷积核</br>
* **第二层卷积：** 卷积尺寸5*5，上层32核，64个卷积核</br>
* **隐含层：** 1024个节点，relu激活</br>
* **dropout层：** dropout系数0.5</br>
* **输出层：** 10个节点，softmax激活</br>
* **损失函数：** 交叉熵</br>
* **学习速率：** 1e-4，较小的学习速率</br>

* **附**</br>

&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://blog.csdn.net/bea_tree/article/details/51376577">卷积网络</a>是一种前馈神经网络，它的人工神经元可以响应一部分覆盖范围内的周围单元，对于大型图像处理有出色表现。它包括卷积层(convolutional layer)和池化层(pooling layer)。</br>

<h1 id='6'>进阶的卷积网络</h1>

* **训练集：** cifar10训练集</br>
* **LRN层：** 针对Relu函数这样的没边界的函数很有效，在卷积中挑选反应最大的</br>


* **附**</br>

&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://blog.csdn.net/yangdashi888/article/details/77918311">LRN层</a>对局部神经元的活动创建竞争机制，使得其中响应比较大的值变得相对更大，并抑制其他反馈较小的神经元，增强了模型的泛化能力。</br>

<h1 id='7'>Adam优化算法</h1>

现在，我们详细的了解下Adam算法，我们从以下几个方面来说明该算法。</br>
&nbsp;&nbsp;&nbsp;&nbsp;Adam 算法是什么，它为优化深度学习模型带来了哪些优势</br>
&nbsp;&nbsp;&nbsp;&nbsp;Adam 算法的原理机制是怎么样的，它与相关的 AdaGrad 和 RMSProp 方法有什么区别</br>
&nbsp;&nbsp;&nbsp;&nbsp;Adam 算法应该如何调参，它常用的配置参数是怎么样的</br>
&nbsp;&nbsp;&nbsp;&nbsp;Adam 算法如何实现</br>

* **什么是 Adam 优化算法？**</br>
Adam 是一种可以替代传统随机梯度下降过程的一阶优化算法，它能基于训练数据迭代地更新神经网络权重。Adam 优化算法应用在非凸优化问题中所获得的优势：</br>
&nbsp;&nbsp;&nbsp;&nbsp;* 直截了当地实现</br>
&nbsp;&nbsp;&nbsp;&nbsp;* 高效的计算</br>
&nbsp;&nbsp;&nbsp;&nbsp;* 所需内存少</br>
&nbsp;&nbsp;&nbsp;&nbsp;* 梯度对角缩放的不变性（第二部分将给予证明）</br>
&nbsp;&nbsp;&nbsp;&nbsp;* 适合解决含大规模数据和参数的优化问题</br>
&nbsp;&nbsp;&nbsp;&nbsp;* 适用于非稳态（non-stationary）目标</br>
&nbsp;&nbsp;&nbsp;&nbsp;* 适用于解决包含很高噪声或稀疏梯度的问题</br>
&nbsp;&nbsp;&nbsp;&nbsp;* 超参数可以很直观地解释，并且基本上只需极少量的调参</br>

* **Adam 优化算法的基本机制,它与相关的 AdaGrad 和 RMSProp 方法有什么区别**</br>
Adam 通过计算梯度的一阶矩估计和二阶矩估计而为不同的参数设计独立的自适应性学习率。接下来看看Adam和AdaGrad，RMSProp的区别</br>
&nbsp;&nbsp;&nbsp;&nbsp;* 适应性梯度算法（AdaGrad）为每一个参数保留一个学习率以提升在稀疏梯度（即自然语言和计算机视觉问题）上的性能。</br>
&nbsp;&nbsp;&nbsp;&nbsp;* 均方根传播（RMSProp）基于权重梯度最近量级的均值为每一个参数适应性地保留学习率。这意味着算法在非稳态和在线问题上有很有优秀的性能。</br>
&nbsp;&nbsp;&nbsp;&nbsp;* Adam 算法同时获得了 AdaGrad 和 RMSProp 算法的优点。Adam 不仅如 RMSProp 算法那样基于一阶矩均值计算适应性参数学习率，它同时还充分利用了梯度的二阶矩均值（即有偏方差/uncentered variance）。具体来说，算法计算了梯度的指数移动均值（exponential moving average），超参数 beta1 和 beta2 控制了这些移动均值的衰减率。移动均值的初始值和 beta1、beta2 值接近于 1（推荐值），因此矩估计的偏差接近于 0。该偏差通过首先计算带偏差的估计而后计算偏差修正后的估计而得到提升。</br>

* **Adam 的参数配置**</br>
&nbsp;&nbsp;&nbsp;&nbsp;* alpha：同样也称为学习率或步长因子，它控制了权重的更新比率（如 0.001）。较大的值（如 0.3）在学习率更新前会有更快的初始学习，而较小的值（如 1.0E-5）会令训练收敛到更好的性能</br>
&nbsp;&nbsp;&nbsp;&nbsp;* beta1：一阶矩估计的指数衰减率（如 0.9）</br>
&nbsp;&nbsp;&nbsp;&nbsp;* beta2：二阶矩估计的指数衰减率（如 0.999）。该超参数在稀疏梯度（如在 NLP 或计算机视觉任务中）中应该设置为接近 1 的数</br>
&nbsp;&nbsp;&nbsp;&nbsp;* epsilon：该参数是非常小的数，其为了防止在实现中除以零（如 10E-8）</br>

* **Adam 的实现**</br>

| |
| -------- |
|![](/image/image1.png)|

&nbsp;&nbsp;&nbsp;&nbsp;如图，在确定了参数α、β1、β2 和随机目标函数 f(θ) 之后，我们需要初始化参数向量、一阶矩向量、二阶矩向量和时间步。然后当参数θ没有收敛时，循环迭代地更新各个部分。即时间步 t 加 1、更新目标函数在该时间步上对参数θ所求的梯度、更新偏差的一阶矩估计和二阶原始矩估计，再计算偏差修正的一阶矩估计和偏差修正的二阶矩估计，然后再用以上计算出来的值更新模型的参数θ

<h1 id='8'>数学知识点补充</h1>
&nbsp;&nbsp;&nbsp;&nbsp;在这里也想大家补充一些<a href="/Math.md">数学知识</a>，主要从数学分析，高等代数，概率论，数据挖掘，信息论等方便讲述。
