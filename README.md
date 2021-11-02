# Caricature-Photo-Recognition

### 漫画人脸识别

任务：给出一张漫画，找到它对应同一个人的照片。（**C2P**）

此时探针（**probe**）是漫画，图库（**gallery**）是照片。

## 该代码运行于 Pytorch 1.8.1和CUDA11.2

#### 数据集文件简介：

* train: 训练集文件，里面有123个文件夹，每个文件夹对应一个人名字，是这个人的所有图片（漫画和照片，漫画C开头，照片P开头）
* test：测试集文件，里面有3038张图片，其中探针（probe，漫画）有2915张，就是对应我们需要提交的2915个结果；图库（gallery，照片）有123张，对应每个人一张。我们需要做的就是为这2915张漫画分别找到它对应的是123张照片中的哪一张。
* FR_Train_dev.txt：训练集文档，共126行，每一行是人名、该人的漫画数量和该人的照片数量。你可能会奇怪为什么123个人有126行？因为其中有三个人的数据在比赛中被屏蔽了（Xi 、Wen、Hu），你自己脑补是谁。
* FR_Probe_C2P.txt: 测试集中的探针文档，共2915行，每一行的数字对应的是test文件夹中的一张漫画ID，我们要按这个文档的顺序依次完成每一张漫画的匹配。
* FR_Gallery_C2P.txt：测试集中的图库文档，共123行，每一行的数字对应了test文件夹中的一张照片ID，我们在匹配探针漫画时就是在这123张照片中找到对应的。
* result.csv和submit_example.csv：结果提交文件，共2915行，每一行对应着FR_Probe_C2P.txt中每一张漫画的匹配结果，这个匹配结果就是图库照片的ID号。

#### 代码简介：

* dataset.py --读取train文件夹中的训练集，数据集加载代码. 

* net_sphere.py --使用SphereNet网络，网络定义文件.

* train_learning.py --最关键的py文件，训练我们的SphereNet网络. 

* utils.py --一些辅助用的小工具

* c2p_test.py --使用训练好的网络来测试，完成在test文件夹中的C2P任务，并保存要提交到比赛网站的结果CSV文件。


完整的数据集在这儿 [WebCaricature](https://cs.nju.edu.cn/rl/WebCaricature.htm)，但是比赛只允许我们使用小量进行人脸检测和对齐之后的数据。

### 基本思路：

基于 [SephereNet](http://www.cvlibs.net/publications/Coors2018ECCV.pdf)来提取图片（不管是漫画还是照片）的特征，训练时提取每一张图片的特征后进行分类，使用SephereNet中的AngleLoss损失，训练好之后，在测试时，给定一张探针（即probe，是一张漫画），用网络分别得到探针和每一张图库（即gallery，是一张照片）的特征相似度。选出相似度最大的照片作为该漫画的匹配结果。

### 改进方向：

1. 漫画域和照片域在算法中没有考虑区分，跨域识别还有很多可以利用的东西。比如漫画域和照片域分别使用不同的网络，再增加域之间的Loss。
2. 网络可以换成更新更适合的。
3. 没有充分利用训练集的特征。
4. 读的相关论文不够多，思路不够灵活。

code version 1.0 by HJC (from nju to ucas)

README.md by HJC (from nju to ucas)

