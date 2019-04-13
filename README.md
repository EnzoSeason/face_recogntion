# face_recogntion

We use python (numpy, skimage, sklearn) to detect the human faces in the images.

## 1. 项目目标
1. 创建图像分类器

2. 创建人脸识别算法

3. 评估人脸识别结果

## 2. 项目进程
### 2019/4/13
#### 1. 已生成两个图像分类器
1. 使用**全部**训练集图像，已生成图像在不同方位，不同尺度上，人脸和非人脸的训练集数据

2. 使用HOG算法c生成特征

3. 使用LinearSVC生成二元分类器，输出-1对应非人脸图像，1对应人脸图像（模型在clf_hog_v1.pkl）

4. 使用CalibratedClassifierCV和LinearSVC生成二元分类器，输出两列数据，非人脸的概率和人脸概率 （模型在clf_proba_v1.pkl）

#### 2. 已完成简单的人脸识别算法
1. 在**一张图**上， 使用**尺寸不变**的k窗口检测人脸， 分数为**是人脸的概率**
2. 数据格数， **第一列表示图片编号**的格式不符合要求。
**注意！**，加**粗体**的地方是目前不符合项目要求的，需要修改。

#### 3. 未做评估人脸识别结果
我觉得我们需要做这个评估。 我们需要将训练集拆分成training set 和validation set。绘制 la courbe de précision/rappel. （见tp03)
