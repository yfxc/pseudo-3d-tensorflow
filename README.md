pseudo-3d-tensorflow
=========================
Tensorflow implement for pseudo-3d-residual-network.
****

|Author|yfxc|
|---|---
|E-mail|1512165940@qq.com
|Tensorflow|1.10+

****
## Introduction
Pseudo-3d-residual-network is mainly used for action recognition,paper url:
http://openaccess.thecvf.com/content_ICCV_2017/papers/Qiu_Learning_Spatio-Temporal_Representation_ICCV_2017_paper.pdf

Here is the tensorflow version.
## Preparing your own dataset.
Suppose you are about to use UCF dataset.Firstly converting videos to images is necessary.
To do this,you could run codes like follows:(Suppose UCF-101 dataset is in the same directory as the code-files.)
- **./process_video2image.sh UCF101** 
- And next step,you should get the train.list and test.list which you would afterwards fetch from for training data and testing
data individually:(number ‘5’ indicates that one-fifth of all data is testing data.)
- **./process_gettxt.sh UCF101 5**

**Note that:Due to the fact that *Relative Path* of the video clips exist in 'train.list' and 'test.list',
So you must make sure that 'DataGenerator.py' and UCF-101 are in the same directory! or modify the codes by yourself.**
## Train or Eval model     
After getting your own data.You can run **python train.py --txt='./train.list'** to train model.
You can also train and test model in 'tf-p3d-train_eval.ipynb' with jupyter notebook.
## Others
- You could change some model settings in 'settings.py',except the options(called 'IS_DA') for whether or not to use data augmentation in 'train.py'.
- Changing the properties for data augmentation in 'DataAugmenter.py'
## Updates
- Use **tf.layers.batch_normalization(training=...)** instead of tf.contrib.layers.batch_norm(is_training=...) which may lead to wrong answers when testing.
