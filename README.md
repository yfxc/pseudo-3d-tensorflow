pseudo-3d-tensorflow
=========================
Tensorflow implement for pseudo-3d-residual-network.
****

|Author|yfxc|
|---|---
|E-mail|gez9487@163.com

****
## Introduction
Pseudo-3d-residual-network is mainly used for action recognition,paper url:
http://openaccess.thecvf.com/content_ICCV_2017/papers/Qiu_Learning_Spatio-Temporal_Representation_ICCV_2017_paper.pdf

Here is the tensorflow version.
## Preparing your own dataset.
-----------------
    Suppose you are about to use UCF dataset.Firstly converting videos to images is necessary.
    To do this,you could run codes like follows:(Suppose UCF dataset is in the same directory as the code-files.)
- ./process_video2image.sh UCF101   
    - And next step,you should get the train.list and test.list which you would fetch afterwards for training data and test
    - data individually:(number 5 indicates the possibility )
    - ./process_gettxt.sh UCF101 5
