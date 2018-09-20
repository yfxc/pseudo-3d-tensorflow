import tensorflow as tf
from settings import *
def get_conv_weight(name,kshape,wd=0.0005):
    with tf.device('/cpu:0'):
        var=tf.get_variable(name,shape=kshape,initializer=tf.contrib.layers.xavier_initializer())
    if wd!=0:
        weight_decay = tf.nn.l2_loss(var)*wd
        tf.add_to_collection('weightdecay_losses', weight_decay)
    return var

def convS(name,l_input,in_channels,out_channels):
    return tf.nn.bias_add(tf.nn.conv3d(l_input,get_conv_weight(name=name,
                                                               kshape=[1,3,3,in_channels,out_channels]),
                                                               strides=[1,1,1,1,1],padding='SAME'),
                                              get_conv_weight(name+'_bias',[out_channels],0))
def convT(name,l_input,in_channels,out_channels):
    return tf.nn.bias_add(tf.nn.conv3d(l_input,get_conv_weight(name=name,
                                                               kshape=[3,1,1,in_channels,out_channels]),
                                                               strides=[1,1,1,1,1],padding='SAME'),
                                              get_conv_weight(name+'_bias',[out_channels],0))

#build the bottleneck struction of each block.
class Bottleneck():
    def __init__(self,l_input,inplanes,planes,stride=1,downsample='',n_s=0,depth_3d=47):
        
        self.X_input=l_input
        self.downsample=downsample
        self.planes=planes
        self.inplanes=inplanes
        self.depth_3d=depth_3d
        self.ST_struc=('A','B','C')
        self.len_ST=len(self.ST_struc)
        self.id=n_s
        self.n_s=n_s
        self.ST=list(self.ST_struc)[self.id % self.len_ST]
        self.stride_p=[1,1,1,1,1]
       
        if self.downsample!='':
            self.stride_p=[1,1,2,2,1]
        if n_s<self.depth_3d:
            if n_s==0:
                self.stride_p=[1,1,1,1,1]
        else:
            if n_s==self.depth_3d:
                self.stride_p=[1,2,2,2,1]
            else:
                self.stride_p=[1,1,1,1,1]
    #P3D has three types of bottleneck sub-structions.
    def ST_A(self,name,x):
        x=convS(name+'_S',x,self.planes,self.planes)
        x=tf.layers.batch_normalization(x,training=IS_TRAIN)
        x=tf.nn.relu(x)
        x=convT(name+'_T',x,self.planes,self.planes)
        x=tf.layers.batch_normalization(x,training=IS_TRAIN)
        x=tf.nn.relu(x)
        return x
    
    def ST_B(self,name,x):
        tmp_x=convS(name+'_S',x,self.planes,self.planes)
        tmp_x=tf.layers.batch_normalization(tmp_x,training=IS_TRAIN)
        tmp_x=tf.nn.relu(tmp_x)
        x=convT(name+'_T',x,self.planes,self.planes)
        x=tf.layers.batch_normalization(x,training=IS_TRAIN)
        x=tf.nn.relu(x)
        return x+tmp_x
    
    def ST_C(self,name,x):
        x=convS(name+'_S',x,self.planes,self.planes)
        x=tf.layers.batch_normalization(x,training=IS_TRAIN)
        x=tf.nn.relu(x)
        tmp_x=convT(name+'_T',x,self.planes,self.planes)
        tmp_x=tf.layers.batch_normalization(tmp_x,training=IS_TRAIN)
        tmp_x=tf.nn.relu(tmp_x)
        return x+tmp_x
    
    def infer(self):
        residual=self.X_input
        if self.n_s<self.depth_3d:
            out=tf.nn.conv3d(self.X_input,get_conv_weight('conv3_{}_1'.format(self.id),[1,1,1,self.inplanes,self.planes]),
                             strides=self.stride_p,padding='SAME')
            out=tf.layers.batch_normalization(out,training=IS_TRAIN)
            
        else:
            param=self.stride_p
            param.pop(1)
            out=tf.nn.conv2d(self.X_input,get_conv_weight('conv2_{}_1'.format(self.id),[1,1,self.inplanes,self.planes]),
                             strides=param,padding='SAME')
            out=tf.layers.batch_normalization(out,training=IS_TRAIN)
    
        out=tf.nn.relu(out)    
        if self.id<self.depth_3d:
            if self.ST=='A':
                out=self.ST_A('STA_{}_2'.format(self.id),out)
            elif self.ST=='B':
                out=self.ST_B('STB_{}_2'.format(self.id),out)
            elif self.ST=='C':
                out=self.ST_C('STC_{}_2'.format(self.id),out)
        else:
            out=tf.nn.conv2d(out,get_conv_weight('conv2_{}_2'.format(self.id),[3,3,self.planes,self.planes]),
                                  strides=[1,1,1,1],padding='SAME')
            out=tf.layers.batch_normalization(out,training=IS_TRAIN)
            out=tf.nn.relu(out)

        if self.n_s<self.depth_3d:
            out=tf.nn.conv3d(out,get_conv_weight('conv3_{}_3'.format(self.id),[1,1,1,self.planes,self.planes*BLOCK_EXPANSION]),
                             strides=[1,1,1,1,1],padding='SAME')
            out=tf.layers.batch_normalization(out,training=IS_TRAIN)
        else:
            out=tf.nn.conv2d(out,get_conv_weight('conv2_{}_3'.format(self.id),[1,1,self.planes,self.planes*BLOCK_EXPANSION]),
                             strides=[1,1,1,1],padding='SAME')
            out=tf.layers.batch_normalization(out,training=IS_TRAIN)
           
        if len(self.downsample)==1:
            residual=tf.nn.conv2d(residual,get_conv_weight('dw2d_{}'.format(self.id),[1,1,self.inplanes,self.planes*BLOCK_EXPANSION]),
                                  strides=[1,2,2,1],padding='SAME')
            residual=tf.layers.batch_normalization(residual,training=IS_TRAIN)
        elif len(self.downsample)==2:
            residual=tf.nn.conv3d(residual,get_conv_weight('dw3d_{}'.format(self.id),[1,1,1,self.inplanes,self.planes*BLOCK_EXPANSION]),
                                  strides=self.downsample[1],padding='SAME')
            residual=tf.layers.batch_normalization(residual,training=IS_TRAIN)
        out+=residual
        out=tf.nn.relu(out)
        
        return out

#build a singe block of p3d,depth_3d=47 means p3d-199
class make_block():
    def __init__(self,_X,planes,num,inplanes,cnt,depth_3d=47,stride=1):
        self.input=_X
        self.planes=planes
        self.inplanes=inplanes
        self.num=num
        self.cnt=cnt
        self.depth_3d=depth_3d
        self.stride=stride
        if self.cnt<depth_3d:
            if self.cnt==0:
                stride_p=[1,1,1,1,1]
            else:
                stride_p=[1,1,2,2,1]
            if stride!=1 or inplanes!=planes*BLOCK_EXPANSION:
                self.downsample=['3d',stride_p]
        else:
            if stride!=1 or inplanes!=planes*BLOCK_EXPANSION:
                self.downsample=['2d']
    def infer(self):
        x=Bottleneck(self.input,self.inplanes,self.planes,self.stride,self.downsample,n_s=self.cnt,depth_3d=self.depth_3d).infer()
        self.cnt+=1
        self.inplanes=BLOCK_EXPANSION*self.planes
        for i in range(1,self.num):
            x=Bottleneck(x,self.inplanes,self.planes,n_s=self.cnt,depth_3d=self.depth_3d).infer()
            self.cnt+=1
        return x

#build structure of the p3d network.
def inference_p3d(_X,_dropout,BATCH_SIZE):
    cnt=0
    conv1_custom=tf.nn.conv3d(_X,get_conv_weight('firstconv1',[1,7,7,RGB_CHANNEL,64]),strides=[1,1,2,2,1],padding='SAME')
    conv1_custom_bn=tf.layers.batch_normalization(conv1_custom,training=IS_TRAIN)
    conv1_custom_bn_relu=tf.nn.relu(conv1_custom_bn)
    x=tf.nn.max_pool3d(conv1_custom_bn_relu,[1,2,3,3,1],strides=[1,2,2,2,1],padding='SAME')
    b1=make_block(x,64,3,64,cnt)
    x=b1.infer()
    cnt=b1.cnt
   
    x=tf.nn.max_pool3d(x,[1,2,1,1,1],strides=[1,2,1,1,1],padding='SAME')
    
    b2=make_block(x,128,8,256,cnt,stride=2)
    x=b2.infer()
    cnt=b2.cnt
    x=tf.nn.max_pool3d(x,[1,2,1,1,1],strides=[1,2,1,1,1],padding='SAME')
    
    b3=make_block(x,256,36,512,cnt,stride=2)
    x=b3.infer()
    cnt=b3.cnt
    x=tf.nn.max_pool3d(x,[1,2,1,1,1],strides=[1,2,1,1,1],padding='SAME')
    
    shape=x.shape.as_list()
    x=tf.reshape(x,shape=[-1,shape[2],shape[3],shape[4]])
    
    x=make_block(x,512,3,1024,cnt,stride=2).infer()
    
    #Caution:make sure avgpool on the input which has the same shape as kernelsize has been setted padding='VALID'
    x=tf.nn.avg_pool(x,[1,5,5,1],strides=[1,1,1,1],padding='VALID')
    
    x=tf.reshape(x,shape=[-1,2048])
    if(IS_TRAIN):
        x=tf.nn.dropout(x,keep_prob=0.5)
    else:
        x=tf.nn.dropout(x,keep_prob=1)
    
    x=tf.layers.dense(x,NUM_CLASS)
    
    return x
