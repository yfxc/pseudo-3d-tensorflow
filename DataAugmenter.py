import numpy as np
import cv2
import random
#Class for data augmentation,including random flip,scale,vertical/horizon shift,changing brightness,histogram equalization.
class DataAugmenter():
    def __init__(self,isFlip=True,isShift=True,isScale=True,isBrightness=True,isHistogram_eq=True):
        #flipcode depends on your own dataset.
        self.tp=0
        self.angle_val=np.random.uniform(6,10)
        self.angle=np.random.choice([-self.angle_val,self.angle_val])
        self.scale=np.random.uniform(1.0,1.1)
        self.random_br=np.random.uniform(0.5,2.0)
        self.x_shift=None
        self.y_shift=None
    def Flip(self):
        if self.tp==0:
            self.input=cv2.flip(self.input,1)
        else:
            self.input=cv2.flip(self.input,0)
    def Shift(self):
        M=np.float32([[1,0,self.x_shift],[0,1,self.y_shift]])  
        self.input=cv2.warpAffine(self.input,M,(self.cols,self.rows))
        
    def Scale(self):
        M=cv2.getRotationMatrix2D((self.cols/2,self.rows/2),0,self.scale)
        self.input=cv2.warpAffine(self.input, M, (self.cols, self.rows))
        
    def Brightness(self):
        hsv=cv2.cvtColor(self.input,cv2.COLOR_RGB2HSV)
        mask=hsv[:,:,2] * self.random_br >255
        v_channel=np.where(mask,255,hsv[:,:,2] * self.random_br)
        hsv[:,:,2]=v_channel
        self.input=cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
        
    def Histogram(self):
        tpimg=cv2.cvtColor(self.input,cv2.COLOR_RGB2HSV)
        tpimg[:,:,2] = cv2.equalizeHist(tpimg[:,:,2])
        self.input=cv2.cvtColor(tpimg,cv2.COLOR_HSV2RGB)
    
    def Rotate(self): 
        M=cv2.getRotationMatrix2D((self.cols/2,self.rows/2),self.angle,1)
        self.input=cv2.warpAffine(self.input, M, (self.cols, self.rows)) 
        
    def show(self):
        self.input=Image.fromarray(self.input)
        self.input.show()
        self.input=np.array(self.input)
    
    def Apply(self,_Clip):
        res=[]
        #set the possibility of all measures of DA:
        # 50 % possibility for Flip , Rotate , Scale , Brightness changing , Histogram-equal ops.
        pro_flip=random.choice([0,1])
        pro_rotate=random.choice([0,1])
        #pro_shift=random.choice([0,1])
        pro_scale=random.choice([0,1])
        pro_bri=random.choice([0,1])
        pro_his=random.choice([0,1])
        # DO NOT USE SHIFT
        pro_shift=0 
        #reset new property of DA
        self.__init__()
        for pic in _Clip:
            self.input=pic
            self.rows,self.cols,_=pic.shape
            if pro_his:
                self.Histogram()   
                
            if pro_bri:
                self.Brightness()
                
            if pro_flip:  
                self.Flip()
            if pro_rotate:
                self.Rotate()
            if pro_shift:
                if self.x_shift==None:     
                    x=np.random.randint(self.cols/20,self.cols/16)
                    y=np.random.randint(self.rows/20,self.rows/16)
                    x_shift=random.choice([-x,x])
                    y_shift=random.choice([-y,y])
                    self.x_shift=x_shift
                    self.y_shift=y_shift
                self.Shift()
            if pro_scale:
                self.Scale()
            res.append(self.input)
        return res 
