import numpy as np
import random
import PIL.Image as Image
import cv2
import os
import time
from DataAugmenter import DataAugmenter
from settings import *
DA=DataAugmenter()

#is_da:True if using data augmentation. 
def get_frames_data(filename,num_frames_per_clip=NUM_FRAMES_PER_CLIP,is_da=False):
    
    ret_arr=[]
    s_index=0
    for parent, dirnames,filenames in os.walk(filename):
        if(len(filenames)<num_frames_per_clip):
            print("Get invaild data!")
            return [],s_index
        filenames=sorted(filenames)
        s_index=random.randint(0,len(filenames)-num_frames_per_clip)
        for i in range(s_index,s_index+num_frames_per_clip):
            image_name=str(filename)+'/'+str(filenames[i])
            img=Image.open(image_name)
            img_data=np.array(img)
            ret_arr.append(img_data)
    if is_da:
        return DA.Apply(ret_arr),s_index
    else:
        return ret_arr,s_index

class DataGenerator:
	def __init__(self,filename,batch_size,num_frames_per_clip,shuffle=True,crop_size=CROP_SIZE,is_da=False):
        
		self.index=0
		self.lines=open(filename,'r')
		self.lines=list(self.lines)
		self.len=len(self.lines)
		
		self.batch_size=batch_size
		self.num_frames_per_clip=num_frames_per_clip
		self.indexlist=[]
		self.crop_size=crop_size
		self.is_da=is_da
		if shuffle:
			self.video_indices=range(len(self.lines))
			random.seed(time.time())
			random.shuffle(self.video_indices)
		else:
			self.video_indices=range(0,len(self.lines))
	def next_batch(self):
		data=[]
		labels=[]
		crop_size=self.crop_size
		self.indexlist=[]
        
		if self.index + self.batch_size > self.len:
			self.index=0
		for index in self.video_indices[self.index:self.index+self.batch_size]:
			self.indexlist.append(index)
			line=self.lines[index].strip('\n').split()
			dirname=line[0]
			label=line[1]
			tmp_data,_=get_frames_data(dirname,self.num_frames_per_clip,self.is_da)
			img_datas=[]
			if(len(tmp_data)!=0):
				#first=True    
				for j in xrange(len(tmp_data)):
                    
					img=Image.fromarray(tmp_data[j].astype(np.uint8))  
                        
					if(img.width>img.height):
						scale=float(crop_size)/float(img.height)
						img=np.array(cv2.resize(np.array(img),(int(img.width*scale+1),crop_size))).astype(np.float32)      
					else:                    
						scale=float(crop_size)/float(img.width)
						img=np.array(cv2.resize(np.array(img),(crop_size,int(img.height*scale+1)))).astype(np.float32)
					crop_x=int((img.shape[0]-crop_size)/2)
					crop_y=int((img.shape[1]-crop_size)/2)
					img=img[crop_x:crop_x+crop_size,crop_y:crop_y+crop_size,:] #-np_mean[j] 
					img_datas.append(img)
					#if first:
						#t_img=img.astype(np.uint8)
						#Image.fromarray(t_img).show()
						#first=False
				data.append(img_datas)
				labels.append(int(label))
               
		self.index+=self.batch_size     
		return np.array(data).astype(np.float32),np.array(labels).astype(np.int64),self.indexlist
