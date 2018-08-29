import tensorflow as tf
import numpy as np
import os
import random
import PIL.Image as Image
import cv2
import os
import copy
import time
import P3D
import DataGenerator
from settings import *

import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--txt',type=str)

args=parser.parse_args()

IS_DA=True  #true if using data augmentation

def compute_loss(name_scope,logit,labels):
    cross_entropy_mean=tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logit))
    tf.summary.scalar(name_scope+'_cross_entropy',
                     cross_entropy_mean
                     )
    weight_decay_loss=tf.get_collection('weightdecay_losses')
    tf.summary.scalar(name_scope+'_weight_decay_loss',tf.reduce_mean(weight_decay_loss))
    total_loss=cross_entropy_mean+weight_decay_loss
    tf.summary.scalar(name_scope+'_total_loss',tf.reduce_mean(total_loss))
    return total_loss

def compute_accuracy(logit,labels):
    correct=tf.equal(tf.argmax(logit,1),labels)
    acc=tf.reduce_mean(tf.cast(correct,tf.float32))
    return acc

def run():
	MOVING_AVERAGE_DECAY=0.9
	MODEL_PATH=''
	USE_PRETRAIN=False
	MAX_STEPS=36000

	dataloader=DataGenerator.DataGenerator(filename=args.txt,
                                batch_size=BATCH_SIZE,
                                num_frames_per_clip=NUM_FRAMES_PER_CLIP,
                                shuffle=True,is_da=IS_DA)
	
	
	with tf.Graph().as_default():
		global_step=tf.get_variable('global_step',[],initializer=tf.constant_initializer(0),trainable=False)
		
		input_placeholder=tf.placeholder(tf.float32,shape=(BATCH_SIZE,NUM_FRAMES_PER_CLIP,CROP_SIZE,CROP_SIZE,RGB_CHANNEL))
		label_placeholder=tf.placeholder(tf.int64,shape=(BATCH_SIZE))
		
		
		logit=P3D.inference_p3d(input_placeholder,0.5,BATCH_SIZE)
		acc=compute_accuracy(logit,label_placeholder)
		tf.summary.scalar('accuracy',acc)
		loss=compute_loss('default_loss',logit,label_placeholder)
		
		
		varlist1=[]
		varlist2=[]
		for param in tf.trainable_variables():
		    if param.name!='dense/bias:0' and param.name!='dense/kernel:0':
		        varlist1.append(param)
		    else:
		        varlist2.append(param)
		#if necessary,you can set different learning rate for FC layers and the other individually.
		learning_rate_stable = tf.train.exponential_decay(0.0005,
		                                           global_step,decay_steps=2100,decay_rate=0.6,staircase=True)
		learning_rate_finetune = tf.train.exponential_decay(0.0005,
		                                           global_step,decay_steps=2100,decay_rate=0.6,staircase=True)
		
		opt_stable=tf.train.AdamOptimizer(learning_rate_stable)
		opt_finetuning=tf.train.AdamOptimizer(learning_rate_finetune)
		
		#when using BN,this dependecy must be built.
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) 
		optim_op1=opt_stable.minimize(loss,var_list=varlist1)
		optim_op2=opt_finetuning.minimize(loss,var_list=varlist2,global_step=global_step)
		
		with tf.control_dependencies(update_ops):
		    optim_op_group=tf.group(optim_op1,optim_op2)
		    
		
		
		variable_averages=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,num_updates=global_step)
		variable_averages_op=variable_averages.apply(tf.trainable_variables())
		
		train_op=tf.group(optim_op_group,variable_averages_op)
		
		#when using BN,only store trainable parameters is not enough,cause MEAN and VARIANCE for BN is not
		#trainable but necessary for test stage.
		saver=tf.train.Saver(tf.global_variables())
		init=tf.global_variables_initializer()
		sess=tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
		sess.run(init)
		if USE_PRETRAIN:
		    saver.restore(sess,MODEL_PATH)
		    print('Checkpoint reloaded.')
		else:
		    print('Training from scratch.')
		merged=tf.summary.merge_all()
		train_writer=tf.summary.FileWriter('./visual_logs/train',sess.graph)
		test_writer=tf.summary.FileWriter('./visual_logs/test',sess.graph)
		duration=0
		print('Start training.')
		for step in xrange(1,MAX_STEPS):
		    sess.graph.finalize()
		    start_time=time.time()
		    train_images,train_labels,_=dataloader.next_batch()
		    sess.run(train_op,feed_dict={
		                    input_placeholder:train_images,
		                    label_placeholder:train_labels})
		    duration+=time.time()-start_time
		    
		    
		    if step!=0 and step % 10==0:
		        curacc,curloss=sess.run([acc,loss],feed_dict={
		                    input_placeholder:train_images,
		                    label_placeholder:train_labels})
		        print('Step %d: %.2f sec -->loss : %.4f =====acc : %.2f' % (step, duration,np.mean(curloss),curacc))
		        duration=0
		    if step!=0 and step % 50==0:
		        mer=sess.run(merged,feed_dict={
		                    input_placeholder:train_images,
		                    label_placeholder:train_labels})
		        train_writer.add_summary(mer, step)
		    if step >7000 and step % 800==0 or (step+1)==MAX_STEPS:
		        saver.save(sess,'./TFCHKP_{}'.format(step),global_step=step)
		    
		print('done')   

if __name__=='__main__':
	print('Preparing for training,this may take several seconds.')
	run()        
        
    


