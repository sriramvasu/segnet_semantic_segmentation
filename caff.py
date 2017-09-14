import caffe
import tensorflow as tf
import numpy as np
import os
from image_reader import *

# caffe.set_device(0)
prototxt_file='segnet_model_driving_webdemo.prototxt'
model_file='segnet_weights_driving_webdemo.caffemodel'
caffe.set_mode_cpu()
net=caffe.Net(prototxt_file,caffe.TEST,weights=model_file)
print net.blobs['data'].data.shape

train_data_dir='SegNet-Tutorial/CamVid/train/'
train_label_dir='SegNet-Tutorial/CamVid/trainannot/'
test_data_dir='SegNet-Tutorial/CamVid/test/'
test_label_dir='SegNet-Tutorial/CamVid/testannot/'

def trans_reshape(x,transposer=None,reshaper=None):
	if(transposer==None):
		transposer=range(len(x.shape))
	if(reshaper==None):
		reshaper=np.transpose(x,transposer).shape
	return np.transpose(x,transposer).reshape(reshaper)

batch_size=1
r=image_reader(train_data_dir,train_label_dir,batch_size)
image_size=r.image_size
num_classes=12
transposer=[0,3,1,2]
reshaper=[]
while(r.epoch==0):
	data=r.next_batch()
	net.blobs['data'].data[...]=np.transpose(data[0],transposer)
	net.forward()
	pred=net.blobs['argmax'].data.reshape([batch_size]+image_size[:-1])
	viz=np.zeros([pred.shape[0]]+image_size)
	colors=color_map(num_classes)
	for cl in range(num_classes):
		t=np.where(pred==cl)
		viz[t]=colors[cl,:]
	plt.figure(1)
	plt.imshow(viz[0,:])
	plt.figure(2)
	plt.imshow(data[1][0,:])
	plt.figure(3)
	plt.imshow(data[0][0,:])
	plt.show()
	plt.pause(0.05)