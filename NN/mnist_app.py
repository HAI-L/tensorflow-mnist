from PIL import Image
import tensorflow as tf 
import numpy as np 
import mnist_forward
import mnist_backward
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def pre_pic(picName):
	img=Image.open(picName)
	reIm=img.resize((28,28),Image.ANTIALIAS)#消除锯齿的方法
	im_arr=np.array(reIm.convert('L'))#转化为灰度图
	threshold=50
	#白底黑字 转化成黑底白字
	for i in range(28):
		for j in range(28):
			im_arr[i][j]=255-im_arr[i][j]
			if (im_arr[i][j]<threshold):
				im_arr[i][j]=0
			else:
				im_arr[i][j]=255

	nm_arr=im_arr.reshape([1,784])
	nm_arr=nm_arr.astype(np.float32)
	img_ready=np.multiply(nm_arr,1.0/255.0)#转化成0和1
	return img_ready
def restore_model(testPicArr):
	with tf.Graph().as_default() as g:
		x=tf.placeholder(tf.float32,[None,mnist_forward.INPUT_NODE])
		y=mnist_forward.forward(x,None)
		preValue=tf.argmax(y,1)

		ema=tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
		ema_restore=ema.variables_to_restore()
		saver=tf.train.Saver(ema_restore)

		with tf.Session() as sess:
				ckpt=tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
				if ckpt and ckpt.model_checkpoint_path:
					#加载模型
					saver.restore(sess,ckpt.model_checkpoint_path)
					preValue=sess.run(preValue,feed_dict={x:testPicArr})
					return preValue
				else:
					print('No checkpoint file found')
					return -1

def application():
	testNum=int(input('input the number of test pic tures:'))
	for i in range(testNum):
		testPic=input('the path of test picture:')
		testPicArr=pre_pic(testPic)
		preValue=restore_model(testPicArr)
		print('The prediction number is :',preValue[0])

def main():
	application()

if __name__ == '__main__':
	main()