import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data 
import mnist_lenet5_forward
import os
import numpy as np 
BATCH_SIZE=200
LEARNING_RATE_BASE=0.5#最初学习率
LEARNING_RATE_DECAY=0.99#学习率衰减率
REGULARIZER=0.0001#正则化系数
STEPS=50000
MOVING_AVERAGE_DECAY=0.99#滑动平均衰减率
MODEL_SAVE_PATH='./model/'
MODEL_NAME='mnist_model_cnn'

def backward(mnist):
	x=tf.placeholder(tf.float32,[BATCH_SIZE,
		mnist_lenet5_forward.IMAGE_SIZE,
		mnist_lenet5_forward.IMAGE_SIZE,
		mnist_lenet5_forward.NUM_CHANNELS
		])
	y_=tf.placeholder(tf.float32,[None,mnist_lenet5_forward.OUTPUT_NODE])
	y=mnist_lenet5_forward.forward(x,True,REGULARIZER)
	global_step=tf.Variable(0,trainable=False)
	#交叉熵
	ce=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
	cem=tf.reduce_mean(ce)#损失函数
	loss=cem+tf.add_n(tf.get_collection('losses'))#加正则化
	#指数衰减
	learning_rate=tf.train.exponential_decay(
		LEARNING_RATE_BASE,global_step,
		mnist.train.num_examples/BATCH_SIZE,
		LEARNING_RATE_DECAY,
		staircase=True)
	#梯度下降
	train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
	#滑动平均
	ema=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
	ema_op=ema.apply(tf.trainable_variables())
	with tf.control_dependencies([train_step,ema_op]):
		train_op=tf.no_op(name='train')

	saver=tf.train.Saver()#实例化saver对象

	with tf.Session() as sess:
		init_op=tf.global_variables_initializer()
		sess.run(init_op)
		#断点续训
		# ckpt=tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
		# if ckpt and ckpt.model_checkpoint_path:
		# 	#加载模型
		# 	saver.restore(sess,ckpt.model_checkpoint_path)

		for i in range(STEPS):
			xs,ys=mnist.train.next_batch(BATCH_SIZE)
			reshaped_xs=np.reshape(xs,[BATCH_SIZE,
				mnist_lenet5_forward.IMAGE_SIZE,
				mnist_lenet5_forward.IMAGE_SIZE,
				mnist_lenet5_forward.NUM_CHANNELS])
			_,loss_value,step=sess.run([train_op,loss,global_step],feed_dict={x:reshaped_xs,y_:ys})
			if i%1000==0:
				print('After%d training  step(s),loss on training batch is %g'% (step,loss_value))	
				saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)
def main():
	mnist=input_data.read_data_sets('./data',one_hot=True)
	backward(mnist)

if __name__ == '__main__':
	main()