import tensorflow as tf 
IMAGE_SIZE=28#图像大小
NUM_CHANNELS=1#通道数
CONV1_SIZE=5#卷积窗口大小
CONV1_KERNEL_NUM=32#卷积核数
CONV2_SIZE=5
CONV2_KERNEL_NUM=64
FC_SIZE=512#隐层
OUTPUT_NODE=10#输出

def get_weight(shape,regularizer):
	w=tf.Variable(tf.truncated_normal(shape,stddev=0.1))
	if regularizer!=None: tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(regularizer)(w))#l2范式
	return w

def get_bias(shape):
	b=tf.Variable(tf.zeros(shape))
	return b

def conv2d(x,w):
	return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')#使用零填充

def max_pool_2x2(x):
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def forward(x,train,regularizer):
	conv1_w=get_weight([CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_KERNEL_NUM],regularizer)#5x5x1x32
	conv1_b=get_bias([CONV1_KERNEL_NUM])#32
	conv1=conv2d(x,conv1_w)#28x28x32
	relu1=tf.nn.relu(tf.nn.bias_add(conv1,conv1_b))#离散化
	pool1=max_pool_2x2(relu1)#14x14x32

	conv2_w=get_weight([CONV2_SIZE,CONV2_SIZE,CONV1_KERNEL_NUM,CONV2_KERNEL_NUM],regularizer)#5x5x32x64
	conv2_b=get_bias([CONV2_KERNEL_NUM])#64
	conv2=conv2d(pool1,conv2_w)#14x14x64
	relu2=tf.nn.relu(tf.nn.bias_add(conv2,conv2_b))
	pool2=max_pool_2x2(relu2)#7x7x64
	#拉直[batch,7x7x64]
	pool_shape=pool2.get_shape().as_list()
	nodes=pool_shape[1]*pool_shape[2]*pool_shape[3]
	reshaped=tf.reshape(pool2,[pool_shape[0],nodes])
	#全连接
	fc1_w=get_weight([nodes,FC_SIZE],regularizer)
	fc1_b=get_bias([FC_SIZE])
	fc1=tf.nn.relu(tf.matmul(reshaped,fc1_w)+fc1_b)
	if train:fc1=tf.nn.dropout(fc1,0.5)

	fc2_w=get_weight([FC_SIZE,OUTPUT_NODE],regularizer)
	fc2_b=get_bias([OUTPUT_NODE])
	y=tf.matmul(fc1,fc2_w)+fc2_b
	return y