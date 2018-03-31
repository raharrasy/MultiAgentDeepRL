import tensorflow as tf
import math

class ConvNet(object): 
	def __init__(self, initWidth, initHeight, initDepth):
		self.initWidth = initWidth
		self.initHeight = initHeight
		self.initDepth = initDepth
		self.x = tf.placeholder(tf.float32, [None,self.initWidth,self.initHeight,self.initDepth])
		self.y = tf.placeholder(tf.float32, [None, 8])
		self.x_target = tf.placeholder(tf.float32, [None,self.initWidth,self.initHeight,self.initDepth])
		self.weights =tf.placeholder(tf.float32,[None,1])

		self.W1 = tf.Variable(tf.truncated_normal([8, 8, 4, 32],mean=0,stddev=0.02)) 
		self.b1 = tf.Variable(tf.constant(0.05, shape=[32]))
		conv = tf.nn.conv2d(self.x, self.W1, strides=[1, 2, 2, 1], padding='SAME')
		conv_with_b = tf.nn.bias_add(conv, self.b1)
		conv_1 = tf.nn.relu(conv_with_b) 
		k = 2
		pool = tf.nn.max_pool(conv_1, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

		self.W2 = tf.Variable(tf.truncated_normal([4, 4, 32, 64],mean=0,stddev=0.02))
		self.b2 = tf.Variable(tf.constant(0.05, shape=[64]))

		conv2 = tf.nn.conv2d(pool, self.W2, strides=[1, 1, 1, 1], padding='SAME')
		conv_with_b2 = tf.nn.bias_add(conv2, self.b2)
		conv_3 = tf.nn.relu(conv_with_b2) 

		k = 2
		pool2 = tf.nn.max_pool(conv_3, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
        
		calculatedWidth = math.ceil(math.ceil(math.ceil(self.initWidth/2)/2)/2)
		calculatedHeight = math.ceil(math.ceil(math.ceil(self.initHeight/2)/2)/2)
		calculatedDepth = 64

		self.W3 = tf.Variable(tf.truncated_normal([calculatedWidth*calculatedHeight*calculatedDepth, 128],mean=0,stddev=0.02))
		self.b3 = tf.Variable(tf.constant(0.05, shape=[128]))

		self.W_out = tf.Variable(tf.truncated_normal([128, 8],mean=0,stddev=0.02)) 
		self.b_out = tf.Variable(tf.constant(0.05, shape=[8]))

		res_reshaped = tf.reshape(pool2, [-1, self.W3.get_shape().as_list()[0]])
		local = tf.add(tf.matmul(res_reshaped, self.W3), self.b3)
		local_out = tf.nn.relu(local)

		self.out = tf.add(tf.matmul(local_out, self.W_out), self.b_out)

		loss = tf.reduce_sum(tf.squared_difference(tf.multiply(self.y,self.weights),tf.multiply(self.out,self.weights)))
		self.train_op = tf.train.GradientDescentOptimizer(1).minimize(loss)

		self.W1_target = tf.Variable(self.W1.initialized_value())
		self.b1_target = tf.Variable(self.b1.initialized_value())
		conv_target = tf.nn.conv2d(self.x_target, self.W1_target, strides=[1, 2, 2, 1], padding='SAME')
		conv_with_b_target = tf.nn.bias_add(conv_target, self.b1_target)
		conv_1_target = tf.nn.relu(conv_with_b_target) 

		k = 2
		pool_target = tf.nn.max_pool(conv_1_target, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

		self.W2_target = tf.Variable(self.W2.initialized_value())
		self.b2_target = tf.Variable(self.b2.initialized_value())

		conv2_target = tf.nn.conv2d(pool_target, self.W2_target, strides=[1, 1, 1, 1], padding='SAME')
		conv_with_b2_target = tf.nn.bias_add(conv2_target, self.b2_target)
		conv_3_target = tf.nn.relu(conv_with_b2_target) 

		k = 2
		pool2_target = tf.nn.max_pool(conv_3_target, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

		self.W3_target = tf.Variable(self.W3.initialized_value())
		self.b3_target = tf.Variable(self.b3.initialized_value())

		self.W_out_target = tf.Variable(self.W_out.initialized_value())
		self.b_out_target = tf.Variable(self.b_out.initialized_value())

		res_reshaped_target = tf.reshape(pool2_target, [-1, self.W3_target.get_shape().as_list()[0]])
		local_target = tf.add(tf.matmul(res_reshaped_target, self.W3_target), self.b3_target)
		local_out_target = tf.nn.relu(local_target)

		self.out_target = tf.add(tf.matmul(local_out_target, self.W_out_target), self.b_out_target)

		self.W1_t_holder = tf.placeholder(tf.float32, [8,8,4,32])
		self.b1_t_holder = tf.placeholder(tf.float32, [32])
		self.W2_t_holder = tf.placeholder(tf.float32, [4, 4, 32, 64])
		self.b2_t_holder = tf.placeholder(tf.float32, [64])
		self.W3_t_holder = tf.placeholder(tf.float32, [calculatedWidth*calculatedHeight*calculatedDepth, 128])
		self.b3_t_holder = tf.placeholder(tf.float32, [128])
		self.W_out_t_holder = tf.placeholder(tf.float32, [128, 8])
		self.b_out_t_holder = tf.placeholder(tf.float32, [8])

		self.w1_t_op = self.W1_target.assign(self.W1_t_holder)
		self.b1_t_op = self.b1_target.assign(self.b1_t_holder)
		self.w2_t_op = self.W2_target.assign(self.W2_t_holder)
		self.b2_t_op = self.b2_target.assign(self.b2_t_holder)
		self.w3_t_op = self.W3_target.assign(self.W3_t_holder)
		self.b3_t_op = self.b3_target.assign(self.b3_t_holder)
		self.w_out_t_op = self.W_out_target.assign(self.W_out_t_holder)
		self.b_out_t_op = self.b_out_target.assign(self.b_out_t_holder)

		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())
		self.saver = tf.train.Saver(max_to_keep=500)

	def computeRes(self,observation):
		action_q_vals = self.sess.run(self.out, feed_dict={self.x: observation})
		return action_q_vals

	def learn(self, data_batch_input, data_batch_output):
		self.sess.run(self.train_op, feed_dict={self.x: data_batch_input, self.y: data_batch_output, self.weights: multiplier})

	def targetCompute(self,observation):
		action_q_vals = self.sess.run(self.out_target, feed_dict={self.x_target: observation})
		return action_q_vals

	def copyNetwork(self):
		a,b,c,d,e,f,g,h = self.sess.run([self.W1,self.b1,self.W2,self.b2,self.W3,self.b3,self.W_out,self.b_out])
		self.sess.run([self.w1_t_op,self.b1_t_op,self.w2_t_op,self.b2_t_op,self.w3_t_op,self.b3_t_op,self.w_out_t_op,self.b_out_t_op], feed_dict=
		{self.W1_t_holder : a, self.b1_t_holder : b, self.W2_t_holder : c, self.b2_t_holder : d, self.W3_t_holder : e, self.b3_t_holder : f,
		self.W_out_t_holder : g, self.b_out_t_holder :h})

	def checkpointing(self, filename, step = 0):
		save_path = self.saver.save(self.sess, filename, global_step = step)
