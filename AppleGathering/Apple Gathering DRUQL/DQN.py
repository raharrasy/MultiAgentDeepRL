import tensorflow as tf
import numpy as np

class ConvNet(object): 
	def __init__(self):
		self.x = tf.placeholder(tf.float32, [None,32,42,4])
		self.y = tf.placeholder(tf.float32, [None, 8])
		self.x_target = tf.placeholder(tf.float32, [None,32,42,4])
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

		self.W3 = tf.Variable(tf.truncated_normal([1536, 128],mean=0,stddev=0.02))
		self.b3 = tf.Variable(tf.constant(0.05, shape=[128]))

		self.W_out = tf.Variable(tf.truncated_normal([128, 8],mean=0,stddev=0.02)) 
		self.b_out = tf.Variable(tf.constant(0.05, shape=[8]))

		res_reshaped = tf.reshape(pool2, [-1, self.W3.get_shape().as_list()[0]])
		local = tf.add(tf.matmul(res_reshaped, self.W3), self.b3)
		local_out = tf.nn.relu(local)

		self.out = tf.add(tf.matmul(local_out, self.W_out), self.b_out)

		#loss = tf.metrics.mean_squared_error(self.y,self.out,weights=self.weights)
		loss = tf.reduce_mean(tf.squared_difference(tf.multiply(self.y,self.weights),tf.multiply(self.out,self.weights)))
		self.train_op = tf.train.GradientDescentOptimizer(1).minimize(loss)
		#self.var_grad = tf.gradients(loss, [self.W1,self.b1,self.W2,self.b2,self.W3,self.b3,self.W_out,self.b_out])

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

		self.W1_holder = tf.placeholder(tf.float32, [8,8,4,32])
		self.b1_holder = tf.placeholder(tf.float32, [32])
		self.W2_holder = tf.placeholder(tf.float32, [4, 4, 32, 64])
		self.b2_holder = tf.placeholder(tf.float32, [64])
		self.W3_holder = tf.placeholder(tf.float32, [1536, 128])
		self.b3_holder = tf.placeholder(tf.float32, [128])
		self.W_out_holder = tf.placeholder(tf.float32, [128, 8])
		self.b_out_holder = tf.placeholder(tf.float32, [8])

		self.w1_a_op = self.W1.assign(self.W1_holder)
		self.b1_a_op = self.b1.assign(self.b1_holder)
		self.w2_a_op = self.W2.assign(self.W2_holder)
		self.b2_a_op = self.b2.assign(self.b2_holder)
		self.w3_a_op = self.W3.assign(self.W3_holder)
		self.b3_a_op = self.b3.assign(self.b3_holder)
		self.w_out_a_op = self.W_out.assign(self.W_out_holder)
		self.b_out_a_op = self.b_out.assign(self.b_out_holder)

		self.W1_t_holder = tf.placeholder(tf.float32, [8,8,4,32])
		self.b1_t_holder = tf.placeholder(tf.float32, [32])
		self.W2_t_holder = tf.placeholder(tf.float32, [4, 4, 32, 64])
		self.b2_t_holder = tf.placeholder(tf.float32, [64])
		self.W3_t_holder = tf.placeholder(tf.float32, [1536, 128])
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
		self.saver = tf.train.Saver()

		#self.sess.run(tf.global_variables_initializer())


		# Make target network



	def computeRes(self,observation):
		#image_list = [get_image(file_path) for file_path in batch_files]
		#x_reshaped = tf.reshape(observation, shape=[-1, 32, 42, 4])
		#x_reshaped = tf.reshape(observation, shape=[-1, 32, 42, 4])
		action_q_vals = self.sess.run(self.out, feed_dict={self.x: observation})
		return action_q_vals

	def learn(self, data_batch_input, data_batch_output, multiplier):
		#W1,b1,W2,b2,W3,b3,W_out,b_out = self.sess.run([self.W1,self.b1,self.W2,self.b2,self.W3,self.b3,self.W_out,self.b_out])
		#sumGrads = None
		#for ii in range(len(data_batch_input)):
		#	grad = self.sess.run(self.var_grad, feed_dict={self.x: np.asarray([data_batch_input[ii]]), self.y: np.asarray([data_batch_output[ii]])})
		#	if ii != 0:
		#		grad2 = [np.add(a,np.multiply(b,multiplier[ii][0])) for a,b in zip(sumGrads,grad)]
		#	else:
		#		grad2 = [np.multiply(b,multiplier[ii][0]) for b in grad]
		#	grad = grad2 
		#	sumGrads = grad
		#W1 = np.subtract(W1,np.multiply(0.02,sumGrads[0]))
		#b1 = np.subtract(b1,np.multiply(0.02,sumGrads[1]))
		#W2 = np.subtract(W2,np.multiply(0.02,sumGrads[2]))
		#b2 = np.subtract(b2,np.multiply(0.02,sumGrads[3]))
		#W3 = np.subtract(W3,np.multiply(0.02,sumGrads[4]))
		#b3 = np.subtract(b3,np.multiply(0.02,sumGrads[5]))
		#W_out = np.subtract(W_out,np.multiply(0.02,sumGrads[6]))
		#b_out = np.subtract(b_out,np.multiply(0.02,sumGrads[7]))

		#self.sess.run([self.w1_a_op,self.b1_a_op,self.w2_a_op,self.b2_a_op,self.w3_a_op,self.b3_a_op,self.w_out_a_op,self.b_out_a_op], feed_dict=
		#	{self.W1_holder : W1, self.b1_holder : b1, self.W2_holder : W2, self.b2_holder : b2, self.W3_holder : W3, self.b3_holder : b3,
		#	self.W_out_holder : W_out, self.b_out_holder : b_out})

		self.sess.run(self.train_op, feed_dict={self.x: data_batch_input, self.y: data_batch_output, self.weights: multiplier})
	def targetCompute(self,observation):
		action_q_vals = self.sess.run(self.out_target, feed_dict={self.x_target: observation})
		return action_q_vals

	def copyNetwork(self):
		a,b,c,d,e,f,g,h = self.sess.run([self.W1,self.b1,self.W2,self.b2,self.W3,self.b3,self.W_out,self.b_out])
		self.sess.run([self.w1_t_op,self.b1_t_op,self.w2_t_op,self.b2_t_op,self.w3_t_op,self.b3_t_op,self.w_out_t_op,self.b_out_t_op], feed_dict={self.W1_t_holder : a, self.b1_t_holder : b, self.W2_t_holder : c, self.b2_t_holder : d, self.W3_t_holder : e, self.b3_t_holder : f, self.W_out_t_holder : g, self.b_out_t_holder :h})


	def save(self, filename):
		save_path = self.saver.save(self.sess, filename)
		print("Model saved in file: %s" % save_path)

	def checkpointing(self, filename):
		save_path = self.saver.save(self.sess, filename)
		self.sess.close()
		self.sess = tf.Session()
		self.saver.restore(self.sess, filename)


		
		#print("haha")


	# def model(): 
	# 	x_reshaped = tf.reshape(x, shape=[-1, 24, 24, 1])
	# 	conv_out1 = conv_layer(x_reshaped, W1, b1)
	# 	maxpool_out1 = maxpool_layer(conv_out1)
	# 	norm1 = tf.nn.lrn(maxpool_out1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
	# 	conv_out2 = conv_layer(norm1, W2, b2)
	# 	norm2 = tf.nn.lrn(conv_out2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
	# 	maxpool_out2 = maxpool_layer(norm2)
	# 	maxpool_reshaped = tf.reshape(maxpool_out2, [-1, W3.get_shape().as_list()[0]])
	# 	local = tf.add(tf.matmul(maxpool_reshaped, W3), b3) local_out = tf.nn.relu(local)
	# 	out = tf.add(tf.matmul(local_out, W_out), b_out)
	# 	return out

