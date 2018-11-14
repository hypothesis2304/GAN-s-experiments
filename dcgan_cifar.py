from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import pickle
import os
import glob
import numpy as np
import tensorflow as tf

num_channels = 3
img_size = 32
img_size_flat = img_size * img_size * num_channels
num_classes = 10
num_files_train = 5
images_per_file = 10000
num_images_train = num_files_train * images_per_file


def unpickle(file):
	with open(file, 'rb') as f:
		data = pickle.load(f, encoding='bytes')
	return data

def convert_images(image):
	image_float = np.array(image, dtype=float) / 255.0
	images = image_float.reshape([-1, num_channels, img_size, img_size])
	images = images.transpose([0, 2, 3, 1])
	return images

def load_data(filename):
	data = unpickle(filename)
	raw_images = data[b'data']
	label = np.array(data[b'labels'])
	images = convert_images(raw_images)
	return images,label


def one_hot_encoded(class_numbers, num_classes=None):
	if num_classes is None:
		num_classes = np.max(class_numbers) + 1
	return np.eye(num_classes, dtype=float)[class_numbers]

path = "./cifar-10-batches-py/"

def load_training_data():
	images = np.zeros(shape=[num_images_train, img_size, img_size, num_channels], dtype=float)
	cls = np.zeros(shape=[num_images_train], dtype=int)
	begin = 0
	for i in os.listdir(path):
		if i[0] == "d":
			newpath = os.path.join(path+'/'+str(i))
			images_batch, cls_batch = load_data(filename=newpath)
			num_images = len(images_batch)
			end = begin + num_images
			images[begin:end, :] = images_batch
			cls[begin:end] = cls_batch
			begin = end
	return images, cls, one_hot_encoded(class_numbers=cls, num_classes=num_classes)


def load_test_data():
	for i in os.listdir(path):
		if i == "test_batch":
			newpath = os.path.join(path+'/'+i)
			images, cls = load_data(filename=newpath)
	return images, cls, one_hot_encoded(class_numbers=cls, num_classes=num_classes)

def load_class_names():
	for i in os.listdir(path):
		if i == "batches.meta":
			newpath = os.path.join(path+'/'+i)
			raw = unpickle(filename=newpath)[b'label_names']
			names = [x.decode('utf-8') for x in raw]
	return names

train_images, train_labels, train_encodings = load_training_data()
train_new_images = train_images
test_images, test_labels, test_encodings = load_test_data()

for i in range(num_channels):
	t=train_images[:,:,:,i]
	t=np.reshape(t,(50000,1024))
	mean=np.mean(t,0,keepdims=True)
	std=np.sqrt(np.mean(np.square(t-mean), 0,keepdims=True))
	t -= mean
	t /= std
	train_images[:,:,:,i]= np.reshape(t,(50000,img_size,img_size))

def preprocess_img(x):  
	return 2 * x - 1.0

def deprocess_img(x):
	out = []
	x = (x + 1.0) / 2.0
	x = tf.reshape(x, [-1, img_size, img_size, num_channels])
	for i in range(num_channels):
		q=train_images[:,:,:,i]
		q=np.reshape(q,(50000,1024))
		mean=np.mean(q,0,keepdims=True)
		std=np.sqrt(np.mean(np.square(q-mean), 0,keepdims=True))

		t = x[:,:,:,i]
		t=tf.reshape(t,[-1,1024])
		t = t*std
		t += mean
		t = tf.reshape(t, [-1,img_size, img_size])
		out.append(t)
	x = tf.reshape(tf.stack(out), shape=(-1, img_size,img_size, num_channels))	
	return x

def  next_batch(num, data, labels):
	idx = np.arange(0, len(data))
	np.random.shuffle(idx)
	idx = idx[:num]
	data_shuffle = [data[i] for i in idx]
	lables_shuffle = [labels[i] for i in idx]

	return np.asarray(data_shuffle), np.asarray(lables_shuffle)


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session

def leaky_relu(x, alpha=0.01):
	return tf.maximum(x,0.0)+alpha*tf.minimum(x,0.0)

def sample_uniform_noise(batch_size, dim):
    shape=np.array([batch_size,dim]) 
    return tf.random_uniform(shape,minval=-1,maxval=1)

def sample_gaussian_noise(batch_size, dim):
	return tf.random_normal(shape=[batch_size, dim])

###############################################################

def discriminator(x):
	with tf.name_scope("discriminator"):
		with tf.variable_scope("discriminator"):		 
			input_layer = tf.reshape(x, [-1, 32, 32, 3])

			conv1 = tf.layers.conv2d(inputs=input_layer,filters=32,kernel_size=[5, 5],activation=leaky_relu)
			pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
			
			conv2 = tf.layers.conv2d(inputs=pool1,filters=64,kernel_size=[5, 5],activation=leaky_relu)
			pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
       		
			[N, H, Wi, C] = pool2.get_shape().as_list()
        	
			pool2_flat = tf.reshape(pool2, [-1, H*Wi*C])#pool2 is of shape(7,7,64)
			dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=leaky_relu)
			dense2 = tf.layers.dense(inputs=dense, units=256, activation=leaky_relu)
			logits=tf.layers.dense(inputs=dense2,units=1,activation=None,use_bias=True)
	return logits

def generator(z):
	with tf.name_scope("generator_data"):
		with tf.variable_scope("generator"):
			# fill in the required architecture in here
			full_1=tf.layers.dense(inputs=z,units=1024,activation=tf.nn.relu)
			batch_norm_1=tf.layers.batch_normalization(full_1)

			full_2=tf.layers.dense(inputs=batch_norm_1,units=2048,activation=tf.nn.relu)
			batch_norm_2=tf.layers.batch_normalization(full_2)

			full_3=tf.layers.dense(inputs=batch_norm_2,units=8*8*128,activation=tf.nn.relu)
			batch_norm_3=tf.layers.batch_normalization(full_3)

			input_layer = tf.reshape(batch_norm_2, [-1, 8, 8, 128])

			[N, H, Wi, C] = input_layer.get_shape().as_list()
        	
			deconv1 = tf.layers.conv2d_transpose(input_layer, filters=64, kernel_size=4, strides=2, padding="same", activation=tf.nn.relu)
			bnorm = tf.layers.batch_normalization(deconv1, axis=3)
        	
			img = tf.layers.conv2d_transpose(bnorm, filters=3, kernel_size=4, strides=2, padding="same", activation=tf.nn.tanh)
			img=tf.reshape(img,[-1,img_size*img_size*num_channels])
			img1 = deprocess_img(img)
	tf.summary.image("generated_data", img1)
	return img


def cross_entropy_loss(logits_real, logits_fake):
	with tf.name_scope("cross_entropy"): 
		with tf.name_scope("G_loss"):
			G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_fake), logits=logits_fake))
		tf.summary.scalar("G_loss", G_loss)
		with tf.name_scope("D_loss"):
			D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_real), logits=logits_real))
			D_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(logits_fake), logits=logits_fake))
		tf.summary.scalar("D_loss", D_loss)
	return D_loss, G_loss

def get_solvers(learning_rate=1e-4, beta1=0.5, beta2=0.999):
	with tf.name_scope("Train"):
		D_solver = tf.train.AdamOptimizer(learning_rate=1e-3, beta1=beta1, beta2=beta2)
		G_solver = tf.train.AdamOptimizer(learning_rate=1e-3, beta1=beta1, beta2=beta2)
	return D_solver, G_solver

tf.reset_default_graph()

batch_size = 64
noise_dim = 512

with tf.name_scope("inputs"):
	x = tf.placeholder(tf.float32, [None, img_size, img_size, num_channels])
	z = sample_gaussian_noise(batch_size, noise_dim)
with tf.name_scope("G_mapping"):
	G_sample = generator(z)

with tf.name_scope("real_fake"):
	with tf.variable_scope("") as scope:
		x_reshaped = tf.reshape(x, [-1, img_size*img_size*num_channels])
		#logits_real = discriminator(preprocess_img(x_reshaped))
		logits_real = discriminator(x_reshaped)
		scope.reuse_variables()
		logits_fake = discriminator(G_sample)
		
with tf.name_scope("variables"):
	D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
	G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')

with tf.name_scope("Calculating_loss"):
	D_optimizer, G_optimizer = get_solvers()
	D_loss, G_loss = cross_entropy_loss(logits_real, logits_fake)

D_train_step = D_optimizer.minimize(D_loss, var_list=D_vars)
G_train_step = G_optimizer.minimize(G_loss, var_list=G_vars)

D_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'discriminator')
G_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'generator')


def run_gan(sess, G_train_step, G_loss, D_train_step, D_loss, G_extra_step, D_extra_step, batch_size=64, dim=512,\
			show_every=100, print_every=20, num_epochs=200):

	iterations = int(train_images.shape[0] * num_epochs/batch_size)
	with tf.name_scope("running_GAN"):
		for it in range(iterations):
			with tf.name_scope("Get_batch"):
				imgs, Labels = next_batch(batch_size, train_images, train_labels)

			if it % show_every == 99:
				runoptions = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
				run_metadata = tf.RunMetadata()
				summary, D, G, D_loss_curr, G_loss_curr = sess.run([merged, D_train_step, G_train_step, D_loss, G_loss], options=runoptions, run_metadata=run_metadata, feed_dict={x: imgs})
				train_writer.add_run_metadata(run_metadata, "step%05d" % it)
				train_writer.add_summary(summary, it)	
				print("Adding run metadata for", it)
			else:
				summary, D, G, D_loss_curr, G_loss_curr = sess.run([merged, D_train_step, G_train_step, D_loss, G_loss], feed_dict={x: imgs})
				train_writer.add_summary(summary, it)	
				
			if it % print_every == 0:
				print('Iter: {}, D: {:.4}, G:{:.4}'.format(it,D_loss_curr,G_loss_curr))
	
		train_writer.close()		
	return 

with get_session() as sess:
	merged = tf.summary.merge_all()
	train_writer = tf.summary.FileWriter( "./tensorflow/dc_cifarsummary3" , sess.graph)
	sess.run(tf.global_variables_initializer())
	run_gan(sess, G_train_step, G_loss, D_train_step, D_loss, G_extra_step, D_extra_step)

##############################################################




# def main(_):
#   if tf.gfile.Exists(FLAGS.log_dir):
#     tf.gfile.DeleteRecursively(FLAGS.log_dir)
#   tf.gfile.MakeDirs(FLAGS.log_dir)
#   train()


# if __name__ == '__main__':
#   parser = argparse.ArgumentParser()
#   parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
#                       default=False,
#                       help='If true, uses fake data for unit testing.')
#   parser.add_argument('--max_steps', type=int, default=1000,
#                       help='Number of steps to run trainer.')
#   parser.add_argument('--learning_rate', type=float, default=0.001,
#                       help='Initial learning rate')
#   parser.add_argument('--dropout', type=float, default=0.9,
#                       help='Keep probability for training dropout.')
#   parser.add_argument(
#       '--data_dir',
#       type=str,
#       default='/tmp/tensorflow/mnist/input_data',
#       help='Directory for storing input data')
#   parser.add_argument(
#       '--log_dir',
#       type=str,
#       default='/tmp/tensorflow/mnist/logs/mnist_with_summaries',
#       help='Summaries log directory')
#   FLAGS, unparsed = parser.parse_known_args()
#   tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
