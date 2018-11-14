from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import pickle
import os
import glob
import scipy.misc
import numpy as np
import tensorflow as tf


FLAGS = None

test = {}
codes = {}
batch = {}

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


path = "/home/badri/Documents/cifar-10-batches-py/"

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
test_images, test_labels, test_encodings = load_test_data()

def preprocess_img(x):
    return 2 * x - 1.0

def deprocess_img(x):
    return (x + 1.0) / 2.0

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
			x1=tf.layers.dense(inputs=x,units=1024,activation=leaky_relu,use_bias=True)
			x2=tf.layers.dense(inputs=x1,units=512,activation=leaky_relu,use_bias=True)
			x3=tf.layers.dense(inputs=x2,units=512,activation=leaky_relu,use_bias=True)
			logits=tf.layers.dense(inputs=x3,units=1,activation=None,use_bias=True)
	return logits

def generator(z):
	with tf.name_scope("generator_data"):
		with tf.variable_scope("generator"):
			# fill in the required architecture in here
			x1=tf.layers.dense(inputs=z,units=1024,activation=tf.nn.relu,use_bias=True)
			x2=tf.layers.dense(inputs=x1,units=1024,activation=tf.nn.relu,use_bias=True)
			x3=tf.layers.dense(inputs=x2,units=2048,activation=tf.nn.relu,use_bias=True)
			img=tf.layers.dense(inputs=x3,units=3072,activation=tf.nn.tanh,use_bias=True)
			img1 = tf.reshape(img, [-1, img_size, img_size, 3])
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

def get_solvers(learning_rate=1e-3, beta1=0.9, beta2=0.999):
	with tf.name_scope("Train"):
		D_solver = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=beta1, beta2=beta2)
		G_solver = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=beta1, beta2=beta2)
	return D_solver, G_solver

tf.reset_default_graph()

batch_size = 64
noise_dim = 96

with tf.name_scope("inputs"):
	x = tf.placeholder(tf.float32, [None, img_size, img_size, num_channels])
	z = sample_gaussian_noise(batch_size, noise_dim)
with tf.name_scope("G_mapping"):
	G_sample = generator(z)

with tf.name_scope("real_fake"):
	with tf.variable_scope("") as scope:
		x_reshaped = tf.reshape(x, [-1, img_size*img_size*num_channels])
		logits_real = discriminator(preprocess_img(x_reshaped))
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


def run_gan(sess, G_train_step, G_loss, D_train_step, D_loss, G_extra_step, D_extra_step, batch_size=64, dim=96,\
			show_every=100, print_every=50, num_epochs=20):

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
	train_writer = tf.summary.FileWriter( "/home/badri/Desktop/cifar_data" , sess.graph)
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
