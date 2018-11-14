from __future__ import print_function, division
import tensorflow as tf
import numpy as np

import argparse
import sys

import pickle
import os
import glob
import numpy as np
import tensorflow as tf

num_channels = 1
num_classes = 10
img_size = 28
y_dim = 10

def preprocess_img(x):
    return 2 * x - 1.0

def deprocess_img(x):
    return (x + 1.0) / 2.0

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./GANs/cs231n/datasets/MNIST_data', one_hot=True)

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session

def one_hot_encoded(class_numbers, num_classes=10):
    if num_classes is None:
        num_classes = np.max(class_numbers) + 1
    return np.eye(num_classes, dtype=float)[class_numbers]

def leaky_relu(x, alpha=0.2):
    return tf.maximum(x,0.0)+alpha*tf.minimum(x,0.0)

def sample_noise(batch_size, dim):
    
    shape=np.array([batch_size,dim]) 
    return tf.random_uniform(shape,minval=-1,maxval=1)
    # return tf.random_normal(shape)

def conv_concat(x, y):
    [xb, xw, xh, xc] = x.get_shape().as_list()
    [yb, yw, yh, yc] = y.get_shape().as_list()

    return tf.concat([x, tf.tile(tf.reshape(y, [-1, 1, 1, yc]), [1, xw, xh, 1])], 3)
    # return tf.concat([x, y*tf.ones_like([x_size])], 3)

def gan_loss(logits_real, logits_fake):
    
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_fake), logits=logits_fake))
    D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_real), logits=logits_real))
    D_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(logits_fake), logits=logits_fake))
    
    return D_loss, G_loss

def get_solvers(learning_rate=2e-4, beta1=0.5,beta2=0.9):
    
    D_solver = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=beta1,beta2=beta2)
    G_solver = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=beta1,beta2=beta2)
    
    return D_solver, G_solver

def run_a_gan(sess, G_train_step, G_loss, D_train_step, D_loss, G_extra_step, D_extra_step,\
              show_every=250, store_every=10000, print_every=50, batch_size=64, num_epoch=50):
    
    max_iter = int(mnist.train.num_examples*num_epoch/batch_size)
    # times=1
    for it in range(max_iter):
        minibatch,minibatch_y = mnist.train.next_batch(batch_size)
        
        if it % store_every == 9999:
            runoptions = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            
            _, D_loss_curr = sess.run([D_train_step, D_loss], feed_dict={x: minibatch, y: minibatch_y})      
            _, G_loss_curr = sess.run([G_train_step, G_loss], feed_dict={y: minibatch_y})
            
            summary = sess.run(merged, options=runoptions, run_metadata=run_metadata, feed_dict={y: minibatch_y})

            # saver.save(sess, "checkpoint_" + str(times*5) + "010_mnist")
            # times += 1

            # train_writer.add_run_metadata(run_metadata, "step%05d" % it)
            # train_writer.add_summary(summary, it)   
            
            print("Adding run metadata for", it)
        
        elif it % show_every == 0:

            _, D_loss_curr = sess.run([D_train_step, D_loss], feed_dict={x: minibatch, y: minibatch_y})
            _, G_loss_curr = sess.run([G_train_step, G_loss], feed_dict={y: minibatch_y})
            
            summary = sess.run(merged, feed_dict={y: minibatch_y})
            train_writer.add_summary(summary, it)
        
        else:
            
            _, D_loss_curr = sess.run([D_train_step, D_loss], feed_dict={x: minibatch, y: minibatch_y})
            _, G_loss_curr = sess.run([G_train_step, G_loss], feed_dict={y: minibatch_y})

        if it % print_every == 0:
            print('Iter: {}, D: {:.4}, G:{:.4}'.format(it,D_loss_curr,G_loss_curr))
    return

def discriminator(batch_size, x, y):
    with tf.variable_scope("discriminator"):
        # TODO: implement architecture

        yb = tf.reshape(y, [batch_size, 1, 1, y_dim])

        input_layer = tf.reshape(x, [-1, img_size, img_size, num_channels])

        # input_concat = conv_concat(input_layer, yb)

        conv1 = tf.layers.conv2d(inputs=input_layer,filters=64,kernel_size=[5, 5],strides=[2,2], padding="SAME", kernel_initializer=tf.truncated_normal_initializer(stddev=0.02,dtype=tf.float32))
        act1=leaky_relu(conv1)

        act1 = conv_concat(act1, yb)

        conv2 = tf.layers.conv2d(inputs=act1,filters=128, kernel_size=[5, 5], strides=[2,2], padding="SAME",kernel_initializer=tf.truncated_normal_initializer(stddev=0.02,dtype=tf.float32))
        bnorm1 = tf.layers.batch_normalization(conv2, axis=3, momentum=0.9, epsilon=1e-5,gamma_initializer=tf.random_normal_initializer(1., 0.02))
        act2 = leaky_relu(bnorm1)

        flatten = tf.contrib.layers.flatten(act2)
        
        logits = tf.layers.dense(inputs=flatten, units=1)
        #print(logits.get_shape())
        return logits

def generator(batch_size, z, y):
    with tf.name_scope("gen_block"):
        with tf.variable_scope("generator"):

            # yb = tf.reshape(y, [batch_size, 1, 1, y_dim]) # (64,1,1,10)

            z = tf.concat([z, y], 1) #(64,110)   

            flat = tf.layers.dense(inputs=z, units=7*7*128, kernel_initializer=tf.random_normal_initializer(stddev=0.02))
            reshaped = tf.reshape(flat, [-1, 7, 7, 128])
            bnorm = tf.layers.batch_normalization(reshaped, axis=3, momentum=0.9, epsilon=1e-5, gamma_initializer=tf.random_normal_initializer(1., 0.02))
            act = tf.nn.relu(bnorm)

            # act = conv_concat(act, yb)
        
            deconv1 = tf.layers.conv2d_transpose(act, filters=64, kernel_size=5, strides=2, padding="same")
            bnorm1 = tf.layers.batch_normalization(deconv1, axis=3, momentum=0.9, epsilon=1e-5,gamma_initializer=tf.random_normal_initializer(1., 0.02))
            act1 = tf.nn.relu(bnorm1)

            deconv2 = tf.layers.conv2d_transpose(act1, filters=num_channels, kernel_size=5, strides=2, padding="same")
            deconv_final = tf.nn.tanh(deconv2)

            img=tf.reshape(deconv_final, [-1,img_size*img_size*num_channels])
            
            tf.summary.image("gen_block", deprocess_img(deconv_final), max_outputs=8)
        
            return img

tf.reset_default_graph()

batch_size = 64
# our noise dimension
noise_dim = 100

# placeholders for images from the training dataset
x = tf.placeholder(tf.float32, [None, img_size*img_size*num_channels])
y = tf.placeholder(tf.float32, [None, y_dim])
z = sample_noise(batch_size, noise_dim)
# generated images
G_sample = generator(batch_size, z, y)

with tf.variable_scope("") as scope:
    #scale images to be -1 to 1
    logits_real = discriminator(batch_size, preprocess_img(x), y)
    # Re-use discriminator weights on new inputs
    scope.reuse_variables()
    logits_fake = discriminator(batch_size, G_sample, y)
    
# Get the list of variables for the discriminator and generator
D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'discriminator')
G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'generator') 

D_solver,G_solver = get_solvers()
D_loss, G_loss = gan_loss(logits_real, logits_fake)

D_train_step = D_solver.minimize(D_loss, var_list=D_vars)
G_train_step = G_solver.minimize(G_loss, var_list=G_vars)

D_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS,'discriminator')
G_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS,'generator')

with get_session() as sess:
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter( "./cgan3", sess.graph)
    sess.run(tf.global_variables_initializer())
    # saver = tf.train.Saver()
    run_a_gan(sess,G_train_step,G_loss,D_train_step,D_loss,G_extra_step,D_extra_step)
