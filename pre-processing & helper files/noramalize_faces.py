from __future__ import print_function, division
import tensorflow as tf
import numpy as np


import argparse
import sys
import scipy.misc
import pickle
import os
import glob
import numpy as np
import tensorflow as tf

num_channels = 3
img_size = 128
img_size_flat = img_size * img_size * num_channels

path = "/home/krishnasumanth/faces_dataset/"

train_images = []
id = 0
os.chdir(path)

for file in os.listdir(path):
    train_images.append(scipy.misc.imread(file))

train_images = np.array(train_images)   
print(train_images.shape)

def pre_process(x):
    x = x/127.5
    return (x-1)/2

def deprocess(x):
    x+1/2.0
    return x*127.5 


def  next_batch(num, data):
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    return np.asarray(data_shuffle)
    
def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session

def leaky_relu(x, alpha=0.2):
    return tf.maximum(x,0.0)+alpha*tf.minimum(x,0.0)

def sample_noise(batch_size, dim):
    
    shape=np.array([batch_size,dim]) 
    #return tf.random_uniform(shape,minval=-1,maxval=1)
    return tf.random_normal(shape)

def gan_loss(logits_real, logits_fake):
    D_loss = None
    G_loss = None

    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_fake), logits=logits_fake))
    D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_real), logits=logits_real))
    D_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(logits_fake), logits=logits_fake))
       
    return D_loss, G_loss

def get_solvers(learning_rate=2e-4, beta1=0.5,beta2=0.999):
    
    D_solver = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=beta1,beta2=beta2)
    G_solver = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=beta1,beta2=beta2)
    
    return D_solver, G_solver

def run_a_gan(sess, G_train_step, G_loss, D_train_step, D_loss, G_extra_step, D_extra_step,\
              show_every=250, store_every=10000, print_every=50, batch_size=128, num_epoch=200):
    
    # print("check1")
    max_iter = int(train_images.shape[0]*num_epoch/batch_size)
    times=1
    for it in range(max_iter):
        minibatch = next_batch(batch_size, train_images)
        # print(type(minibatch))
        # print(minibatch.shape)
        if it % store_every == 9999:
            runoptions = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            
            _, D_loss_curr = sess.run([D_train_step, D_loss], feed_dict={x: minibatch})
            _, G_loss_curr = sess.run([G_train_step, G_loss])
            
            summary = sess.run(merged, options=runoptions, run_metadata=run_metadata)

            print("saving model: ")
            print()

            saver.save(sess, "checkpoint_" + str(times*10) + "000_mnist")
            times += 1

            train_writer.add_run_metadata(run_metadata, "step%05d" % it)
            train_writer.add_summary(summary, it)   
            
            print("Adding run metadata for", it)
        
        elif it % show_every == 249:
            # print("Adding summary: :) ")
            _, D_loss_curr = sess.run([D_train_step, D_loss], feed_dict={x: minibatch})
            _, G_loss_curr = sess.run([G_train_step, G_loss])
            
            summary = sess.run(merged)
            train_writer.add_summary(summary, it)
        else:
            # print("check2")
            _, D_loss_curr = sess.run([D_train_step, D_loss], feed_dict={x: minibatch})
            # print("check3")
            _, G_loss_curr = sess.run([G_train_step, G_loss])

        if it % print_every == 0:
            print('Iter: {}, D: {:.4}, G:{:.4}'.format(it,D_loss_curr,G_loss_curr))
    return

def discriminator(x):
    with tf.variable_scope("discriminator"):
        # TODO: implement architecture
        input_layer = tf.reshape(x, [-1, img_size, img_size, num_channels])

        conv1 = tf.layers.conv2d(inputs=input_layer,filters=64,kernel_size=[5, 5],strides=[2,2], padding="SAME") #64
        
        conv2 = tf.layers.conv2d(inputs=conv1,filters=128, kernel_size=[5, 5], strides=[2,2], padding="SAME") #32
        bnorm1 = tf.layers.batch_normalization(conv2)
        act1 = leaky_relu(bnorm1)
        
        conv3 = tf.layers.conv2d(inputs=act1,filters=256, kernel_size=[5, 5], strides=[2,2], padding="SAME") #16
        bnorm2 = tf.layers.batch_normalization(conv3)
        act2 = leaky_relu(bnorm2)

        conv4 = tf.layers.conv2d(inputs=act2,filters=256, kernel_size=[5, 5], strides=[2,2], padding="SAME") #8
        bnorm3 = tf.layers.batch_normalization(conv4)
        act3 = leaky_relu(bnorm3)      

        conv5 = tf.layers.conv2d(inputs=act3,filters=512, kernel_size=[5, 5], strides=[2,2], padding="SAME") #4
        bnorm4 = tf.layers.batch_normalization(conv5)
        act4 = leaky_relu(bnorm4)

        flatten = tf.contrib.layers.flatten(act4)
        
        logits = tf.layers.dense(inputs=flatten, units=1)
        #print(logits.get_shape())
        return logits

def generator(z):
    with tf.name_scope("gen_block"):
        with tf.variable_scope("generator"):


            full_2=tf.layers.dense(inputs=z, units=4*4*512)
            reshaped = tf.reshape(full_2, [-1, 4, 4, 512])
            batch_norm_2=tf.layers.batch_normalization(reshaped, axis=3)
            act2 = tf.nn.relu(batch_norm_2)
        
            deconv1 = tf.layers.conv2d_transpose(act2, filters=256, kernel_size=5, strides=2, padding="same")#8
            bnorm = tf.layers.batch_normalization(deconv1, axis=3)
            act3 = tf.nn.relu(bnorm)

            deconv2 = tf.layers.conv2d_transpose(act3, filters=128, kernel_size=5, strides=2, padding="same")#16
            bnorm1 = tf.layers.batch_normalization(deconv2, axis=3)
            act4 = tf.nn.relu(bnorm1)

            deconv3 = tf.layers.conv2d_transpose(act4, filters=128, kernel_size=5, strides=2, padding="same")#32
            bnorm2 = tf.layers.batch_normalization(deconv3, axis=3)
            act5 = tf.nn.relu(bnorm2)

            deconv4 = tf.layers.conv2d_transpose(act5, filters=64, kernel_size=5, strides=2, padding="same")#64
            bnorm3 = tf.layers.batch_normalization(deconv4, axis=3)
            act6 = tf.nn.relu(bnorm3)

            deconv5 = tf.layers.conv2d_transpose(act6, filters=3, kernel_size=5, strides=2, padding="same")#128
            deconv_final = tf.nn.tanh(deconv5)

        # print(img.get_shape())
            img=tf.reshape(deconv_final, [-1, img_size*img_size*num_channels])
            
            tf.summary.image("gen_block", deprocess(deconv_final), max_outputs=6)
        
            return img


tf.reset_default_graph()

batch_size = 128
# our noise dimension
noise_dim = 100

# placeholders for images from the training dataset
x = tf.placeholder(tf.float32, [None, img_size, img_size, num_channels])
z = sample_noise(batch_size, noise_dim)
# generated images
G_sample = generator(z)

with tf.variable_scope("") as scope:
    #scale images to be -1 to 1
    logits_real = discriminator(pre_process(x))
    # Re-use discriminator weights on new inputs
    scope.reuse_variables()
    logits_fake = discriminator(G_sample)
    
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
    train_writer = tf.summary.FileWriter( "./faces_summ2", sess.graph)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    run_a_gan(sess,G_train_step,G_loss,D_train_step,D_loss,G_extra_step,D_extra_step)