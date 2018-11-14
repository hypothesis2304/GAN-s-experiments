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

num_channels = 3
img_size = 32
img_size_flat = img_size * img_size * num_channels
num_classes = 10
num_files_train = 5
images_per_file = 10000
num_images_train = num_files_train * images_per_file

def deprocess(image):
    return image*255.0

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
              show_every=250, store_every=10000, print_every=50, batch_size=128, num_epoch=500):
    
    max_iter = int(train_images.shape[0]*num_epoch/batch_size)
    times=1
    for it in range(max_iter):
        minibatch,minbatch_y = next_batch(batch_size, train_images, train_labels)
        
        if it % store_every == 9999:
            runoptions = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            
            _, D_loss_curr = sess.run([D_train_step, D_loss], feed_dict={x: minibatch})
            _, G_loss_curr = sess.run([G_train_step, G_loss])
            
            summary = sess.run(merged, options=runoptions, run_metadata=run_metadata)

            print("saving model: ")
            print()

            saver.save(sess, "checkpoint_" + str(times*10) + "000_cifar")
            times += 1

            train_writer.add_run_metadata(run_metadata, "step%05d" % it)
            train_writer.add_summary(summary, it)   
            
            print("Adding run metadata for", it)
        
        elif it % show_every == 0:
            _, D_loss_curr = sess.run([D_train_step, D_loss], feed_dict={x: minibatch})
            _, G_loss_curr = sess.run([G_train_step, G_loss])
            
            summary = sess.run(merged)
            train_writer.add_summary(summary, it)
        else:
            _, D_loss_curr = sess.run([D_train_step, D_loss], feed_dict={x: minibatch})
            _, G_loss_curr = sess.run([G_train_step, G_loss])

        if it % print_every == 0:
            print('Iter: {}, D: {:.4}, G:{:.4}'.format(it,D_loss_curr,G_loss_curr))
    return

def discriminator(x):
    with tf.variable_scope("discriminator"):
        # TODO: implement architecture
        input_layer = tf.reshape(x, [-1, 32, 32, 3])

        conv1 = tf.layers.conv2d(inputs=input_layer,filters=64,kernel_size=[3, 3],strides=[1,1], padding="SAME", kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))#32
        bnorm1 = tf.layers.batch_normalization(conv1, axis=3, momentum=0.9, epsilon=1e-5)
        act1 = leaky_relu(bnorm1)

        conv2 = tf.layers.conv2d(inputs=act1,filters=64,kernel_size=[3, 3],strides=[1,1], padding="SAME", kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))#32
        bnorm2 = tf.layers.batch_normalization(conv2, axis=3, momentum=0.9, epsilon=1e-5)
        act2 = leaky_relu(bnorm2)

        conv3 = tf.layers.conv2d(inputs=act2,filters=128, kernel_size=[3, 3], strides=[2,2], padding="SAME",  kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))#16
        bnorm3 = tf.layers.batch_normalization(conv3, axis=3, momentum=0.9, epsilon=1e-5)
        act3 = leaky_relu(bnorm3)

        conv4 = tf.layers.conv2d(inputs=act3,filters=128, kernel_size=[3, 3], strides=[2,2], padding="SAME",  kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))#8
        bnorm4 = tf.layers.batch_normalization(conv4, axis=3, momentum=0.9, epsilon=1e-5)
        act4 = leaky_relu(bnorm4)   

        conv5 = tf.layers.conv2d(inputs=act4,filters=256, kernel_size=[3, 3], strides=[2,2], padding="SAME",  kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))#4
        bnorm5 = tf.layers.batch_normalization(conv5, axis=3, momentum=0.9, epsilon=1e-5)
        act5 = leaky_relu(bnorm5)

        flatten = tf.contrib.layers.flatten(act5)
        
        logits = tf.layers.dense(inputs=flatten, units=1)

        return logits

def generator(z):
    with tf.name_scope("gen_block"):
        with tf.variable_scope("generator"):


            flat = tf.layers.dense(inputs=z, units=4*4*256)
            reshaped = tf.reshape(flat, [-1, 4, 4, 256])
            batch_norm=tf.layers.batch_normalization(reshaped, axis=3, momentum=0.9, epsilon=1e-5)
            act = tf.nn.relu(batch_norm)
        
            deconv1 = tf.layers.conv2d_transpose(act, filters=128, kernel_size=3, strides=2, padding="same",  kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))#8
            bnorm1 = tf.layers.batch_normalization(deconv1, axis=3, momentum=0.9, epsilon=1e-5)
            act1 = tf.nn.relu(bnorm1)

            deconv2 = tf.layers.conv2d_transpose(act1, filters=128, kernel_size=3, strides=1, padding="same",  kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))#8
            bnorm2 = tf.layers.batch_normalization(deconv2, axis=3, momentum=0.9, epsilon=1e-5)
            act2 = tf.nn.relu(bnorm2)

            deconv3 = tf.layers.conv2d_transpose(act2, filters=64, kernel_size=3, strides=2, padding="same",  kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))#16
            bnorm3 = tf.layers.batch_normalization(deconv3, axis=3, momentum=0.9, epsilon=1e-5)
            act3 = tf.nn.relu(bnorm3)

            deconv4 = tf.layers.conv2d_transpose(act3, filters=64, kernel_size=3, strides=1, padding="same",  kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))#16
            bnorm4= tf.layers.batch_normalization(deconv4, axis=3, momentum=0.9, epsilon=1e-5)
            act4 = tf.nn.relu(bnorm4)

            deconv5 = tf.layers.conv2d_transpose(act4, filters=3, kernel_size=3, strides=2, padding="same",  kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))#32
            deconv_final = tf.nn.tanh(deconv5)

            img=tf.reshape(deconv_final, [-1,img_size*img_size*num_channels])
            
            tf.summary.image("gen_block",deprocess(deconv_final), max_outputs=8)
        
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
    logits_real = discriminator(x)
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
    with tf.name_scope("original"):
        orig_images, orig_labels = next_batch(batch_size, train_new_images, train_labels)
        tf.summary.image("original", deprocess(orig_images), max_outputs=6)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter( "./cifar_deep", sess.graph)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    run_a_gan(sess,G_train_step,G_loss,D_train_step,D_loss,G_extra_step,D_extra_step)