from __future__ import print_function, division
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#%matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def show_images(images):
    images = np.reshape(images, [images.shape[0], -1])  # images reshape to (batch_size, D)
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.show(block=False)
        plt.imshow(img.reshape([sqrtimg,sqrtimg]))
    return

def preprocess_img(x):
    return 2 * x - 1.0

def deprocess_img(x):
    return (x + 1.0) / 2.0

def rel_error(x,y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def count_params():
    """Count the number of parameters in the current TensorFlow graph """
    param_count = np.sum([np.prod(x.get_shape().as_list()) for x in tf.global_variables()])
    return param_count


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session

#answers = np.load('gan-checks-tf.npz')

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/home/badri/Documents/courses/cs231n/assignment3_2017/assignment3/cs231n/datasets/MNIST_data', one_hot=False)

def leaky_relu(x, alpha=0.01):
    """Compute the leaky ReLU activation function.
    
    Inputs:
    - x: TensorFlow Tensor with arbitrary shape
    - alpha: leak parameter for leaky ReLU
    
    Returns:
    TensorFlow Tensor with the same shape as x
    """
    # TODO: implement leaky ReLU
    return tf.maximum(x,0.0)+alpha*tf.minimum(x,0.0)

def sample_noise(batch_size, dim):
    """Generate random uniform noise from -1 to 1.
    
    Inputs:
    - batch_size: integer giving the batch size of noise to generate
    - dim: integer giving the dimension of the the noise to generate
    
    Returns:
    TensorFlow Tensor containing uniform noise in [-1, 1] with shape [batch_size, dim]
    """
    # TODO: sample and return noise
    shape=np.array([batch_size,dim]) 
    return tf.random_uniform(shape,minval=-1,maxval=1)
    # return tf.random_shuffle(shape)
def discriminator(x):
    """Compute discriminator score for a batch of input images.
    
    Inputs:
    - x: TensorFlow Tensor of flattened input images, shape [batch_size, 784]
    
    Returns:
    TensorFlow Tensor with shape [batch_size, 1], containing the score 
    for an image being real for each input image.
    """
    
    with tf.variable_scope("discriminator"):
        # TODO: implement architecture
        x1=tf.layers.dense(inputs=x,units=256,activation=leaky_relu,use_bias=True)
        x2=tf.layers.dense(inputs=x1,units=256,activation=leaky_relu,use_bias=True)
        logits=tf.layers.dense(inputs=x2,units=1,activation=None,use_bias=True)
        logits_new = tf.nn.sigmoid(logits)
        return logits_new

def generator(z):
    """Generate images from a random noise vector.
    
    Inputs:
    - z: TensorFlow Tensor of random noise with shape [batch_size, noise_dim]
    
    Returns:
    TensorFlow Tensor of generated images, with shape [batch_size, 784].
    """
    with tf.variable_scope("generator"):
        # TODO: implement architecture
        x1=tf.layers.dense(inputs=z,units=1024,activation=tf.nn.relu,use_bias=True)
        x2=tf.layers.dense(inputs=x1,units=1024,activation=tf.nn.relu,use_bias=True)
        img=tf.layers.dense(inputs=x2,units=784,activation=tf.nn.tanh,use_bias=True)
        
        return img

x = tf.placeholder(tf.float32, [None, 784])
z = sample_noise(10, 96)
G_sample = generator(z)

with tf.variable_scope("") as scope:
    logits_real = discriminator(preprocess_img(x))
    scope.reuse_variables()
    logits_fake = discriminator(G_sample)
    
# print(x.get_shape.as_list())
mini, minbatch_y = mnist.train.next_batch(10)

with get_session() as sess:
    sess.run(tf.global_variables_initializer())
    # print("*****")
    _, o = sess.run([logits_real, logits_fake], feed_dict={x: mini})
    print("logits_Real")
    print(_)
    print("logits_fake")
    print(o)
