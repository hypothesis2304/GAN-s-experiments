from __future__ import print_function, division
import tensorflow as tf
import numpy as np

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

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./GANs/cs231n/datasets/MNIST_data', one_hot=False)

def leaky_relu(x, alpha=0.01):
    out = np.zeros_like(x)
    out = tf.maximum(x, 0)
    out = tf.where(tf.equal(out,0.0), alpha*x, out)
    return out

def sample_noise(batch_size, dim):
    z = tf.random_uniform(shape=[batch_size, dim], minval = -1, maxval=1)
    return z

def discriminator(x):
    with tf.variable_scope("discriminator"):
        x_image = tf.reshape(x, [-1, 28, 28, 1])
        
        conv1 = tf.layers.conv2d(inputs=x_image, filters=32, kernel_size=5, strides=1, activation=leaky_relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=2, strides=2)
        
        conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=5, strides=1, activation=leaky_relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=2, strides=2)
        
        [N, H, Wi, C] = pool2.get_shape().as_list()
        
        flatten_layer = tf.reshape(pool2, [-1, H*Wi*C])
        
        fc1 = tf.layers.dense(inputs=flatten_layer, units=4*4*64, activation=None)
        h1 = leaky_relu(fc1, alpha=0.01)
        
        fc2 = tf.layers.dense(inputs=h1, units=1, activation=None)
        logits =fc2
        
    return logits

def generator(z):
    with tf.name_scope("generator_block"):
        with tf.variable_scope("generator"):    
            fc1 = tf.layers.dense(inputs=z, units=1024, activation=tf.nn.relu)
            bnorm1 = tf.layers.batch_normalization(fc1, center=True, scale=True)
        
            fc2 = tf.layers.dense(inputs=bnorm1, units=7*7*128, activation=tf.nn.relu)
            bnorm2 = tf.layers.batch_normalization(fc2, center=True, scale=True)
       
            bnorm2_reshape = tf.reshape(bnorm2, [-1, 7, 7, 128])
            N, H, Wi, C = bnorm2_reshape.get_shape().as_list()
        
            deconv1 = tf.layers.conv2d_transpose(bnorm2_reshape, filters=64, kernel_size=4, strides=2, padding="same", activation=tf.nn.relu)
            bnorm = tf.layers.batch_normalization(deconv1, axis=3)
            deconv2 = tf.layers.conv2d_transpose(bnorm, filters=1, kernel_size=4, strides=2, padding="same", activation=tf.nn.tanh)
        
            N, H, W, C = deconv2.get_shape().as_list()
            img = tf.reshape(deconv2,[-1, H*W*C])

        tf.summary.image("generator_data", deconv2, max_outputs=5)
    return img

def gan_loss(logits_real, logits_fake):
    D_loss = None
    G_loss = None
    
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_fake), logits=logits_fake))
    D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_real), logits=logits_real))
    D_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(logits_fake), logits=logits_fake))
    
    return D_loss, G_loss

def get_solvers(learning_rate=1e-3, beta1=0.5, beta2=0.999):
    D_solver = None
    G_solver = None
    
    D_solver = tf.train.AdamOptimizer(learning_rate=1e-3, beta1=beta1, beta2=beta2)
    G_solver = tf.train.AdamOptimizer(learning_rate=1e-3, beta1=beta1, beta2=beta2)
    
    return D_solver, G_solver

tf.reset_default_graph()
batch_size = 128
noise_dim = 96
x = tf.placeholder(tf.float32, [None, 784])
z = sample_noise(batch_size, noise_dim)
G_sample = generator(z)

with tf.variable_scope("") as scope:
    #scale images to be -1 to 1
    logits_real = discriminator(preprocess_img(x))
    scope.reuse_variables()
    logits_fake = discriminator(G_sample)
    
# Get the list of variables for the discriminator and generator
D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator') 

D_solver, G_solver = get_solvers()
D_loss, G_loss = gan_loss(logits_real, logits_fake)

D_train_step = D_solver.minimize(D_loss, var_list=D_vars)
G_train_step = G_solver.minimize(G_loss, var_list=G_vars)
D_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'discriminator')
G_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'generator')


# a giant helper function
def run_a_gan(sess, G_train_step, G_loss, D_train_step, D_loss, G_extra_step, D_extra_step,\
              show_every=100, print_every=20, batch_size=128, num_epoch=100):
    
    max_iter = int(mnist.train.num_examples*num_epoch/batch_size)
    for it in range(max_iter):
        minibatch,minbatch_y = mnist.train.next_batch(batch_size)
        if it % show_every == 99:
            runoptions = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            summary, D, G, D_loss_curr, G_loss_curr = sess.run([merged, D_train_step, G_train_step, D_loss, G_loss], options=runoptions, run_metadata=run_metadata, feed_dict={x: minibatch})
            train_writer.add_run_metadata(run_metadata, "step%05d" % it)
            train_writer.add_summary(summary, it)   
            print("Adding run metadata for", it)
        else:
            summary, D, G, D_loss_curr, G_loss_curr = sess.run([merged, D_train_step, G_train_step, D_loss, G_loss], feed_dict={x: minibatch})
            train_writer.add_summary(summary, it)   
                
        if it % print_every == 0:
            print('Iter: {}, D: {:.4}, G:{:.4}'.format(it,D_loss_curr,G_loss_curr))
    return

with get_session() as sess:
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter( "./mnist2_summ1" , sess.graph)
    sess.run(tf.global_variables_initializer())
    run_a_gan(sess,G_train_step,G_loss,D_train_step,D_loss,G_extra_step,D_extra_step,num_epoch=100)
    train_writer.close()
