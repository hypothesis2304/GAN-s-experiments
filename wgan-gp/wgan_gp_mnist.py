from __future__ import print_function, division
import tensorflow as tf
import numpy as np


def preprocess_img(x):
    return 2 * x - 1.0

def deprocess_img(x):
    return (x + 1.0) / 2.0

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./GANs/cs231n/datasets/MNIST_data', one_hot=False)

def leaky_relu(x, alpha=0.2):
    
    return tf.maximum(x,0.0)+alpha*tf.minimum(x,0.0)

def sample_noise(batch_size, dim):
    
    shape=np.array([batch_size,dim]) 
    # return tf.random_uniform(shape,minval=-1,maxval=1)
    return tf.random_normal(shape)

def wgangp_loss(logits_real, logits_fake, batch_size, x, G_sample):
    
    D_loss = tf.reduce_mean(logits_fake) - tf.reduce_mean(logits_real)
    G_loss = -1.0*tf.reduce_mean(logits_fake)
    
    # lambda from the paper
    lam = 10
    
    # random sample of batch_size (tf.random_uniform)
    eps = tf.random_normal(shape=[batch_size, 1])
    
    x_hat = eps*x + (1 - eps)*G_sample

    with tf.variable_scope('',reuse=True) as scope:
        grad_D_x_hat = None
        grad_D_x_hat = tf.gradients(discriminator(x_hat), [x_hat])[0]
        print(grad_D_x_hat)
        grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grad_D_x_hat), reduction_indices=[1]))
        grad_pen = lam*tf.reduce_mean((grad_norm - 1.0)**2)

    
    D_loss += grad_pen

    return D_loss, G_loss
    
def get_solvers(learning_rate=1e-4, beta1=0, beta2=0.9):
    
    D_solver = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=beta1,beta2=beta2)
    G_solver = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=beta1,beta2=beta2)
    
    return D_solver, G_solver

def run_a_gan(sess, G_train_step, G_loss, D_train_step, D_loss, G_extra_step, D_extra_step,\
              show_every=500, print_every=50, store_every=5000, batch_size=64, num_epoch=100):
    max_iter = int(mnist.train.num_examples*num_epoch/batch_size)
    
    for it in range(max_iter):
        minibatch,minbatch_y = mnist.train.next_batch(batch_size)
        
        if it % store_every == 4999:
            
            runoptions = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            
            _, D_loss_curr = sess.run([D_train_step, D_loss], feed_dict={x: minibatch})
            _, G_loss_curr = sess.run([G_train_step, G_loss])
            summary = sess.run(merged, options=runoptions, run_metadata=run_metadata)
            
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
        input_layer = tf.reshape(x, [-1, 28, 28, 1])

        conv1 = tf.layers.conv2d(inputs=input_layer,filters=64,kernel_size=[5, 5],strides=[2,2], padding="SAME")
        act1 = leaky_relu(conv1)

        conv2 = tf.layers.conv2d(inputs=act1,filters=128, kernel_size=[5, 5], strides=[2,2], padding="SAME")
        bnorm2 = tf.layers.batch_normalization(conv2)
        act2 = leaky_relu(bnorm2)   

        flatten = tf.contrib.layers.flatten(act2)

        dense = tf.layers.dense(inputs=flatten, units=1024, activation=leaky_relu)

        logits = tf.layers.dense(inputs=dense, units=1)
        #print(logits.get_shape())
        return logits

def generator(z):
    with tf.name_scope("gen_block"):
        with tf.variable_scope("generator"):
           # TODO: implement architecture
            full_1=tf.layers.dense(inputs=z,units=1024)
            batch_norm_1=tf.layers.batch_normalization(full_1)
            act1 = tf.nn.relu(batch_norm_1)

            full_2=tf.layers.dense(inputs=act1,units=7*7*128)
            reshaped = tf.reshape(full_2, [-1, 7, 7, 128])
            batch_norm_2=tf.layers.batch_normalization(reshaped, axis=3)
            act2 = tf.nn.relu(batch_norm_2)
        
            deconv1 = tf.layers.conv2d_transpose(act2, filters=64, kernel_size=4, strides=2, padding="same")
            bnorm = tf.layers.batch_normalization(deconv1, axis=3)
            act3 = tf.nn.relu(bnorm)

            deconv2 = tf.layers.conv2d_transpose(act3, filters=1, kernel_size=4, strides=2, padding="same")
            deconv_final = tf.nn.tanh(deconv2)

            img=tf.reshape(deconv2,[-1,784])
            
            tf.summary.image("gen_block", deconv_final, max_outputs=8)
        
            return img 

tf.reset_default_graph()

batch_size = 64
# our noise dimension
noise_dim = 96

# placeholders for images from the training dataset
x = tf.placeholder(tf.float32, [None, 784])
z = sample_noise(batch_size, noise_dim)
# generated images
G_sample = generator(z)

with tf.variable_scope("") as scope:
    #scale images to be -1 to 1
    logits_real = discriminator(preprocess_img(x))
    # Re-use discriminator weights on new inputs
    scope.reuse_variables()
    print(G_sample.get_shape())
    logits_fake = discriminator(G_sample)
    
# Get the list of variables for the discriminator and generator
D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'discriminator')
G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'generator') 

D_solver,G_solver = get_solvers()

D_loss, G_loss = wgangp_loss(logits_real, logits_fake, batch_size, x, G_sample)

D_train_step = D_solver.minimize(D_loss, var_list=D_vars)
G_train_step = G_solver.minimize(G_loss, var_list=G_vars)
D_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS,'discriminator')
G_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS,'generator')

with get_session() as sess:
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter( "./wgangp_mnist_summ1", sess.graph)
    sess.run(tf.global_variables_initializer())
    run_a_gan(sess,G_train_step,G_loss,D_train_step,D_loss,G_extra_step,D_extra_step)